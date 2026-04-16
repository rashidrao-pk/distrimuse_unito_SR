import os
import glob
from pathlib import Path

import cv2
import numpy as np


def _ensure_gray(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _crop_with_mask(frame: np.ndarray, mask_gray: np.ndarray):
    if frame is None or mask_gray is None:
        return None, None

    mask_gray = _ensure_gray(mask_gray)

    if mask_gray.shape[:2] != frame.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, masked_full

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    return cropped, masked_full


def _resize_128(image: np.ndarray, keep_aspect: bool = True, target=(128, 128)) -> np.ndarray:
    target_w, target_h = target

    if image is None:
        return None

    if not keep_aspect:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _prepare_binary_mask(mask: np.ndarray, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _make_masked_input_visual(
    frame: np.ndarray,
    masks: list,
    blur_ksize: int = 31,
    dim_factor: float = 0.35,
    outline_color=(255, 255, 255),
    outline_thickness: int = 6
):
    """
    Create a visualization where masked safety areas remain sharp,
    while the outside region is blurred + dimmed.
    """
    h, w = frame.shape[:2]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    resized_masks = []

    for m in masks:
        if m is None:
            continue
        mask_bin = _prepare_binary_mask(m, (h, w))
        resized_masks.append(mask_bin)
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

    k = max(3, int(blur_ksize))
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    dimmed_blurred = np.clip(blurred.astype(np.float32) * dim_factor, 0, 255).astype(np.uint8)

    combined_mask_3 = cv2.merge([combined_mask, combined_mask, combined_mask])
    result = np.where(combined_mask_3 > 0, frame, dimmed_blurred)

    for mask_bin in resized_masks:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, outline_color, outline_thickness)

    return result


def _list_images(input_dir, recursive=True, extensions=None):
    if extensions is None:
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    image_paths = []
    input_dir = Path(input_dir)

    if recursive:
        for ext in extensions:
            image_paths.extend(input_dir.rglob(ext))
            image_paths.extend(input_dir.rglob(ext.upper()))
    else:
        for ext in extensions:
            image_paths.extend(input_dir.glob(ext))
            image_paths.extend(input_dir.glob(ext.upper()))

    image_paths = sorted(set(image_paths))
    return [str(p) for p in image_paths]


class SavedFramePreprocessor:
    def __init__(
        self,
        input_dir,
        save_dir,
        area_names,
        static_mask_paths,
        save_every_n=1,
        image_format="png",
        keep_aspect=True,
        save_masked_full=False,
        save_masked_input=True,
        masked_input_subdir="masked_input",
        masked_input_blur_ksize=31,
        masked_input_dim_factor=0.35,
        masked_input_outline_thickness=6,
        class_label="normal",
        recursive=True
    ):
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.area_names = area_names
        self.static_mask_paths = static_mask_paths
        self.save_every_n = int(save_every_n)
        self.image_format = str(image_format).lower()
        self.keep_aspect = bool(keep_aspect)
        self.save_masked_full = bool(save_masked_full)
        self.save_masked_input = bool(save_masked_input)
        self.masked_input_subdir = masked_input_subdir
        self.masked_input_blur_ksize = int(masked_input_blur_ksize)
        self.masked_input_dim_factor = float(masked_input_dim_factor)
        self.masked_input_outline_thickness = int(masked_input_outline_thickness)
        self.class_label = class_label
        self.recursive = recursive

        os.makedirs(self.save_dir, exist_ok=True)

        if len(self.area_names) != len(self.static_mask_paths):
            raise ValueError(
                f"Length mismatch: area_names={len(self.area_names)} vs static_mask_paths={len(self.static_mask_paths)}"
            )

        self.area_masks = {}
        for area_name, mask_path in zip(self.area_names, self.static_mask_paths):
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")

            mask = _ensure_gray(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            if mask is None:
                raise RuntimeError(f"Could not read mask for {area_name}: {mask_path}")

            self.area_masks[area_name] = mask
            os.makedirs(os.path.join(self.save_dir, area_name, self.class_label), exist_ok=True)

            if self.save_masked_full:
                os.makedirs(os.path.join(self.save_dir, area_name, "masked_full"), exist_ok=True)

            print(f"[INFO] Loaded mask for {area_name}: {mask_path}")

        if self.save_masked_input:
            os.makedirs(os.path.join(self.save_dir, self.masked_input_subdir), exist_ok=True)

    def process(self):
        image_paths = _list_images(self.input_dir, recursive=self.recursive)

        if len(image_paths) == 0:
            print(f"[WARNING] No images found in: {self.input_dir}")
            return

        print(f"[INFO] Found {len(image_paths)} images in: {self.input_dir}")
        print(f"[INFO] Saving processed outputs to: {self.save_dir}")
        print(f"[INFO] Save every N frames: {self.save_every_n}")

        processed_count = 0
        sampled_count = 0

        for idx, img_path in enumerate(image_paths, start=1):
            if idx % self.save_every_n != 0:
                continue

            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue

            sampled_count += 1

            stem = Path(img_path).stem

            # save combined masked_input visualization
            if self.save_masked_input:
                combined_visual = _make_masked_input_visual(
                    frame=frame,
                    masks=[self.area_masks[a] for a in self.area_names],
                    blur_ksize=self.masked_input_blur_ksize,
                    dim_factor=self.masked_input_dim_factor,
                    outline_thickness=self.masked_input_outline_thickness
                )

                masked_input_dir = os.path.join(self.save_dir, self.masked_input_subdir)
                masked_input_path = os.path.join(
                    masked_input_dir,
                    f"masked_input_{stem}.{self.image_format}"
                )

                ok = cv2.imwrite(masked_input_path, combined_visual)
                if not ok:
                    print(f"[ERROR] Failed saving masked input to {masked_input_path}")

            # save per-area cropped outputs
            for area_name in self.area_names:
                mask = self.area_masks[area_name]
                cropped, masked_full = _crop_with_mask(frame, mask)

                if cropped is None:
                    print(f"[WARNING] No valid crop for area {area_name} in {img_path}")
                    continue

                resized = _resize_128(
                    cropped,
                    keep_aspect=self.keep_aspect,
                    target=(128, 128)
                )

                out_dir = os.path.join(self.save_dir, area_name, self.class_label)
                out_path = os.path.join(
                    out_dir,
                    f"{area_name}_{stem}.{self.image_format}"
                )

                ok = cv2.imwrite(out_path, resized)
                if not ok:
                    print(f"[ERROR] Failed saving {out_path}")
                    continue

                if self.save_masked_full and masked_full is not None:
                    masked_dir = os.path.join(self.save_dir, area_name, "masked_full")
                    masked_path = os.path.join(
                        masked_dir,
                        f"masked_{area_name}_{stem}.{self.image_format}"
                    )
                    cv2.imwrite(masked_path, masked_full)

            processed_count += 1
            if processed_count % 20 == 0:
                print(f"[INFO] Processed and saved {processed_count} sampled images")

        print(f"[INFO] Done. Processed {processed_count} sampled images.")


import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--area_names", nargs="+", required=True)
    parser.add_argument("--static_mask_paths", nargs="+", required=True)

    parser.add_argument("--save_every_n", type=int, default=1)
    parser.add_argument("--image_format", default="png")
    parser.add_argument("--keep_aspect", type=bool, default=True)

    parser.add_argument("--save_masked_input", type=bool, default=True)
    parser.add_argument("--masked_input_subdir", default="masked_input")
    parser.add_argument("--masked_input_blur_ksize", type=int, default=31)
    parser.add_argument("--masked_input_dim_factor", type=float, default=0.35)
    parser.add_argument("--masked_input_outline_thickness", type=int, default=6)

    parser.add_argument("--class_label", default="normal")

    return parser.parse_args()


def main():
    args = parse_args()

    processor = SavedFramePreprocessor(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
        area_names=args.area_names,
        static_mask_paths=args.static_mask_paths,
        save_every_n=args.save_every_n,
        image_format=args.image_format,
        keep_aspect=args.keep_aspect,
        save_masked_input=args.save_masked_input,
        masked_input_subdir=args.masked_input_subdir,
        masked_input_blur_ksize=args.masked_input_blur_ksize,
        masked_input_dim_factor=args.masked_input_dim_factor,
        masked_input_outline_thickness=args.masked_input_outline_thickness,
        class_label=args.class_label,
    )

    processor.process()


if __name__ == "__main__":
    main()