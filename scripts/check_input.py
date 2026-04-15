import os
import time
import argparse
from collections import OrderedDict

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage, CompressedImage
from cv_bridge import CvBridge


ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]

AREA_DISPLAY_NAMES = {
    "PLeft": "Pallet Left",
    "PRight": "Pallet Right",
    "RoboArm": "Robo Arm",
    "ConvBelt": "Conveyor Belt",
}


def ordered_area_list(areas):
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def _ensure_gray(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _prepare_binary_mask(mask, frame_shape_hw):
    h, w = frame_shape_hw
    mask = _ensure_gray(mask)
    if mask is None:
        return None
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _extract_mask_contours(mask_gray, frame_shape_hw):
    mask_bin = _prepare_binary_mask(mask_gray, frame_shape_hw)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_bin


def _crop_with_mask(frame, mask_gray):
    mask_bin = _prepare_binary_mask(mask_gray, frame.shape[:2])
    masked_full = cv2.bitwise_and(frame, frame, mask=mask_bin)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, masked_full, mask_bin

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = masked_full[y_min:y_max + 1, x_min:x_max + 1]
    bbox = (x_min, y_min, x_max, y_max)
    return cropped, bbox, masked_full, mask_bin


def _resize_128(image, keep_aspect=True, target=(128, 128), return_meta=False):
    target_w, target_h = target

    if image is None:
        return (None, None) if return_meta else None

    if not keep_aspect:
        out = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        meta = {
            "new_w": target_w,
            "new_h": target_h,
            "x_off": 0,
            "y_off": 0,
            "target_w": target_w,
            "target_h": target_h,
            "orig_h": image.shape[0],
            "orig_w": image.shape[1],
        }
        return (out, meta) if return_meta else out

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return (None, None) if return_meta else None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    meta = {
        "new_w": new_w,
        "new_h": new_h,
        "x_off": x_off,
        "y_off": y_off,
        "target_w": target_w,
        "target_h": target_h,
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
    }

    return (canvas, meta) if return_meta else canvas


def create_union_mask(area_inputs, frame_shape_hw):
    h, w = frame_shape_hw
    union_mask = np.zeros((h, w), dtype=np.uint8)

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        mask_bin = info.get("mask_bin")
        if mask_bin is None:
            continue
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask = np.maximum(union_mask, mask_bin)

    return union_mask


def overlay_outside_safety_blur(frame_bgr, area_inputs, blur_ksize=31, darken_factor=0.35):
    if len(area_inputs) == 0:
        return frame_bgr.copy()

    union_mask = create_union_mask(area_inputs, frame_bgr.shape[:2])

    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    darkened = (blurred.astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)

    union_mask_3 = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    out = np.where(union_mask_3 > 0, frame_bgr, darkened)
    return out


def resize_and_center(image, target_w, target_h, bg_color=(0, 0, 0)):
    if image is None:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def scale_contours(contours, scale, x_off, y_off):
    scaled = []
    for cnt in contours:
        cnt_scaled = cnt.astype(np.float32).copy()
        cnt_scaled[:, 0, 0] = x_off + cnt_scaled[:, 0, 0] * scale
        cnt_scaled[:, 0, 1] = y_off + cnt_scaled[:, 0, 1] * scale
        scaled.append(cnt_scaled.astype(np.int32))
    return scaled


def draw_preprocessing_dashboard(frame_bgr, area_inputs, width=1600, height=1000):
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)

    pad = 16
    panel_w = (width - 3 * pad) // 2
    panel_h = (height - 3 * pad) // 2

    tl = (pad, pad, pad + panel_w, pad + panel_h)
    tr = (2 * pad + panel_w, pad, width - pad, pad + panel_h)
    bl = (pad, 2 * pad + panel_h, pad + panel_w, height - pad)
    br = (2 * pad + panel_w, 2 * pad + panel_h, width - pad, height - pad)

    def draw_panel_title(title, box):
        x1, y1, x2, y2 = box
        cv2.putText(canvas, title, (x1 + 12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

    draw_panel_title("Input Frame + Safety Areas", tl)
    draw_panel_title("Masked Full Frame", tr)
    draw_panel_title("Per-Area Crops", bl)
    draw_panel_title("Final 128x128 Model Inputs", br)

    inner_margin = 12
    title_h = 40

    def inner_box(box):
        x1, y1, x2, y2 = box
        return (x1 + inner_margin, y1 + title_h, x2 - inner_margin, y2 - inner_margin)

    tl_in = inner_box(tl)
    tr_in = inner_box(tr)
    bl_in = inner_box(bl)
    br_in = inner_box(br)

    h, w = frame_bgr.shape[:2]

    # -----------------------------
    # TOP-LEFT: input + contours
    # -----------------------------
    tl_w = tl_in[2] - tl_in[0]
    tl_h = tl_in[3] - tl_in[1]
    scale = min(tl_w / w, tl_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    inp = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_off = tl_in[0] + (tl_w - new_w) // 2
    y_off = tl_in[1] + (tl_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = inp

    for area_name in ordered_area_list(area_inputs.keys()):
        info = area_inputs[area_name]
        contours = info.get("contours", [])
        bbox = info.get("bbox")

        scaled = scale_contours(contours, scale, x_off, y_off)
        if len(scaled) > 0:
            cv2.drawContours(canvas, scaled, -1, (0, 255, 255), 2)
            pt = scaled[0][0][0]
            cv2.putText(canvas, AREA_DISPLAY_NAMES.get(area_name, area_name),
                        (int(pt[0]), max(20, int(pt[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rx1 = int(x_off + x1 * scale)
            ry1 = int(y_off + y1 * scale)
            rx2 = int(x_off + x2 * scale)
            ry2 = int(y_off + y2 * scale)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

    # -----------------------------
    # TOP-RIGHT: masked frame
    # -----------------------------
    masked_bg = overlay_outside_safety_blur(frame_bgr, area_inputs)
    tr_w = tr_in[2] - tr_in[0]
    tr_h = tr_in[3] - tr_in[1]
    masked_disp = resize_and_center(masked_bg, tr_w, tr_h, bg_color=(0, 0, 0))
    canvas[tr_in[1]:tr_in[1] + tr_h, tr_in[0]:tr_in[0] + tr_w] = masked_disp

    # -----------------------------
    # BOTTOM-LEFT: crops grid
    # -----------------------------
    crop_panel = np.full((bl_in[3] - bl_in[1], bl_in[2] - bl_in[0], 3), 245, dtype=np.uint8)
    draw_area_grid(crop_panel, area_inputs, key_name="crop")
    canvas[bl_in[1]:bl_in[3], bl_in[0]:bl_in[2]] = crop_panel

    # -----------------------------
    # BOTTOM-RIGHT: resized 128x128 inputs
    # -----------------------------
    model_panel = np.full((br_in[3] - br_in[1], br_in[2] - br_in[0], 3), 245, dtype=np.uint8)
    draw_area_grid(model_panel, area_inputs, key_name="resized", show_meta=True)
    canvas[br_in[1]:br_in[3], br_in[0]:br_in[2]] = model_panel

    return canvas


def draw_area_grid(panel, area_inputs, key_name="crop", show_meta=False):
    areas = ordered_area_list(area_inputs.keys())
    h, w = panel.shape[:2]

    if len(areas) == 0:
        cv2.putText(panel, "No active areas / no masks matched", (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        return panel

    cols = 2
    rows = 2
    pad = 12
    cell_w = (w - (cols + 1) * pad) // cols
    cell_h = (h - (rows + 1) * pad) // rows

    for idx, area_name in enumerate(areas[:4]):
        r = idx // cols
        c = idx % cols
        x1 = pad + c * (cell_w + pad)
        y1 = pad + r * (cell_h + pad)
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        cv2.rectangle(panel, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(panel, AREA_DISPLAY_NAMES.get(area_name, area_name),
                    (x1 + 8, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        img = area_inputs[area_name].get(key_name)
        if img is not None:
            disp_h = cell_h - 40
            disp_w = cell_w - 16
            disp = resize_and_center(img, disp_w, disp_h, bg_color=(0, 0, 0))
            panel[y1 + 32:y1 + 32 + disp_h, x1 + 8:x1 + 8 + disp_w] = disp

        if show_meta:
            meta = area_inputs[area_name].get("resize_meta")
            bbox = area_inputs[area_name].get("bbox")
            if meta is not None:
                text1 = f"new:{meta['new_w']}x{meta['new_h']} off:({meta['x_off']},{meta['y_off']})"
                cv2.putText(panel, text1, (x1 + 8, y2 - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                text2 = f"bbox: ({x_min},{y_min})-({x_max},{y_max})"
                cv2.putText(panel, text2, (x1 + 8, y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

    return panel


class CheckInputNode(Node):
    def __init__(self, args):
        super().__init__("check_input_node")

        self.args = args
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_log_t = time.time()

        self.areas = (
            ALL_SAFETY_AREAS
            if args.safety_area.upper() == "ALL"
            else ordered_area_list([args.safety_area])
        )

        if len(args.area_names) != len(args.static_mask_paths):
            raise ValueError("area_names and static_mask_paths must have the same length")

        self.area_masks = OrderedDict()
        pairs = list(zip(args.area_names, args.static_mask_paths))
        pairs = sorted(
            pairs,
            key=lambda x: ALL_SAFETY_AREAS.index(x[0]) if x[0] in ALL_SAFETY_AREAS else 999
        )

        for area_name, mask_path in pairs:
            if area_name not in self.areas:
                continue
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {area_name}: {mask_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not load mask for {area_name}: {mask_path}")

            self.area_masks[area_name] = mask
            self.get_logger().info(f"[mask] loaded {area_name}: {mask_path} shape={mask.shape}")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if args.use_compressed:
            self.subscription = self.create_subscription(
                CompressedImage,
                args.camera_topic,
                self.callback_compressed,
                sensor_qos,
            )
            self.get_logger().info(f"Subscribed to CompressedImage topic: {args.camera_topic}")
        else:
            self.subscription = self.create_subscription(
                RosImage,
                args.camera_topic,
                self.callback_raw,
                sensor_qos,
            )
            self.get_logger().info(f"Subscribed to Image topic: {args.camera_topic}")

        self.get_logger().info(f"Active areas: {self.areas}")

    def preprocess_area(self, frame_bgr, area_name):
        mask = self.area_masks[area_name]

        contours, mask_bin = _extract_mask_contours(mask, frame_bgr.shape[:2])
        cropped, bbox, masked_full, mask_bin = _crop_with_mask(frame_bgr, mask)

        if cropped is None:
            return {
                "status": "mask_failed",
                "crop": None,
                "resized": None,
                "bbox": None,
                "contours": contours,
                "mask_bin": mask_bin,
                "masked_full": masked_full,
                "resize_meta": None,
            }

        resized, resize_meta = _resize_128(
            cropped,
            keep_aspect=self.args.keep_aspect,
            target=(self.args.target_size, self.args.target_size),
            return_meta=True,
        )

        return {
            "status": "ok",
            "crop": cropped.copy(),
            "resized": resized.copy(),
            "bbox": bbox,
            "contours": contours,
            "mask_bin": mask_bin,
            "masked_full": masked_full,
            "resize_meta": resize_meta,
        }

    def process_frame(self, frame_bgr):
        self.frame_count += 1

        area_inputs = OrderedDict()
        for area_name in self.areas:
            area_inputs[area_name] = self.preprocess_area(frame_bgr, area_name)

        if self.args.show_dashboard:
            dashboard = draw_preprocessing_dashboard(
                frame_bgr,
                area_inputs,
                width=self.args.dashboard_width,
                height=self.args.dashboard_height,
            )
            cv2.imshow("ADVIS Preprocessing Check", dashboard)

        else:
            # basic separate windows
            frame_vis = frame_bgr.copy()
            for area_name in self.areas:
                info = area_inputs[area_name]
                contours = info.get("contours", [])
                bbox = info.get("bbox")

                cv2.drawContours(frame_vis, contours, -1, (0, 255, 255), 2)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_vis, AREA_DISPLAY_NAMES.get(area_name, area_name),
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                crop = info.get("crop")
                resized = info.get("resized")
                if crop is not None:
                    cv2.imshow(f"{area_name} - crop", crop)
                if resized is not None:
                    cv2.imshow(f"{area_name} - model_input_128", resized)

            cv2.imshow("Input Frame", frame_vis)

        if self.frame_count % self.args.log_every_n == 0:
            msg_parts = [f"frame={self.frame_count}"]
            for area_name in self.areas:
                info = area_inputs[area_name]
                bbox = info.get("bbox")
                meta = info.get("resize_meta")
                if bbox is None:
                    msg_parts.append(f"{area_name}: mask_failed")
                else:
                    msg_parts.append(
                        f"{area_name}: bbox={bbox}, resized={self.args.target_size}x{self.args.target_size}, meta={meta}"
                    )
            self.get_logger().info(" | ".join(msg_parts))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.get_logger().info("ESC pressed. Shutting down.")
            rclpy.shutdown()

    def callback_compressed(self, msg):
        try:
            frame_bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                self.get_logger().error("Failed to decode compressed frame.")
                return
            self.process_frame(frame_bgr)
        except Exception as e:
            self.get_logger().error(f"Compressed callback error: {e}")

    def callback_raw(self, msg):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_frame(frame_bgr)
        except Exception as e:
            self.get_logger().error(f"Raw callback error: {e}")


def parse_args():
    p = argparse.ArgumentParser("Check preprocessing / model inputs from ROS camera topic")

    p.add_argument("--camera_topic", default="/camera/back_view/image_raw")
    p.add_argument("--use_compressed", action="store_true",
                   help="Use sensor_msgs/CompressedImage instead of sensor_msgs/Image")

    p.add_argument("--safety_area", default="ALL")
    p.add_argument("--area_names", nargs="+", default=["PLeft", "PRight", "RoboArm", "ConvBelt"])
    p.add_argument("--static_mask_paths", nargs="+", required=True)

    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--keep_aspect", action="store_true", default=True)

    p.add_argument("--show_dashboard", action="store_true")
    p.add_argument("--dashboard_width", type=int, default=1600)
    p.add_argument("--dashboard_height", type=int, default=1000)

    p.add_argument("--log_every_n", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CheckInputNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()