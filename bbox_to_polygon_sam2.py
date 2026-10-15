from pathlib import Path
from contextlib import nullcontext
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================================================
# 1. CẤU HÌNH
# =========================================================

ROOT = Path(r"D:\NCKH2026-2027")

# Dataset bbox gốc
SOURCE_DATASET = ROOT / "BRACOL_REVIEWED"

# Dataset segmentation đầu ra
OUTPUT_DATASET = ROOT / "BRACOL_SEG_SAM"

# SAM 2.1 Base+
CHECKPOINT = (
    ROOT
    / "sam2"
    / "checkpoints"
    / "sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Các tập dữ liệu
SPLITS = ["train", "valid", "test"]

# Mapping class BRACOL
CLASS_NAMES = {
    0: "Cercospora",
    1: "Miner",
    2: "Phoma",
    3: "Rust",
}

# True = nếu label polygon đã tồn tại thì bỏ qua
# Hữu ích nếu chương trình bị dừng giữa chừng rồi chạy lại
SKIP_EXISTING = True

# None = chạy toàn bộ
# Để test thử 5 ảnh thì đổi thành 5
MAX_IMAGES_PER_SPLIT = None

# Đơn giản hóa polygon.
# Càng nhỏ -> polygon càng chi tiết nhưng nhiều điểm hơn.
POLYGON_EPSILON_RATIO = 0.0015

# Mask quá nhỏ thì xem như SAM thất bại
MIN_MASK_AREA = 20

# Điểm SAM thấp hơn mức này vẫn lưu polygon,
# nhưng đưa ảnh vào danh sách cần review.
REVIEW_SCORE_THRESHOLD = 0.80

# Nếu mask bị chia thành nhiều mảnh và mảnh thứ 2
# lớn hơn 20% mảnh chính -> đánh dấu review.
SECOND_COMPONENT_RATIO = 0.20


# =========================================================
# 2. KIỂM TRA
# =========================================================

if not SOURCE_DATASET.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset nguồn: {SOURCE_DATASET}"
    )

if not OUTPUT_DATASET.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset output: {OUTPUT_DATASET}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Không tìm thấy checkpoint SAM: {CHECKPOINT}"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("BRACOL BBOX -> SAM 2.1 -> YOLO SEG")
print("=" * 70)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

print("Source:", SOURCE_DATASET)
print("Output:", OUTPUT_DATASET)
print("Checkpoint:", CHECKPOINT)
print()


# =========================================================
# 3. LOAD SAM 2.1
# =========================================================

print("Đang load SAM 2.1...")

sam_model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device=DEVICE,
)

predictor = SAM2ImagePredictor(sam_model)

print("SAM 2.1 load thành công.")
print()


# =========================================================
# 4. HÀM YOLO BBOX -> XYXY PIXEL
# =========================================================

def yolo_bbox_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    """
    YOLO:
        xc yc width height
    normalized 0-1

    -> SAM:
        x1 y1 x2 y2
    pixel
    """

    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w - 1))
    y2 = max(0.0, min(y2, img_h - 1))

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


# =========================================================
# 5. MASK -> POLYGON
# =========================================================

def mask_to_polygon(mask, img_w, img_h):
    """
    Binary mask SAM -> polygon YOLO normalized.

    Trả về:
        polygon
        fragmented
        reason
    """

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None, False, "Không tìm thấy contour"

    # Sắp xếp contour lớn nhất trước
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    largest = contours[0]
    largest_area = cv2.contourArea(largest)

    if largest_area < MIN_MASK_AREA:
        return (
            None,
            False,
            f"Mask quá nhỏ: {largest_area:.1f}px",
        )

    # Kiểm tra mask có bị chia thành nhiều mảnh lớn không
    fragmented = False

    if len(contours) >= 2:
        second_area = cv2.contourArea(contours[1])

        if (
            largest_area > 0
            and second_area / largest_area
            >= SECOND_COMPONENT_RATIO
        ):
            fragmented = True

    # Làm mượt / giảm số lượng điểm polygon
    perimeter = cv2.arcLength(
        largest,
        True,
    )

    epsilon = max(
        0.5,
        POLYGON_EPSILON_RATIO * perimeter,
    )

    polygon = cv2.approxPolyDP(
        largest,
        epsilon,
        True,
    )

    polygon = polygon.reshape(-1, 2)

    # Nếu simplify quá mạnh thì thử contour gốc
    if len(polygon) < 3:
        polygon = largest.reshape(-1, 2)

    if len(polygon) < 3:
        return None, fragmented, "Polygon có < 3 điểm"

    # Chuẩn hóa tọa độ về 0-1
    polygon = polygon.astype(np.float64)

    polygon[:, 0] /= img_w
    polygon[:, 1] /= img_h

    polygon[:, 0] = np.clip(
        polygon[:, 0],
        0.0,
        1.0,
    )

    polygon[:, 1] = np.clip(
        polygon[:, 1],
        0.0,
        1.0,
    )

    # Làm tròn
    polygon = np.round(polygon, 6)

    # Xóa các điểm liên tiếp trùng nhau
    cleaned = []

    for point in polygon:
        point_tuple = tuple(point)

        if (
            not cleaned
            or point_tuple != cleaned[-1]
        ):
            cleaned.append(point_tuple)

    # Nếu điểm cuối trùng điểm đầu thì bỏ điểm cuối
    if (
        len(cleaned) > 1
        and cleaned[0] == cleaned[-1]
    ):
        cleaned.pop()

    if len(cleaned) < 3:
        return (
            None,
            fragmented,
            "Polygon sau khi làm sạch có < 3 điểm",
        )

    return cleaned, fragmented, None


# =========================================================
# 6. GHI YOLO SEGMENTATION
# =========================================================

def polygon_to_yolo_line(class_id, polygon):
    values = [str(class_id)]

    for x, y in polygon:
        values.append(f"{x:.6f}")
        values.append(f"{y:.6f}")

    return " ".join(values)


# =========================================================
# 7. CONTEXT AUTOCAST
# =========================================================

def autocast_context():
    if DEVICE != "cuda":
        return nullcontext()

    # RTX 5050 hỗ trợ BF16
    if torch.cuda.is_bf16_supported():
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


# =========================================================
# 8. REPORT
# =========================================================

REPORT_DIR = OUTPUT_DATASET / "_sam_reports"
REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

failed_rows = []
review_rows = []

total_images = 0
total_objects = 0
successful_objects = 0
failed_objects = 0


# =========================================================
# 9. XỬ LÝ TRAIN / VALID / TEST
# =========================================================

for split in SPLITS:

    source_images_dir = (
        SOURCE_DATASET
        / split
        / "images"
    )

    source_labels_dir = (
        SOURCE_DATASET
        / split
        / "labels"
    )

    output_labels_dir = (
        OUTPUT_DATASET
        / split
        / "labels"
    )

    output_labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]:
        image_files.extend(
            source_images_dir.glob(extension)
        )

    # tránh duplicate và sắp xếp
    image_files = sorted(
        set(image_files)
    )

    if MAX_IMAGES_PER_SPLIT is not None:
        image_files = image_files[
            :MAX_IMAGES_PER_SPLIT
        ]

    print()
    print("=" * 70)
    print(f"ĐANG XỬ LÝ: {split}")
    print("Số ảnh:", len(image_files))
    print("=" * 70)

    for image_path in tqdm(
        image_files,
        desc=split,
    ):

        total_images += 1

        label_path = (
            source_labels_dir
            / f"{image_path.stem}.txt"
        )

        output_label_path = (
            output_labels_dir
            / f"{image_path.stem}.txt"
        )

        # ---------------------------------------------
        # Nếu chạy lại thì bỏ qua file đã xử lý
        # ---------------------------------------------

        if (
            SKIP_EXISTING
            and output_label_path.exists()
        ):
            continue

        # ---------------------------------------------
        # Kiểm tra label bbox nguồn
        # ---------------------------------------------

        if not label_path.exists():

            failed_rows.append(
                [
                    split,
                    image_path.name,
                    "",
                    "",
                    "Không có file bbox nguồn",
                ]
            )

            review_rows.append(
                [
                    split,
                    image_path.name,
                    "missing_source_label",
                ]
            )

            continue

        # ---------------------------------------------
        # Đọc ảnh
        # ---------------------------------------------

        image_bgr = cv2.imread(
            str(image_path)
        )

        if image_bgr is None:

            failed_rows.append(
                [
                    split,
                    image_path.name,
                    "",
                    "",
                    "Không đọc được ảnh",
                ]
            )

            review_rows.append(
                [
                    split,
                    image_path.name,
                    "cannot_read_image",
                ]
            )

            continue

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        img_h, img_w = image_rgb.shape[:2]

        # ---------------------------------------------
        # Đọc bbox
        # ---------------------------------------------

        lines = [
            x.strip()
            for x in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

        # Ảnh negative không có object
        if len(lines) == 0:

            output_label_path.write_text(
                "",
                encoding="utf-8",
            )

            continue

        output_lines = []
        image_needs_review = False

        # ---------------------------------------------
        # Encode ảnh đúng 1 lần
        # ---------------------------------------------

        try:

            with torch.inference_mode():
                with autocast_context():

                    predictor.set_image(
                        image_rgb
                    )

                    # =================================
                    # XỬ LÝ TỪNG BBOX
                    # =================================

                    for line_number, line in enumerate(
                        lines,
                        start=1,
                    ):

                        total_objects += 1

                        parts = line.split()

                        # YOLO detection chuẩn = 5 giá trị
                        if len(parts) != 5:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    "",
                                    (
                                        "BBox không có đúng "
                                        "5 giá trị"
                                    ),
                                ]
                            )

                            image_needs_review = True
                            continue

                        try:

                            class_id = int(
                                float(parts[0])
                            )

                            xc = float(parts[1])
                            yc = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])

                        except ValueError:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    "",
                                    "Không đọc được bbox",
                                ]
                            )

                            image_needs_review = True
                            continue

                        # Kiểm tra class
                        if class_id not in CLASS_NAMES:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    class_id,
                                    "Class ID không hợp lệ",
                                ]
                            )

                            image_needs_review = True
                            continue

                        # Kiểm tra bbox
                        if bw <= 0 or bh <= 0:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    class_id,
                                    "BBox width/height <= 0",
                                ]
                            )

                            image_needs_review = True
                            continue

                        # ---------------------------------
                        # YOLO bbox -> SAM box XYXY
                        # ---------------------------------

                        box = yolo_bbox_to_xyxy(
                            xc,
                            yc,
                            bw,
                            bh,
                            img_w,
                            img_h,
                        )

                        # ---------------------------------
                        # SAM predict
                        # ---------------------------------

                        try:

                            masks, scores, _ = (
                                predictor.predict(
                                    box=box,
                                    multimask_output=False,
                                )
                            )

                        except Exception as e:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    class_id,
                                    f"SAM predict lỗi: {e}",
                                ]
                            )

                            image_needs_review = True
                            continue

                        if masks is None or len(masks) == 0:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    class_id,
                                    "SAM không trả mask",
                                ]
                            )

                            image_needs_review = True
                            continue

                        mask = masks[0]

                        score = (
                            float(scores[0])
                            if scores is not None
                            else -1
                        )

                        # ---------------------------------
                        # Mask -> polygon
                        # ---------------------------------

                        polygon, fragmented, reason = (
                            mask_to_polygon(
                                mask,
                                img_w,
                                img_h,
                            )
                        )

                        if polygon is None:

                            failed_objects += 1

                            failed_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    line_number,
                                    class_id,
                                    reason,
                                ]
                            )

                            image_needs_review = True
                            continue

                        # ---------------------------------
                        # Ghi polygon
                        # ---------------------------------

                        yolo_line = polygon_to_yolo_line(
                            class_id,
                            polygon,
                        )

                        output_lines.append(
                            yolo_line
                        )

                        successful_objects += 1

                        # ---------------------------------
                        # Đánh dấu mask đáng nghi
                        # nhưng vẫn lưu polygon
                        # ---------------------------------

                        if (
                            score
                            < REVIEW_SCORE_THRESHOLD
                        ):

                            review_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    (
                                        f"low_sam_score="
                                        f"{score:.4f}"
                                    ),
                                ]
                            )

                            image_needs_review = True

                        if fragmented:

                            review_rows.append(
                                [
                                    split,
                                    image_path.name,
                                    "fragmented_mask",
                                ]
                            )

                            image_needs_review = True

        except Exception as e:

            failed_rows.append(
                [
                    split,
                    image_path.name,
                    "",
                    "",
                    f"Lỗi xử lý ảnh: {e}",
                ]
            )

            review_rows.append(
                [
                    split,
                    image_path.name,
                    "image_processing_error",
                ]
            )

            continue

        # ---------------------------------------------
        # Chỉ tạo label nếu có ít nhất 1 polygon
        # ---------------------------------------------

        if len(output_lines) > 0:

            output_label_path.write_text(
                "\n".join(output_lines)
                + "\n",
                encoding="utf-8",
            )

        else:

            # Không tạo file để dễ phát hiện ảnh SAM fail
            if output_label_path.exists():
                output_label_path.unlink()

            image_needs_review = True

        if image_needs_review:

            # tránh duplicate quá nhiều ở danh sách tổng
            review_rows.append(
                [
                    split,
                    image_path.name,
                    "needs_manual_review",
                ]
            )


# =========================================================
# 10. GHI REPORT
# =========================================================

failed_csv = (
    REPORT_DIR
    / "failed_objects.csv"
)

with open(
    failed_csv,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "split",
            "image",
            "bbox_line",
            "class_id",
            "reason",
        ]
    )

    writer.writerows(
        failed_rows
    )


review_csv = (
    REPORT_DIR
    / "needs_review.csv"
)

# loại duplicate
unique_review_rows = []
seen = set()

for row in review_rows:

    key = tuple(row)

    if key not in seen:
        seen.add(key)
        unique_review_rows.append(row)

with open(
    review_csv,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "split",
            "image",
            "reason",
        ]
    )

    writer.writerows(
        unique_review_rows
    )


# =========================================================
# 11. SUMMARY
# =========================================================

print()
print("=" * 70)
print("HOÀN THÀNH")
print("=" * 70)

print("Tổng ảnh xử lý:", total_images)
print("Tổng bbox:", total_objects)

print(
    "Polygon thành công:",
    successful_objects,
)

print(
    "Object thất bại:",
    failed_objects,
)

print()
print(
    "Danh sách lỗi:",
    failed_csv,
)

print(
    "Danh sách cần review:",
    review_csv,
)

print()
print(
    "Polygon được lưu tại:"
)

print(
    OUTPUT_DATASET
    / "train"
    / "labels"
)

print(
    OUTPUT_DATASET
    / "valid"
    / "labels"
)

print(
    OUTPUT_DATASET
    / "test"
    / "labels"
)