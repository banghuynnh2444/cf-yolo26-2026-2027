from pathlib import Path
from multiprocessing import freeze_support

from ultralytics import YOLO


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DATA_YAML = Path(
    r"D:\NCKH2026-2027 - Copy\data.yaml"
)

MODEL_PATH = Path(
    r"D:\NCKH2026-2027 - Copy\runs\bracol_yolo26s_seg-3\weights\best.pt"
)

TEST_IMAGES = Path(
    r"D:\NCKH2026-2027 - Copy\BRACOL_SEG_SAM\test\images"
)

OUTPUT_ROOT = Path(
    r"D:\NCKH2026-2027 - Copy\runs\test_results"
)


# Cấu hình test
IMGSZ = 640
BATCH = 8
DEVICE = 0

# Chỉ dùng cho predict ảnh sau khi evaluate
CONF = 0.25
IOU = 0.70


# Mapping class của BRACOL
CLASS_NAMES = {
    0: "Cercospora",
    1: "Miner",
    2: "Phoma",
    3: "Rust",
}


# ============================================================
# 2. HÀM IN METRICS THEO CLASS
# ============================================================

def print_class_metrics(metric, title):
    """
    In Precision, Recall, AP50 và AP50-95 của từng class.
    """

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"{'ID':<5}"
        f"{'CLASS':<15}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'mAP50':<15}"
        f"{'mAP50-95':<15}"
    )

    print("-" * 90)

    try:
        precision = metric.p
        recall = metric.r
        ap50 = metric.ap50
        ap5095 = metric.ap

        number_classes = len(precision)

        for class_id in range(number_classes):

            class_name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            )

            print(
                f"{class_id:<5}"
                f"{class_name:<15}"
                f"{float(precision[class_id]):<15.4f}"
                f"{float(recall[class_id]):<15.4f}"
                f"{float(ap50[class_id]):<15.4f}"
                f"{float(ap5095[class_id]):<15.4f}"
            )

    except Exception as e:
        print(
            "Không đọc được metric từng class:",
            e
        )


# ============================================================
# 3. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # KIỂM TRA ĐƯỜNG DẪN
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nKhông tìm thấy data.yaml:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nKhông tìm thấy best.pt:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"\nKhông tìm thấy test images:\n{TEST_IMAGES}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # THÔNG TIN
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("YOLO26s-SEG - TEST BRACOL")
    print("=" * 90)

    print(f"Model     : {MODEL_PATH}")
    print(f"Data YAML : {DATA_YAML}")
    print(f"Test imgs : {TEST_IMAGES}")
    print(f"Image size: {IMGSZ}")
    print(f"Batch     : {BATCH}")
    print(f"Device    : CUDA:{DEVICE}")

    print()


    # --------------------------------------------------------
    # ĐẾM ẢNH TEST
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    test_images = [
        p
        for p in TEST_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]

    print(
        f"Số ảnh test tìm thấy: {len(test_images)}"
    )

    print()


    # ========================================================
    # 4. LOAD MODEL
    # ========================================================

    print("=" * 90)
    print("LOAD MODEL")
    print("=" * 90)

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Load best.pt thành công.")

    print()


    # ========================================================
    # 5. TEST / EVALUATE
    # ========================================================

    print("=" * 90)
    print("ĐANG ĐÁNH GIÁ TRÊN TEST SET")
    print("=" * 90)

    metrics = model.val(

        data=str(DATA_YAML),

        # dùng test trong data.yaml
        split="test",

        imgsz=IMGSZ,

        batch=BATCH,

        device=DEVICE,

        # Windows:
        # workers=0 tránh multiprocessing lỗi spawn
        workers=0,

        plots=True,

        save_json=False,

        project=str(OUTPUT_ROOT),

        name="test_eval",

        exist_ok=True,

        verbose=True,
    )


    # ========================================================
    # 6. KẾT QUẢ TỔNG BOX
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("KẾT QUẢ TEST CUỐI CÙNG")
    print("=" * 90)

    print()
    print("[ BOX / OBJECT DETECTION ]")
    print("-" * 50)

    try:
        box_precision = float(metrics.box.mp)
        box_recall = float(metrics.box.mr)
        box_map50 = float(metrics.box.map50)
        box_map5095 = float(metrics.box.map)

        print(
            f"Precision      : {box_precision:.4f}"
        )

        print(
            f"Recall         : {box_recall:.4f}"
        )

        print(
            f"mAP@0.50       : {box_map50:.4f}"
        )

        print(
            f"mAP@0.50:0.95  : {box_map5095:.4f}"
        )

    except Exception as e:

        print(
            "Không lấy được Box metrics:",
            e
        )


    # ========================================================
    # 7. KẾT QUẢ TỔNG MASK
    # ========================================================

    print()
    print("[ MASK / SEGMENTATION ]")
    print("-" * 50)

    try:
        mask_precision = float(metrics.seg.mp)
        mask_recall = float(metrics.seg.mr)
        mask_map50 = float(metrics.seg.map50)
        mask_map5095 = float(metrics.seg.map)

        print(
            f"Precision      : {mask_precision:.4f}"
        )

        print(
            f"Recall         : {mask_recall:.4f}"
        )

        print(
            f"mAP@0.50       : {mask_map50:.4f}"
        )

        print(
            f"mAP@0.50:0.95  : {mask_map5095:.4f}"
        )

    except Exception as e:

        print(
            "Không lấy được Mask metrics:",
            e
        )


    # ========================================================
    # 8. BOX METRIC THEO CLASS
    # ========================================================

    print_class_metrics(
        metrics.box,
        "BOX METRICS THEO TỪNG CLASS"
    )


    # ========================================================
    # 9. MASK METRIC THEO CLASS
    # ========================================================

    print_class_metrics(
        metrics.seg,
        "MASK METRICS THEO TỪNG CLASS"
    )


    # ========================================================
    # 10. SPEED
    # ========================================================

    print()
    print("=" * 90)
    print("TỐC ĐỘ XỬ LÝ")
    print("=" * 90)

    try:

        for key, value in metrics.speed.items():

            print(
                f"{key:<20}: "
                f"{float(value):.3f} ms/image"
            )

    except Exception as e:

        print(
            "Không đọc được speed:",
            e
        )


    # ========================================================
    # 11. PREDICT TOÀN BỘ ẢNH TEST
    # ========================================================

    print()
    print("=" * 90)
    print("PREDICT ẢNH TEST")
    print("=" * 90)

    print(
        f"Confidence threshold: {CONF}"
    )

    print(
        f"IoU threshold       : {IOU}"
    )

    model.predict(

        source=str(TEST_IMAGES),

        imgsz=IMGSZ,

        conf=CONF,

        iou=IOU,

        device=DEVICE,

        # lưu ảnh có mask
        save=True,

        # lưu polygon prediction YOLO
        save_txt=True,

        # lưu confidence
        save_conf=True,

        project=str(OUTPUT_ROOT),

        name="test_predictions",

        exist_ok=True,

        verbose=False,
    )


    # ========================================================
    # 12. THÔNG BÁO HOÀN THÀNH
    # ========================================================

    print()
    print("=" * 90)
    print("TEST HOÀN THÀNH")
    print("=" * 90)

    print()

    print(
        "Evaluation results:"
    )

    print(
        OUTPUT_ROOT / "test_eval"
    )

    print()

    print(
        "Prediction images:"
    )

    print(
        OUTPUT_ROOT / "test_predictions"
    )

    print()

    print(
        "Prediction labels:"
    )

    print(
        OUTPUT_ROOT
        / "test_predictions"
        / "labels"
    )

    print()
    print("=" * 90)


# ============================================================
# 13. WINDOWS MULTIPROCESSING FIX
# ============================================================

if __name__ == "__main__":

    freeze_support()

    main()