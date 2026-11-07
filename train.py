import csv
from multiprocessing import freeze_support
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

ROOT = Path(r"D:\NCKH2026-2027 - Copy")
DATA_YAML = ROOT / "data.yaml"
PROJECT_DIR = ROOT / "runs"
RUN_NAME = "bracol_yolo26s_seg"
MODEL_WEIGHTS = ROOT / "yolo26s-seg.pt"

COLS = [
    "Class", "Images", "Instances",
    "Box P", "Box R", "Box mAP50", "Box mAP50-95",
    "Mask P", "Mask R", "Mask mAP50", "Mask mAP50-95"
]
METRICS = [
    "box_P", "box_R", "box_mAP50", "box_mAP50-95",
    "mask_P", "mask_R", "mask_mAP50", "mask_mAP50-95"
]


def best_epoch(save_dir):
    results_path = save_dir / "results.csv"
    if results_path.exists():
        rows = list(csv.DictReader(open(results_path)))
        w = {
            "metrics/mAP50(B)": 0.1, "metrics/mAP50-95(B)": 0.9,
            "metrics/mAP50(M)": 0.1, "metrics/mAP50-95(M)": 0.9
        }
        return max(rows, key=lambda r: sum(float(r[k]) * v for k, v in w.items()))["epoch"].strip()
    return "best"


def render_table(split, rows, n_images, epoch_num, save_dir):
    fig, ax = plt.subplots(figsize=(18, 3.2))
    ax.axis("off")
    ax.set_title(
        f"YOLO26s-seg - best.pt epoch {epoch_num} - {split.upper()} "
        f"({n_images} images, imgsz 1024)",
        fontsize=14, fontweight="bold", pad=12
    )

    t = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.8)
    t.auto_set_column_width(list(range(len(COLS))))

    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f4f6f")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        name = rows[r - 1][0].lower()
        if name == "all":
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")
        elif name == "phoma":
            cell.set_facecolor("#e2f0e2")

        if c in (9, 10):
            cell.set_text_props(fontweight="bold")

    out_file = save_dir / f"report_{split}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Da tao bang anh: {out_file}")


def evaluate_and_generate_reports(save_dir):
    print("\n" + "=" * 70)
    print("BAT DAU DANH GIA (EVALUATE) VA XUAT 4 FILE KET QUA")
    print("=" * 70)

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[Canh bao] Khong tim thay {best_pt}, su dung last.pt")
        best_pt = save_dir / "weights" / "last.pt"

    eval_model = YOLO(str(best_pt))
    epoch_str = best_epoch(save_dir)

    n_images_dict = {
        "train": len(list((ROOT / "BRACOL_SEG_SAM/train/images").glob("*.*"))),
        "val": len(list((ROOT / "BRACOL_SEG_SAM/valid/images").glob("*.*"))),
        "test": len(list((ROOT / "BRACOL_SEG_SAM/test/images").glob("*.*")))
    }

    lines = ["mode,split,class,images,instances,box_P,box_R,box_mAP50,box_mAP50-95,mask_P,mask_R,mask_mAP50,mask_mAP50-95"]

    for split in ["train", "val", "test"]:
        print(f"\n>>> Dang danh gia tap: {split.upper()}...")
        kw = dict(
            data=str(DATA_YAML),
            split=split,
            imgsz=1024,
            batch=2,
            workers=0,
            device=0,
            project=str(save_dir),
            name=f"eval_{split}",
            exist_ok=True
        )
        r = eval_model.val(**kw)

        nt = r.nt_per_class
        ni = r.nt_per_image

        lines.append(
            f"standard,{split},all,,{int(nt.sum())},"
            + ",".join(f"{v:.4f}" for v in (*r.box.mean_results(), *r.seg.mean_results()))
        )

        for i, c in enumerate(r.ap_class_index):
            vals = (*r.box.class_result(i), *r.seg.class_result(i))
            lines.append(
                f"standard,{split},{eval_model.names[int(c)]},{int(ni[c])},{int(nt[c])},"
                + ",".join(f"{v:.4f}" for v in vals)
            )

    # 1. Ghi eval_summary.csv
    csv_out = save_dir / "eval_summary.csv"
    csv_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Da tao file CSV: {csv_out}")

    # 2. Tao 3 file anh report_train.png, report_val.png, report_test.png
    data = list(csv.DictReader(open(csv_out, encoding="utf-8")))
    for split in ["train", "val", "test"]:
        rows = []
        for d in data:
            if d["split"] != split or d["mode"] != "standard":
                continue
            imgs = n_images_dict[split] if d["class"] == "all" else d["images"]
            rows.append([d["class"], imgs, d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
        render_table(split, rows, n_images_dict[split], epoch_str, save_dir)

    print("\n" + "=" * 70)
    print("HOAN THANH! TAT CA 4 FILE KET QUA DA CO TRONG:")
    print(f"-> {save_dir}")
    print("=" * 70)


def main():
    freeze_support()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Khong tim thay {MODEL_WEIGHTS}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Khong tim thay {DATA_YAML}")

    model = YOLO(str(MODEL_WEIGHTS))

    print("\n>>> BAT DAU HUAN LUYEN (TRAIN 100 EPOCHS)...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1024,
        batch=3,
        device=0,
        workers=4,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        pretrained=True,
        patience=30,
        save=True,
        plots=True,
        amp=True,
        exist_ok=True,
    )

    print("\nTRAIN DONE")

    save_dir = Path(results.save_dir)
    evaluate_and_generate_reports(save_dir)


if __name__ == "__main__":
    main()
