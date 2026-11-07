"""Ve bang ket qua report_<split>.png tu eval_summary.csv giong repo s3."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "bracol_yolo26s_seg-3"
COLS = [
    "Class", "Images", "Instances", 
    "Box P", "Box R", "Box mAP50", "Box mAP50-95",
    "Mask P", "Mask R", "Mask mAP50", "Mask mAP50-95"
]
METRICS = [
    "box_P", "box_R", "box_mAP50", "box_mAP50-95", 
    "mask_P", "mask_R", "mask_mAP50", "mask_mAP50-95"
]

DATA_DIR = Path(r"D:\NCKH2026-2027 - Copy\BRACOL_SEG_SAM")
N_IMAGES = {
    "train": len(list((DATA_DIR / "train/images").glob("*.*"))),
    "val": len(list((DATA_DIR / "valid/images").glob("*.*"))),
    "test": len(list((DATA_DIR / "test/images").glob("*.*")))
}

def best_epoch():
    results_path = RUN / "results.csv"
    if results_path.exists():
        rows = list(csv.DictReader(open(results_path)))
        w = {
            "metrics/mAP50(B)": 0.1, "metrics/mAP50-95(B)": 0.9, 
            "metrics/mAP50(M)": 0.1, "metrics/mAP50-95(M)": 0.9
        }
        return max(rows, key=lambda r: sum(float(r[k]) * v for k, v in w.items()))["epoch"].strip()
    return "73"

BEST_EPOCH = best_epoch()

def render(split, rows):
    fig, ax = plt.subplots(figsize=(18, 3.2))
    ax.axis("off")
    ax.set_title(
        f"YOLO26s-seg - best.pt epoch {BEST_EPOCH} - {split.upper()} "
        f"({N_IMAGES[split]} images, imgsz 1024)", 
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
            
    out = RUN / f"report_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created: {out}")

def main():
    csv_path = RUN / "eval_summary.csv"
    if not csv_path.exists():
        print(f"Error: Khong tim thay {csv_path}. Hay chay evaluate.py truoc.")
        return

    data = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for split in ["train", "val", "test"]:
        rows = []
        for d in data:
            if d["split"] != split or d["mode"] != "standard":
                continue
            images = N_IMAGES[split] if d["class"] == "all" else d["images"]
            rows.append([d["class"], images, d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
        render(split, rows)

if __name__ == "__main__":
    main()
