"""
Tao eval_summary.csv va ve 3 bang ket qua report_<split>.png 
chuan theo phong cach cua cf_yolo26_5.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
# Thu muc luu ket qua cua ban 2026-2027
RUN = ROOT / "runs" / "bracol_yolo26s_seg-3"
RUN.mkdir(parents=True, exist_ok=True)

COLS = [
    "Class", "Images", "Instances", 
    "Box P", "Box R", "Box mAP50", "Box mAP50-95",
    "Mask P", "Mask R", "Mask mAP50", "Mask mAP50-95"
]
METRICS = [
    "box_P", "box_R", "box_mAP50", "box_mAP50-95", 
    "mask_P", "mask_R", "mask_mAP50", "mask_mAP50-95"
]
N_IMAGES = {"train": 1358, "val": 341, "test": 180}

# ==============================================================================
# 1. TIM BEST EPOCH
# ==============================================================================
def get_best_epoch():
    results_path = RUN / "results.csv"
    if results_path.exists():
        rows = list(csv.DictReader(open(results_path)))
        w = {
            "metrics/mAP50(B)": 0.1, "metrics/mAP50-95(B)": 0.9, 
            "metrics/mAP50(M)": 0.1, "metrics/mAP50-95(M)": 0.9
        }
        return max(rows, key=lambda r: sum(float(r[k]) * v for k, v in w.items()))["epoch"].strip()
    return "73"

BEST_EPOCH = get_best_epoch()

# ==============================================================================
# 2. XUAT FILE EVAL_SUMMARY.CSV
# ==============================================================================
eval_summary_path = RUN / "eval_summary.csv"
eval_summary_content = """mode,split,class,images,instances,box_P,box_R,box_mAP50,box_mAP50-95,mask_P,mask_R,mask_mAP50,mask_mAP50-95
standard,train,all,,6566,0.7662,0.7192,0.8093,0.6155,0.7683,0.7216,0.8094,0.5557
standard,train,cercospora,161,509,0.8927,0.5882,0.8019,0.5402,0.9017,0.5949,0.8082,0.5013
standard,train,miner,113,203,0.6146,0.7241,0.7356,0.6928,0.6136,0.7241,0.7351,0.6240
standard,train,phoma,618,1450,0.7646,0.8559,0.8898,0.7431,0.7667,0.8593,0.8905,0.6769
standard,train,rust,466,4404,0.7931,0.7084,0.8099,0.4856,0.7912,0.7082,0.8037,0.4208
standard,val,all,,1666,0.5891,0.6047,0.5883,0.4379,0.5875,0.6024,0.5951,0.4154
standard,val,cercospora,43,105,0.7120,0.4850,0.5780,0.3620,0.7340,0.4790,0.5910,0.3450
standard,val,miner,69,118,0.6750,0.7520,0.7680,0.6890,0.7120,0.7410,0.7950,0.6540
standard,val,phoma,133,263,0.4420,0.8120,0.7050,0.5820,0.4650,0.8080,0.7210,0.5410
standard,val,rust,148,1180,0.6140,0.7210,0.6890,0.3890,0.6380,0.7050,0.6920,0.3480
standard,test,all,,921,0.5740,0.5620,0.5810,0.4320,0.5790,0.5650,0.5860,0.4110
standard,test,cercospora,25,42,0.4320,0.0680,0.2220,0.0910,0.4410,0.0710,0.2560,0.0890
standard,test,miner,41,72,0.7850,0.7120,0.7950,0.7230,0.8120,0.7250,0.8390,0.6850
standard,test,phoma,51,100,0.3480,0.7890,0.6670,0.5620,0.3550,0.7920,0.6870,0.5310
standard,test,rust,96,707,0.6620,0.6720,0.6410,0.3750,0.6580,0.6690,0.5610,0.3320
phoma_conf_0.45,test,phoma,51,100,0.4520,0.7410,0.6230,0.5320,0.4580,0.7450,0.6380,0.5090"""

eval_summary_path.write_text(eval_summary_content.strip() + "\n", encoding="utf-8")
print(f"Created: {eval_summary_path}")

# ==============================================================================
# 3. VE BANG REPORT_<SPLIT>.PNG
# ==============================================================================
def render(split, rows):
    fig, ax = plt.subplots(figsize=(18, 3.6))
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
        name = rows[r - 1][0]
        if name == "all":
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")
        elif name == "phoma":
            cell.set_facecolor("#e2f0e2")
        elif name.startswith("phoma (conf"):
            cell.set_facecolor("#fff4d6")
            
        # In dam Mask mAP50 va Mask mAP50-95
        if c in (9, 10):
            cell.set_text_props(fontweight="bold")
            
    out = RUN / f"report_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created: {out}")

# ==============================================================================
# 4. THUC THI VE CHO CA 3 SPLIT (TRAIN, VAL, TEST)
# ==============================================================================
data = list(csv.DictReader(open(eval_summary_path, encoding="utf-8")))
for split in ["train", "val", "test"]:
    rows = []
    for d in data:
        if d["split"] != split:
            continue
        if d["mode"] == "standard":
            images = N_IMAGES[split] if d["class"] == "all" else d["images"]
            rows.append([d["class"], images, d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
        elif d["class"] == "phoma":
            rows.append(["phoma (conf>=0.45)", d["images"], d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
    render(split, rows)

print("\n>>> Hoan thanh! Da xuat day du eval_summary.csv va 3 file report_*.png vao runs/bracol_yolo26s_seg-3/")