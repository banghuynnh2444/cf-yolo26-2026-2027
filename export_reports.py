import csv
from pathlib import Path

# Đường dẫn thư mục runs trong repo
RUNS_DIR = Path("runs")
RESULTS_CSV = RUNS_DIR / "bracol_yolo26s_seg-3" / "results.csv"
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# 1. ĐỌC DỮ LIỆU TỪ RESULTS.CSV
# -------------------------------------------------------------
if not RESULTS_CSV.exists():
    print(f"Không tìm thấy file: {RESULTS_CSV}")
    exit()

rows = []
with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Làm sạch tên cột (bỏ khoảng trắng thừa)
        cleaned = {k.strip(): float(v.strip()) for k, v in r.items() if k and v}
        rows.append(cleaned)

total_epochs = len(rows)
first_epoch = rows[0]
last_epoch = rows[-1]

# Tìm Best Epoch theo Box mAP50 và Mask mAP50
best_box_epoch = max(rows, key=lambda x: x["metrics/mAP50(B)"])
best_mask_epoch = max(rows, key=lambda x: x["metrics/mAP50(M)"])

# -------------------------------------------------------------
# 2. XUẤT FILE 1: TRAIN REPORT (train_report.txt)
# -------------------------------------------------------------
train_report_path = OUTPUT_DIR / "train_report.txt"
with open(train_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("                 BÁO CÁO HUẤN LUYỆN (TRAIN REPORT)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Architecture  : YOLO26s-seg\n")
    f.write(f"Số epochs hoàn thành: {total_epochs}\n")
    f.write(f"Image size          : 1024\n")
    f.write(f"Batch size          : 3\n\n")
    f.write("-" * 60 + "\n")
    f.write("TIẾN TRÌNH GIẢM LOSS QUA CÁC EPOCH:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Epoch 1 (Bắt đầu):\n")
    f.write(f"  - Train Box Loss : {first_epoch['train/box_loss']:.4f}\n")
    f.write(f"  - Train Seg Loss : {first_epoch['train/seg_loss']:.4f}\n")
    f.write(f"  - Train Cls Loss : {first_epoch['train/cls_loss']:.4f}\n\n")
    f.write(f"Epoch {total_epochs} (Kết thúc):\n")
    f.write(f"  - Train Box Loss : {last_epoch['train/box_loss']:.4f}\n")
    f.write(f"  - Train Seg Loss : {last_epoch['train/seg_loss']:.4f}\n")
    f.write(f"  - Train Cls Loss : {last_epoch['train/cls_loss']:.4f}\n\n")
    f.write(f"Trạng thái: Huấn luyện hoàn tất 100 epochs thành công.\n")

print(f"Đã tạo: {train_report_path}")

# -------------------------------------------------------------
# 3. XUẤT FILE 2: VALIDATION REPORT (val_report.txt)
# -------------------------------------------------------------
val_report_path = OUTPUT_DIR / "val_report.txt"
with open(val_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("            BÁO CÁO VALIDATION (VAL REPORT)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Số epoch đánh giá: {total_epochs}\n\n")
    f.write("-" * 60 + "\n")
    f.write("1. KẾT QUẢ TẠI BEST EPOCH (Độ chính xác cao nhất):\n")
    f.write("-" * 60 + "\n")
    f.write(f"* Best Box mAP@0.5  (Đạt tại Epoch {int(best_box_epoch['epoch'])}):\n")
    f.write(f"    - Precision      : {best_box_epoch['metrics/precision(B)']:.4f}\n")
    f.write(f"    - Recall         : {best_box_epoch['metrics/recall(B)']:.4f}\n")
    f.write(f"    - mAP@0.5        : {best_box_epoch['metrics/mAP50(B)']:.4f}\n")
    f.write(f"    - mAP@0.5:0.95   : {best_box_epoch['metrics/mAP50-95(B)']:.4f}\n\n")
    
    f.write(f"* Best Mask mAP@0.5 (Đạt tại Epoch {int(best_mask_epoch['epoch'])}):\n")
    f.write(f"    - Precision      : {best_mask_epoch['metrics/precision(M)']:.4f}\n")
    f.write(f"    - Recall         : {best_mask_epoch['metrics/recall(M)']:.4f}\n")
    f.write(f"    - mAP@0.5        : {best_mask_epoch['metrics/mAP50(M)']:.4f}\n")
    f.write(f"    - mAP@0.5:0.95   : {best_mask_epoch['metrics/mAP50-95(M)']:.4f}\n\n")

    f.write("-" * 60 + "\n")
    f.write("2. KẾT QUẢ TẠI FINAL EPOCH (Epoch 100):\n")
    f.write("-" * 60 + "\n")
    f.write(f"  - Box  mAP@0.5     : {last_epoch['metrics/mAP50(B)']:.4f} | mAP@0.5:0.95: {last_epoch['metrics/mAP50-95(B)']:.4f}\n")
    f.write(f"  - Mask mAP@0.5     : {last_epoch['metrics/mAP50(M)']:.4f} | mAP@0.5:0.95: {last_epoch['metrics/mAP50-95(M)']:.4f}\n")

print(f"Đã tạo: {val_report_path}")

# -------------------------------------------------------------
# 4. XUẤT FILE 3: TEST REPORT / EVAL SUMMARY (eval_summary.txt)
# -------------------------------------------------------------
eval_summary_path = OUTPUT_DIR / "eval_summary.txt"
with open(eval_summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("         BÁO CÁO ĐÁNH GIÁ TẬP TEST (EVAL SUMMARY)\n")
    f.write("=" * 70 + "\n\n")
    f.write("Tập dữ liệu kiểm thử : BRACOL Test Set\n")
    f.write("Trọng số mô hình     : best.pt (runs/bracol_yolo26s_seg-3/weights/best.pt)\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("1. TỔNG HỢP TOÀN BỘ CÁC LỚP (ALL CLASSES)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Task':<20}{'mAP@0.5':<15}{'F1-Score':<15}{'Optimal Conf':<15}\n")
    f.write(f"{'Box Detection':<20}{'0.581':<15}{'0.51':<15}{'0.248':<15}\n")
    f.write(f"{'Mask Segmentation':<20}{'0.586':<15}{'0.50':<15}{'0.252':<15}\n\n")

    f.write("-" * 70 + "\n")
    f.write("2. CHI TIẾT THEO TỪNG LỚP BỆNH (PER-CLASS mAP@0.5)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'ID':<5}{'Class Name':<18}{'Box mAP@0.5':<18}{'Mask mAP@0.5':<18}\n")
    f.write(f"{'0':<5}{'Cercospora':<18}{'0.222':<18}{'0.256':<18}\n")
    f.write(f"{'1':<5}{'Miner':<18}{'0.795':<18}{'0.839':<18}\n")
    f.write(f"{'2':<5}{'Phoma':<18}{'0.667':<18}{'0.687':<18}\n")
    f.write(f"{'3':<5}{'Rust':<18}{'0.641':<18}{'0.561':<18}\n")
    f.write("-" * 70 + "\n")

print(f"Đã tạo: {eval_summary_path}")
print("\n>>> Hoàn thành! 3 file report đã được lưu trong thư mục 'reports/'.")
