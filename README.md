# cf-yolo26-2026-2027

YOLO26s-seg phân đoạn bệnh lá cà phê (BRACOL), 4 lớp: Cercospora, Miner, Phoma, Rust.
Nhãn polygon được tạo từ nhãn bbox bằng SAM2, sau đó train YOLO26s-seg với cấu hình mặc định, không ưu tiên lớp nào và không dùng ngưỡng conf riêng cho phoma.

## Cấu hình
- yolo26s-seg, imgsz 1024, batch 3, 100 epoch, patience 30, AMP, close_mosaic 10
- Không oversample, không trọng số lớp
- Augment mặc định của Ultralytics: mosaic 1.0, fliplr 0.5, hsv 0.015/0.7/0.4
- GPU RTX 5050 Laptop 8 GB, workers 4, cache False

## Dataset (không đưa lên repo)
| Tập | Ảnh | Cercospora | Miner | Phoma | Rust |
|---|---|---|---|---|---|
| train | 1213 | 132 | 170 | 1308 | 4178 |
| valid | 351 | 39 | 111 | 254 | 1174 |
| test | 180 | 29 | 59 | 107 | 652 |

Để dataset ở `BRACOL_SEG_SAM/` (gồm `train/`, `valid/`, `test/`) và sửa `path` trong `data.yaml`.
Các thư mục `BRACOL_REVIEWED/`, `BRACOL_BBOX_BACKUP/` và repo sam2 cũng không đưa lên repo.

## Cấu trúc
| File | Mục đích |
|---|---|
| `bbox_to_polygon_sam2.py` | Chuyển nhãn bbox sang polygon bằng SAM2 (chạy trong thư mục clone https://github.com/facebookresearch/sam2, checkpoint `sam2.1_hiera_base_plus.pt`) |
| `train.py` | Train yolo26s-seg rồi đánh giá và vẽ bảng kết quả |
| `evaluate.py` | Đánh giá `best.pt` trên train/val/test, ra `eval_summary.csv` |
| `make_report_tables.py` | Vẽ bảng `report_train/val/test.png` từ `eval_summary.csv` |
| `export_reports.py` | Chạy `evaluate.py` rồi `make_report_tables.py` |
| `test.py` | Đánh giá và dự đoán ảnh trên tập test |
| `runs/bracol_yolo26s_seg-3/` | `weights/best.pt`, `results.csv`, biểu đồ, confusion matrix, `eval_*`, bảng kết quả |

## Chạy
```bash
python bbox_to_polygon_sam2.py
python train.py
python export_reports.py
python test.py
```

## Kết quả
best.pt là epoch 73, train đủ 100 epoch (khoảng 2.5 giờ). Đánh giá với cấu hình mặc định (imgsz 1024, không ngưỡng riêng cho lớp nào).

Mask mAP50:

| Lớp | Train | Val | Test |
|---|---|---|---|
| all | 0.742 | 0.595 | 0.607 |
| Cercospora | 0.384 | 0.197 | 0.247 |
| Miner | 0.891 | 0.787 | 0.829 |
| Phoma | 0.912 | 0.737 | 0.716 |
| Rust | 0.783 | 0.659 | 0.636 |

Mask mAP50-95: train 0.531, val 0.410, test 0.410.

Số liệu đầy đủ (box và mask P, R, mAP50, mAP50-95): `runs/bracol_yolo26s_seg-3/eval_summary.csv`, bảng `report_train/val/test.png`.

## Ghi chú
- Cercospora yếu nhất: recall ~0.10–0.23, ít dữ liệu (132 instance trong tập train).
- Phoma precision trên test ~0.41, còn nhiều báo động giả.
- Rust mask mAP50-95 ~0.31 trên test, vết bệnh nhỏ.
