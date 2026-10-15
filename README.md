# cf-yolo26-2026-2027

YOLO26s-seg phân đoạn bệnh lá cà phê (BRACOL), 4 lớp: Cercospora, Miner, Phoma, Rust.

- `bbox_to_polygon_sam2.py`: chuyển nhãn bbox sang polygon bằng SAM2 (chạy trong thư mục clone https://github.com/facebookresearch/sam2, checkpoint `sam2.1_hiera_base_plus.pt`).
- `train.py`: train yolo26s-seg, imgsz 1024, batch 3, 100 epoch.
- `test.py`: đánh giá và dự đoán trên tập test.
- `runs/`: kết quả train và test, `best.pt` ở `runs/bracol_yolo26s_seg-3/weights/`.

Dataset (`BRACOL_REVIEWED`, `BRACOL_BBOX_BACKUP`, `BRACOL_SEG_SAM`), repo sam2 và các file weights tải về không đưa lên repo.
