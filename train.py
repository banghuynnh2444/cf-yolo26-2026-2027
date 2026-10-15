from ultralytics import YOLO
import torch


def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26s-seg.pt")

    results = model.train(
        data=r"D:\NCKH2026-2027 - Copy\data.yaml",

        epochs=100,
        imgsz=1024,
        batch=3,

        device=0,

     
        workers=4,

        project=r"D:\NCKH2026-2027\runs",
        name="bracol_yolo26s_seg",

        pretrained=True,
        patience=30,

        save=True,
        plots=True,

        amp=True,
    )

    print("TRAIN DONE")


if __name__ == "__main__":
    main()