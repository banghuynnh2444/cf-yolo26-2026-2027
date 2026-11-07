from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.yaml"
RUN = ROOT / "runs" / "bracol_yolo26s_seg-3"

def main():
    model_path = RUN / "weights" / "best.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    lines = ["mode,split,class,images,instances,box_P,box_R,box_mAP50,box_mAP50-95,mask_P,mask_R,mask_mAP50,mask_mAP50-95"]

    for split in ["train", "val", "test"]:
        print(f"\n=======================================================")
        print(f"  EVALUATING SPLIT: {split.upper()}")
        print(f"=======================================================")
        kw = dict(
            data=str(DATA),
            split=split,
            imgsz=1024,
            batch=4,
            workers=0,
            device=0,
            project=str(RUN),
            name=f"eval_{split}",
            exist_ok=True
        )
        r = model.val(**kw)

        nt = r.nt_per_class
        ni = r.nt_per_image

        # Line 'all'
        lines.append(
            f"standard,{split},all,,{int(nt.sum())},"
            + ",".join(f"{v:.4f}" for v in (*r.box.mean_results(), *r.seg.mean_results()))
        )

        # Per-class lines
        for i, c in enumerate(r.ap_class_index):
            vals = (*r.box.class_result(i), *r.seg.class_result(i))
            lines.append(
                f"standard,{split},{model.names[int(c)]},{int(ni[c])},{int(nt[c])},"
                + ",".join(f"{v:.4f}" for v in vals)
            )

    out_csv = RUN / "eval_summary.csv"
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n>>> SAVED TO: {out_csv}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
