"""
Chay evaluate tren train/val/test roi ve report_<split>.png
voi cau hinh mac dinh (khong dung nguong conf rieng cho class nao).
"""

import evaluate
import make_report_tables

if __name__ == "__main__":
    evaluate.main()
    make_report_tables.main()
