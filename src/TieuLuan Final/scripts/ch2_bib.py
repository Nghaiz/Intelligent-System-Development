"""Sinh refs/ch2.bib từ bib Chương 2 của Tiểu luận 03, thay ba mục dataset cũ bằng dataset mới."""
import re

from tlf.common import REPORT

OLD = REPORT.parent / "TieuLuan 03" / "refs" / "ch2.bib"
NEW = r"""
@inproceedings{ch2:vsfc2018,
  author = {Nguyen, Kiet Van and Nguyen, Vu Duc and Nguyen, Phu X. V. and Truong, Tham T. H. and Nguyen, Ngan Luu-Thuy},
  title = {{UIT-VSFC}: Vietnamese Students' Feedback Corpus for Sentiment Analysis},
  booktitle = {International Conference on Knowledge and Systems Engineering (KSE)},
  year = {2018}, publisher = {IEEE}
}
@misc{ch2:ds_cdc,
  author = {{Centers for Disease Control and Prevention}},
  title = {{CDC} Diabetes Health Indicators, {BRFSS} 2015},
  howpublished = {UCI Machine Learning Repository, id 891},
  url = {https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators}, urldate = {2026-11-05}
}
@misc{ch2:ds_housing,
  author = {Nguyen, Tien Nhan},
  title = {House Price Prediction Dataset Vietnam - 2024},
  howpublished = {Kaggle}, year = {2024},
  url = {https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024}, urldate = {2026-11-05}
}
@article{ch2:micci2001,
  author = {Micci-Barreca, Daniele},
  title = {A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems},
  journal = {ACM SIGKDD Explorations Newsletter}, volume = {3}, number = {1}, pages = {27--32}, year = {2001}
}
@article{ch2:kaufman2012,
  author = {Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  title = {Leakage in Data Mining: Formulation, Detection, and Avoidance},
  journal = {ACM Transactions on Knowledge Discovery from Data}, volume = {6}, number = {4}, year = {2012}
}
"""

if __name__ == "__main__":
    text = OLD.read_text(encoding="utf-8")
    entries = re.split(r"(?m)^(?=@)", text)
    kept = [e for e in entries if not re.match(r"@\w+\{ch2:ds_(diabetes|house|reviews)", e)]
    (REPORT / "refs" / "ch2.bib").write_text("".join(kept).rstrip() + "\n" + NEW, encoding="utf-8", newline="\n")
