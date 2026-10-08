# NSL-KDD Dataset Directory

This directory should contain the NSL-KDD dataset files required for training and evaluating the intrusion detection models.

## Expected Files

Please place the following two files directly into this `data/` folder:

1. **`KDDTrain+.txt`** (Full NSL-KDD training set)
2. **`KDDTest+.txt`** (Full NSL-KDD test set)

Use the full training and test files above. The smaller `KDDTrain+_20Percent.txt` and
`KDDTest-21.txt` files are different splits and are not interchangeable with the requested
full dataset files; renaming them can make the experiment and reported dataset sizes misleading.

## Where to Download

The NSL-KDD dataset is publicly available from official research repositories:

- **University of New Brunswick (UNB) Canadian Institute for Cybersecurity:**
  https://www.unb.ca/cic/datasets/nsl.html
- **Kaggle Mirror:**
  https://www.kaggle.com/datasets/hassan06/nslkdd

Download and extract the archive, then copy `KDDTrain+.txt` and `KDDTest+.txt` into this folder:

```
intrusion-detection-mini-project/
└── data/
    ├── README.md
    ├── KDDTrain+.txt   <-- Place here
    └── KDDTest+.txt    <-- Place here
```

## Dataset Format

NSL-KDD files are comma-separated values (CSV format without header) containing 43 columns:

- Columns 1–41: Network traffic features (e.g., `duration`, `protocol_type`, `service`, `flag`, etc.)
- Column 42: Attack type label (e.g., `normal`, `neptune`, `smurf`, etc.)
- Column 43: Difficulty level score (integer 0–21)

The loader also accepts 42-column files without the difficulty score. It maps known attack
labels into the five classes and stops with the unexpected label if it encounters an unknown
name, so the mapping can be reviewed instead of silently misclassifying a record.
