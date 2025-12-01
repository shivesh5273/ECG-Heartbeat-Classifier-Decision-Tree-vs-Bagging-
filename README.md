# ECG Heartbeat Classifier — Decision Tree vs Bagging

A small, practical machine learning project that compares a **single Decision Tree** baseline against a **Bagging ensemble of Decision Trees** on the Kaggle *ECG Heartbeat Categorization (MIT-BIH style)* dataset.

The goal is to show how **bagging reduces variance** and often improves generalization on noisy signal classification problems.

---

## What this project covers

- Loading ECG heartbeat dataset (MIT-BIH style, preprocessed beats)
- Dataset evaluation:
  - shape, label counts (class imbalance)
  - missing values & duplicates checks
  - value range checks ([0, 1]) + distribution plots
  - example ECG beat plots per class
- Modeling:
  1. **DecisionTreeClassifier** (baseline)
  2. **BaggingClassifier** (ensemble of decision trees)
- Evaluation:
  - Accuracy
  - Balanced Accuracy (important due to class imbalance)
  - Classification report (precision/recall/F1 per class)
  - Confusion matrix (and optional normalized confusion matrix)

---

## Dataset

From Kaggle: **ECG Heartbeat Categorization Dataset** (MIT-BIH style)

Files used (CSV):
- `mitbih_train.csv`
- `mitbih_test.csv`

Each row represents one ECG beat segment:
- **187 feature columns** = waveform values across timepoints
- **1 label column (last column)** = class label (0–4)

> Note: The dataset is heavily imbalanced (class 0 dominates), so balanced metrics matter.

---

## Key Results 
- Decision Tree is a solid Baseline but can be High-Variance 
- Bagging generally improves overall performance by averaging across many BootStrapped trees
- Because of the class imbalance, accuracy alone can be miss leading - Balanced Accuracy and Per-Class Recall are essential

---

## Notes/Learning 
- Bagging reduces variance and often improves generallization
- Minority classes remain challenging due to imbalance
- Next Upgrades could include:
-   Hyperparameter Tuning (Tree Depth/ Leaf Size)
-   Balanced Random Forest
-   Calibration or Thresholding for specific clinical priorities 
