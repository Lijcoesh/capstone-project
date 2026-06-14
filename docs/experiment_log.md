# Experiment log

A running record of the methodological decisions, experiments (including the ones
that did not work), and findings for this capstone. This is the lab notebook that
the paper's Methods / Results / Discussion are derived from.

> Structured per-run numbers are auto-logged to `results/<pipeline>/metrics.csv`;
> each evaluation's full diagnostic report is saved under
> `results/<pipeline>/reports/`. This file is the narrative around them.

---

## 1. Goal

A **comparative study of EEG-only vs. EEG+ECG** feature sets for **seizure
prediction** with a 1-D CNN. Both models are trained on the **same dataset** so the
only difference is the feature set (the ECG channel) — a controlled comparison.

## 2. Dataset

- **Initial:** CHB-MIT (clinical scalp EEG, 23 channels, US/60 Hz). Used to build and
  validate the pipeline. **Dropped** — it is EEG-only and would have forced the two
  arms onto different datasets.
- **Final:** **SeizeIT2** (OpenNeuro ds005873, BIDS). Wearable **behind-the-ear EEG
  (2 ch) + ECG (1 ch)**, 256 Hz, European EMUs (→ 50 Hz mains). Both arms use this:
  EEG-only = 2 channels, EEG+ECG = 3 channels.
- **Why this is better:** same dataset for both arms removes the dataset as a
  confounder; the wearable setting is the clinically relevant research question
  (can we predict from a comfortable device instead of a 23-electrode cap?).

## 3. Task: prediction, not detection

The title says *prediction*, but the first pipeline was actually doing *detection*
(classifying whether a window **is** a seizure). Corrected to true **prediction**:

- **Pre-ictal** (positive): window in `[onset − 30 min, onset)`.
- **Ictal + post-ictal guard** (excluded, dropped): `[onset, onset + duration + 10 min]`.
- **Interictal** (negative): everything else.
- Overlap priority: exclude > pre-ictal > interictal.

Neither CHB-MIT nor SeizeIT2 ships pre-ictal labels; they are derived from the
seizure onset times.

## 4. Pipeline architecture

Three stages per pipeline (clean separation of concerns), with shared logic in
`src/*_common.py` so the EEG and EEG+ECG arms differ only by the ECG channel:

- `preprocess_*` → `data/processed/*_windows.npz`  (shared: `preprocess_common.py`)
- `train_model_*` → `models/seizure_prediction_*/seizure_cnn.pt`  (shared: `model_common.py`)
- `evaluate_*` → `results/seizure_prediction_*/` (metrics.csv, plots, reports)  (shared: `evaluate_common.py`)

## 5. Preprocessing decisions

- **Power-line notch: 50 Hz** + harmonics (Europe), not 60 Hz (CHB-MIT was Boston).
- **Per-window, per-channel z-score** (handles EEG vs ECG scale differences).
- **Sliding windows:** 2 s, 1 s step (512 timepoints at 256 Hz).
- **Interictal subsampling (5:1):** recordings are ~18 h, so keeping every interictal
  window is infeasible (millions of windows) and ~97 % interictal. We keep ALL
  pre-ictal windows and subsample interictal at a fixed ratio (`--interictal-ratio`,
  default 5). Class imbalance is *additionally* handled by positive class weighting
  in the loss.

## 6. Train/test split — evolution

1. **Global chronological 80/20** (first pipeline). With multiple subjects this put
   the last 20 % entirely on the last subject → accidental **cross-subject** test.
2. **Within-subject (subject-aware) 80/20** (current). Per subject, first 80 % of
   their time-ordered windows → train, last 20 % → test, so every subject is in both.
   Standard for seizure prediction. Deterministic, shared by train and evaluate.

## 7. Experiments & findings

### Run A — EEG+ECG, 4 subjects (sub-001…004), cross-subject split, 30 epochs
- F1 **0.185**, precision 0.13, recall 0.31.
- **Cause:** test set was essentially sub-004 (unseen patient). Cross-subject
  prediction from 2–3 wearable channels is the hardest setting. → switched to
  within-subject split.

### Run B — EEG+ECG, 4 subjects, within-subject split, 30 epochs
- F1 **0.318**, precision 0.293, recall 0.347, specificity 0.788.
- AUC-ROC **0.591** (barely above chance), AUC-PR 0.256 (baseline 0.202).
- **Over/underfit: train F1 0.724 vs test F1 0.318 (gap +0.41) → clear overfitting.**
- Per-subject: sub-003 and sub-004 had **0 pre-ictal windows in test** (too few /
  early seizures → all their pre-ictal fell in train).
- **Diagnosis:** too little data + too much model capacity. More epochs would make it
  worse (already overfitting). Threshold tuning capped by the low AUC.
- **Decision:** add more subjects.

### Run C — EEG+ECG, 10 subjects (sub-001…010), within-subject split, 30 epochs
- Dataset: 307,839 windows, 57,177 pre-ictal (4.4:1). Strongly imbalanced **across**
  subjects: sub-002 alone ≈ 44 % of pre-ictal; subjects 003/006/009/010 have a single
  seizure each.
- Aggregate: F1 **0.294**, precision 0.324, recall 0.269. **AUC-ROC 0.495 (= chance)**,
  AUC-PR 0.288 (baseline 0.293). Over/underfit gap +0.367 (train 0.661 vs test 0.294).
- More data slightly reduced overfitting (gap 0.41 → 0.37) but did **not** improve the
  aggregate; global AUC fell to chance.
- **Per-subject heterogeneity is the real story.** Some patients are predictable
  (sub-008 F1 0.64, P 0.71 R 0.57; sub-005 P 0.81), others near-random (sub-009/010
  recall ≈ 0.01). 3 subjects (003/004/006) had 0 pre-ictal in test (few/early seizures).
- **Per-subject AUC: 0.555 ± 0.059** (n=7 subjects with test positives) — clearly above
  the global 0.495, confirming the pooling artifact. Best: sub-005 0.628, sub-008 0.604;
  worst: sub-009 0.428 (below chance). Weak but real within-patient signal.
- **Interpretation:** the global AUC ≈ 0.5 is partly a *pooling artifact* — the model's
  probability scale differs per patient, so pooling subjects destroys global ranking even
  though within-patient signal exists. This is the classic motivation for **personalized
  (per-patient) models** in seizure prediction. A single global model + global threshold
  does not transfer across heterogeneous patients.
- **Decision:** report per-subject performance (AUC + mean ± std) as the honest headline;
  run the EEG-only arm with the same setup for the comparison; do **not** keep chasing a
  higher global F1 (the finding is heterogeneity/personalization, not a tuning problem).

### Comparison — EEG vs EEG+ECG (10 subjects, within-subject, seed 42)
Headline metric: **mean per-subject AUC** (the pooled/global AUC is contaminated).

| Feature set | Mean per-subject AUC |
|---|---|
| EEG-only | 0.560 ± 0.089 |
| EEG+ECG | 0.555 ± 0.059 |
| Δ (EEG+ECG − EEG) | **−0.005 → ECG does not help on average** |

- **Patient-dependent:** ECG helps sub-008 (+0.178) and sub-007 (+0.053); hurts sub-009
  (−0.153) and sub-005 (−0.111).
- EEG+ECG **overfits more** (train F1 0.66 vs EEG 0.44) for no generalization gain — the
  extra channel mainly added memorization capacity, not usable signal.
- **Caveat:** single seed. The −0.005 difference is well within run-to-run/seed noise
  (per-subject std ~0.06–0.09). A **multi-seed** comparison is needed before claiming it.
- Figure: `results/comparison_eeg_vs_eeg_ecg.png`.

## 8. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Dataset | SeizeIT2 (dropped CHB-MIT) | Same dataset for both arms; wearable = the research question |
| Task | Prediction (pre-ictal vs interictal) | Matches the title; derived 30-min pre-ictal window |
| Notch | 50 Hz | European recordings |
| Interictal sampling | 5:1 ratio | 18-h recordings → infeasible/imbalanced otherwise |
| Class imbalance | Positive class weighting in loss | Standard; complements the 5:1 sampling |
| Split | Subject-aware (within-subject) 80/20 | Standard for prediction; avoids accidental cross-subject |
| More data | Added subjects after overfitting was diagnosed | Biggest lever against overfitting |

## 9. Open questions / next steps

- Re-run with 10 subjects (Run C) — does the train/test gap shrink and do all
  subjects get test positives?
- Report **mean ± std over multiple seeds** (single runs are noisy on small test sets).
- If still overfitting: regularization (weight decay / dropout / smaller model).
- Consider an **event-level** metric (does the model flag each seizure event, not just
  individual windows?) — often more meaningful and higher than window-level F1.
- EEG-only vs EEG+ECG comparison figure (the deliverable) once both arms are trained.
