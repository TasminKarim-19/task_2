# task_2
## BIOT-style Pipeline (CHB-MIT) — Preprocessing, Training, Evaluation, Interpretability


![EEG](https://img.shields.io/badge/EEG-CHB--MIT-blue)
![Model](https://img.shields.io/badge/Model-BIOT%20Transformer-purple)
![Task](https://img.shields.io/badge/Task-Seizure%20Detection-red)
![Logging](https://img.shields.io/badge/Tracking-TensorBoard-orange)
![Metrics](https://img.shields.io/badge/Metrics-AUC--PR%20%7C%20AUROC%20%7C%20BalAcc-green)


This project follows the **BIOT idea**: make EEG from different sources **compatible** before feeding it into a **single transformer model**.  
For CHB-MIT, I convert raw EDF recordings into a unified windowed representation, then train BIOT in two settings: **from scratch** and **fine-tuning** from the **official pretrained BIOT checkpoint**.

---

###  1) Data Preparation (CHB-MIT → BIOT-windows)

**Goal:** transform heterogeneous raw EEG into a **fixed-format**, BIOT-compatible input.

**Steps**
1. **Load raw EDF** per patient/session.
2. **Segment EEG into fixed windows**  
   - Window length: **10s**  
   - With **overlap** (to capture seizure boundaries better)
3. **Label each window** using seizure annotations  
   - Window label ∈ {**seizure**, **non-seizure**}  
   - Label determined by overlap with annotated seizure intervals
4. **Harmonize signals (BIOT- compatibility)**
   - Fixed montage / consistent channel set (map non-matching channels)
   - **Resample** to a common sampling rate
   - **Per-channel 95th-percentile normalization**  
     (scaling to reduce amplitude mismatch across subjects/sessions)

Output: a dataset of standardized EEG windows ready for BIOT training.

---

###  2) Training Settings

I trained BIOT under two required regimes:

#### (1) Training from Scratch
- Random initialization
- Learns CHB-MIT seizure patterns directly from standardized windows

#### (2) Fine-tuning from Official BIOT Pretrained Checkpoint
- Starts from pretrained BIOT weights
- Adapts the representation to CHB-MIT seizure detection

---

###  3) Evaluation Strategy (Imbalanced Seizure Detection)

CHB-MIT is **extremely imbalanced** (very few seizure windows).  
So metrics must reflect **rare-positive** performance.

**Reported Metrics**
- **AUC-PR (Primary)**  
  Most meaningful when positives are rare (focuses on precision/recall).
- **AUROC (Secondary)**  
  Threshold-free separability measure.
- **Balanced Accuracy (Thresholded metric)**  
  Avoids being dominated by majority class.

**Loss Function**
- **Focal Loss** for seizure detection  
  Reduces the impact of many easy negatives and emphasizes harder examples—aligned with BIOT’s motivation for imbalanced regimes.

---

###  4) Experiment Tracking (TensorBoard)

All runs are logged to TensorBoard:
- Train/Val/Test loss
- AUC-PR, AUROC, Balanced Accuracy
- (Optional) learning rate, gradient norms, checkpoint stats

---

### 5) Interpretability (Attention Visualization)

The BIOT codebase does not directly expose attention weights, so I visualized attention using a **QK-based reconstruction**:

$$
\mathrm{Attention}=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)
$$

**What I logged**
- Attention maps from **early**, **middle**, and **late** transformer layers
- Logged directly into TensorBoard for side-by-side comparison across training settings

**Summary observation**
- After fine-tuning, attention becomes more **structured** and **task-focused** in deeper layers, consistent with specialization toward seizure-relevant patterns.

---

##  Summary

- BIOT-compatible **windowing + harmonization** on CHB-MIT
- Two training regimes: **scratch** vs **fine-tune pretrained**
- Imbalance-aware evaluation: **AUC-PR** as primary metric
- **Focal loss** for rare seizure positives
- Attention interpretability via **QK-based attention reconstruction** and TensorBoard logging


