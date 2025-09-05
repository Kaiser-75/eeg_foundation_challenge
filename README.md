# EEG Foundation Challenge (2025)

Cross-task transfer learning on EEG:
- **SSL pretrain** on the passive *Surround Suppression* task (SuS)
- **Supervised transfer** on the active *Contrast Change Detection* task (CCD)
- Outputs: **RT** (regression) and **HIT** (classification)

All preprocessing follows the challenge guidance:
- EEG-only → cap/pad to **129 channels**
- **Average reference**
- **Notch** filter at mains harmonics (default **60 Hz**; set **50 Hz** if needed)
- **Band-pass** **0.5–40 Hz**
- **Resample** to **100 Hz**
- **Deterministic sanitization** (replace NaN/Inf with channel median)
- Windowing: **2.0 s** windows (→ **[129, 200]** samples at 100 Hz), clip ±200 µV, z-score per-channel

---

## 1) Dataset Download

The challenge data (HBN-EEG L100) is hosted on S3.

### Requirements
- Python 3.9–3.11
- AWS CLI  
  ```bash
  pip install awscli
