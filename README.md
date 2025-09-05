# EEG Foundation Challenge (2025)

Self-supervised pretraining on **Surround Suppression (SuS)** and supervised transfer on **Contrast Change Detection (CCD)**.  
All data is processed to **129 channels × 200 samples** (2 s @ 100 Hz).

---

## 1) Install (one line)

```bash
pip install -r requirements.txt
```
## 2) Download dataset (mini sets to get started)

Adjust the destination path if needed. The code expects your base folder in EEG_BASE_DIR.

Windows PowerShell
# Create folders
mkdir -p E:\EEG_foundation\competition_data\R1
mkdir -p E:\EEG_foundation\competition_data\R2

# Download (no credentials needed)
```pip install awscli
```
```
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R1_mini_L100_bdf E:\EEG_foundation\competition_data\R1 --no-sign-request
```
```
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R2_mini_L100_bdf E:\EEG_foundation\competition_data\R2 --no-sign-request
```

# Point code to the data root
$env:EEG_BASE_DIR="E:\EEG_foundation\competition_data"
