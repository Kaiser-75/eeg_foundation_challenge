# EEG Foundation Challenge (2025)

Self-supervised pretraining on **Surround Suppression (SuS)** and supervised transfer on **Contrast Change Detection (CCD)**.  
All data is processed to **129 channels × 200 samples** (2 s @ 100 Hz).

---

## 1) Install requirements

```bash
pip install -r requirements.txt
```
## 2) Download dataset (mini sets to get started)

Adjust the destination path if needed. The code expects your base folder in EEG_BASE_DIR.

Windows PowerShell
# Create folders
```bash
mkdir -p competition_data/R1
```
```bash
mkdir -p competition_data/R2
```

# Download dataset
```bash
pip install awscli
```
```bash
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R1_mini_L100_bdf competition_data/R1 --no-sign-request
```
```bash
aws s3 cp --recursive s3://nmdatasets/NeurIPS25/R2_mini_L100_bdf competition_data/R2 --no-sign-request
```
Note: Miniset consists of R1-R11 where R5 is val set; R12 is not available which is test set
# Point code to the data root
$env:EEG_BASE_DIR="E:\EEG_foundation\competition_data"

## 3) File description

# dataloader.py

-Loads SuS (for SSL) and CCD (for supervised) EEG files.

-Preprocessing (once per recording): EEG-only → cap/pad to 129 ch, average reference, notch (60 Hz), band-pass 0.5–40 Hz, resample 100 Hz, sanitize NaN/Inf.

-Windows: 2.0 s → shape [129, 200], µV clip ±200, per-channel z-score.

# models.py

-Compact 1D CNN encoder for EEG + projection heads (for SSL) and task heads (RT regression, HIT classification).

# train_simclr.py

-Self-supervised SimCLR on SuS with temporal & spatial augmentations.

-Saves checkpoints to ./checkpoints/.

# train_supervised.py
-Loads the SSL encoder; trains linear probe and/or finetunes end-to-end on CCD.

-Metrics: RT (MAE/R²), HIT (AUC/BAcc).


