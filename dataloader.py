

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset

mne.set_log_level("ERROR")

# =========================
# Shared constants
# =========================
MAX_CH: int     = 129      # cap/pad EEG channels to 129
OUT_HZ: float   = 100.0    # competition downsampled rate
# 2.0 s default (Challenge 1 requires ~2 s windows)
DEFAULT_WIN_SEC = 2.0
OUT_T: int      = int(DEFAULT_WIN_SEC * OUT_HZ)

# For CCD supervised anchoring
ANCHOR_SHIFT_AFTER_STIM = 0.5  

# =========================
# Preprocess config
# =========================
@dataclass
class PreprocessConfig:
    l_freq: float = 0.5
    h_freq: float = 40.0
    line_freq: int = 60
    notch: bool = True
    avg_ref: bool = True
    resample_hz: float = OUT_HZ

    # per-window post steps
    amp_clip_uv: Optional[float] = 800.0
    window_standardize: bool = True

# =========================
# Helpers
# =========================
def _harmonics(base: int, sfreq: float, upto: int = 6) -> List[float]:
    ny = sfreq / 2.0
    return [base * k for k in range(1, upto + 1) if base * k < ny - 1e-6]

def _pick_eeg_129(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Pick EEG only, then cap deterministically at 129 channels."""
    with mne.use_log_level("ERROR"):
        r = raw.copy().pick_types(eeg=True, verbose="ERROR")
    if len(r.ch_names) == 0:
        raise ValueError("No EEG channels present.")
    n = min(MAX_CH, len(r.ch_names))
    with mne.use_log_level("ERROR"):
        return r.copy().pick(list(range(n)), verbose="ERROR")

def _sanitize_inplace(r: mne.io.BaseRaw) -> None:
    """Replace any NaN/Inf with channel median (or 0 if entirely non-finite)."""
    data = r.get_data()
    finite = np.isfinite(data)
    if finite.all():
        return
    for ci in range(data.shape[0]):
        ch = data[ci]
        good = np.isfinite(ch)
        if not good.all():
            fill = np.median(ch[good]) if good.any() else 0.0
            ch[~good] = fill
    r._data[:] = data  # in-place

def _preprocess_recording_once(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    """
    Basic, deterministic preprocessing (once per file):
      1) EEG-only → cap to 129
      2) Average reference
      3) Notch at mains harmonics
      4) 0.5–40 Hz band-pass
      5) Resample to 100 Hz
      6) Sanitize NaN/Inf
    """
    with mne.use_log_level("ERROR"):
        r = _pick_eeg_129(raw).copy().load_data()
        if cfg.avg_ref:
            r.set_eeg_reference("average", verbose="ERROR")
        if cfg.notch and cfg.line_freq > 0:
            harms = _harmonics(cfg.line_freq, r.info["sfreq"])
            if harms:
                r.notch_filter(harms, picks=None, verbose="ERROR")
        r.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks=None, verbose="ERROR")
        if cfg.resample_hz and abs(r.info["sfreq"] - cfg.resample_hz) > 1e-6:
            r.resample(cfg.resample_hz, npad="auto")
    _sanitize_inplace(r)
    return r

def _fix_channels(data: np.ndarray, target_ch: int = MAX_CH) -> np.ndarray:
    """Pad/truncate channels deterministically to target_ch."""
    C, T = data.shape
    if C == target_ch:
        return data
    if C > target_ch:
        return data[:target_ch]
    pad = np.zeros((target_ch - C, T), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)

def _crop_window_uv_len(r_pre: mne.io.BaseRaw, tmin: float, win_sec: float) -> np.ndarray:
    """
    Crop [tmin, tmin+win_sec), convert to µV, enforce (MAX_CH, OUT_T_local).
    Deterministic pad/truncation.
    """
    OUT_T_local = int(round(win_sec * OUT_HZ))
    t0, t1 = float(r_pre.times[0]), float(r_pre.times[-1])
    tmax = min(t1, tmin + win_sec)
    if tmax <= tmin:
        return np.zeros((MAX_CH, OUT_T_local), dtype=np.float32)
    with mne.use_log_level("ERROR"):
        seg = r_pre.copy().crop(tmin=float(tmin), tmax=float(tmax), include_tmax=False, verbose="ERROR")
    data = seg.get_data() * 1e6  # µV
    C, T = data.shape
    if T < OUT_T_local:
        pad = np.zeros((C, OUT_T_local - T), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=1)
    elif T > OUT_T_local:
        data = data[:, :OUT_T_local]
    data = _fix_channels(data, target_ch=MAX_CH)
    return data

def _finalize_window(data_uv: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Amplitude clip and per-channel standardization within the window."""
    if cfg.amp_clip_uv is not None and cfg.amp_clip_uv > 0:
        np.clip(data_uv, -cfg.amp_clip_uv, cfg.amp_clip_uv, out=data_uv)
    if cfg.window_standardize:
        mean = np.mean(data_uv, axis=1, keepdims=True)
        std = np.std(data_uv, axis=1, keepdims=True) + 1e-6
        data_uv = (data_uv - mean) / std
    return data_uv.astype(np.float32, copy=False)

def _parse_subject_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    m = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return m.group(1) if m else "sub-UNKNOWN"

# ---------- file readers ----------
def _read_raw(path: Path) -> mne.io.BaseRaw:
    suf = path.suffix.lower()
    with mne.use_log_level("ERROR"):
        if suf == ".bdf":
            return mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
        if suf == ".set":
            return mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
        if suf == ".edf":
            return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        if suf == ".vhdr":
            return mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
        if suf == ".fif":
            return mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    raise RuntimeError(f"Unsupported EEG file type: {suf}")

# =========================
# SuS unlabeled windows (for SSL)
# =========================
class SuSWindowDataset(Dataset):
    """
    Unlabeled windows from Surround Suppression.
    Use `win_sec=6.0` for SSL pretraining (ok), `stride_sec` typically 1.0–3.0 s.
    """
    def __init__(
        self,
        base_dir: Path | str,
        releases: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        preprocess: Optional[PreprocessConfig] = None,
        preload: bool = False,
        stride_sec: float = 3.0,
        win_sec: float = 6.0,
        verbose: str = "INFO",
    ):
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.max_files = max_files
        self.cfg = preprocess or PreprocessConfig()
        self.preload = bool(preload)
        self.stride = float(stride_sec)
        self.win_sec = float(win_sec)
        self.verbose = verbose

        # discover files
        pats = [
            "**/eeg/*task-surroundSupp*_eeg.bdf",
            "**/eeg/*task-surroundSupp*_eeg.set",
            "**/eeg/*task-surroundSupp*_eeg.edf",
            "**/eeg/*task-surroundSupp*_eeg.vhdr",
            "**/eeg/*task-surroundSupp*_eeg.fif",
            "**/eeg/*task-surroundSuppression*_eeg.bdf",
            "**/eeg/*task-surroundSuppression*_eeg.set",
            "**/eeg/*task-surroundSuppression*_eeg.edf",
            "**/eeg/*task-surroundSuppression*_eeg.vhdr",
            "**/eeg/*task-surroundSuppression*_eeg.fif",
        ]
        files: List[Path] = []
        for p in pats:
            files.extend(self.base_dir.glob(p))
        files = sorted(set(files))

        # release filter
        if self.releases:
            def _in_release(path: Path) -> bool:
                parts = {s.upper() for s in path.parts}
                return any(r in parts for r in self.releases)
            files = [f for f in files if _in_release(f)]

        if max_files and max_files > 0:
            files = files[:max_files]

        self.files: List[Path] = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}
        if (verbose or "").upper() == "INFO":
            print(f"[SuSWindowDataset] files={len(files)}  win={self.win_sec:.1f}s stride={self.stride:.1f}s")
            for ex in files[:8]:
                print("   -", ex)

        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float]] = []

        # fast indexing from raw header
        for i, p in enumerate(self.files):
            raw = _read_raw(p)
            sfreq = float(raw.info["sfreq"])
            dur = float(raw.n_times) / max(sfreq, 1e-6)
            n = max(0, int((dur - self.win_sec) // self.stride) + 1)
            for w in range(n):
                self.index.append((i, float(w) * self.stride))
            if self.preload:
                r_pre = _preprocess_recording_once(raw, self.cfg)
                self._cache_pre[i] = r_pre

    def __len__(self) -> int:
        return len(self.index)

    def _load_pre(self, idx: int, path: Path) -> mne.io.BaseRaw:
        if idx in self._cache_pre:
            return self._cache_pre[idx]
        r_pre = _preprocess_recording_once(_read_raw(path), self.cfg)
        if self.preload:
            self._cache_pre[idx] = r_pre
        return r_pre

    def __getitem__(self, k: int):
        file_idx, t_on = self.index[k]
        r_pre = self._load_pre(file_idx, self.files[file_idx])
        X = _crop_window_uv_len(r_pre, float(t_on), self.win_sec)
        X = _finalize_window(X, self.cfg)
        OUT_T_local = int(round(self.win_sec * OUT_HZ))
        assert X.shape == (MAX_CH, OUT_T_local), f"Window shape mismatch: {X.shape}"
        return torch.from_numpy(X), None

    def get_subject(self, k: int) -> str:
        file_idx, _ = self.index[k]
        return self.subjects_for_file[file_idx]

# =========================
# CCD supervised
# =========================
def _find_ccd_files(base_dir: Path) -> List[Path]:
    pats = [
        "**/eeg/*task-contrastChangeDetection*_eeg.bdf",
        "**/eeg/*task-contrastChangeDetection*_eeg.set",
        "**/eeg/*task-contrastChangeDetection*_eeg.edf",
        "**/eeg/*task-contrastChangeDetection*_eeg.vhdr",
        "**/eeg/*task-contrastChangeDetection*_eeg.fif",
    ]
    files: List[Path] = []
    for p in pats:
        files.extend(Path(base_dir).glob(p))
    return sorted(set(files))

def _events_tsv_for(path: Path) -> Path:
    # map ..._eeg.ext → ..._events.tsv (BIDS)
    stem = path.name
    if "_eeg." not in stem:
        raise ValueError(f"CCD EEG filename missing '_eeg': {stem}")
    ev = path.with_name(stem.replace("_eeg", "_events").rsplit(".", 1)[0] + ".tsv")
    return ev

class CCDWindowDataset(Dataset):
    """
    Supervised CCD windows with **stimulus-anchored** cropping and RT label:
      mode='poststim': start = stim_on + 0.5 s
      mode='pretrial': end   = stim_on (start = stim_on - win_sec)
      mode='both'   : include both anchors

    Label:
      - 'rt' : reaction time from stimulus to first button press inside the trial
    """
    def __init__(
        self,
        base_dir: Path | str,
        releases: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        mode: str = "poststim",
        preprocess: Optional[PreprocessConfig] = None,
        preload: bool = False,
        win_sec: float = 2.0,
        verbose: str = "INFO",
    ):
        assert mode in {"poststim", "pretrial", "both"}
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.max_files = max_files
        self.mode = mode
        self.cfg = preprocess or PreprocessConfig()
        self.preload = bool(preload)
        self.win_sec = float(win_sec)
        self.verbose = verbose

        files: List[Path] = _find_ccd_files(self.base_dir)

        if self.releases:
            def _in_release(path: Path) -> bool:
                parts = {s.upper() for s in path.parts}
                return any(r in parts for r in self.releases)
            files = [f for f in files if _in_release(f)]

        if max_files and max_files > 0:
            files = files[:max_files]

        self.files: List[Path] = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}

        if (verbose or "").upper() == "INFO":
            print(f"[CCDWindowDataset] files={len(files)}  mode={mode}  win={self.win_sec:.1f}s")
            for ex in files[:8]:
                print("   -", ex)

        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        # index holds (file_idx, t_on, rt)
        self.index: List[Tuple[int, float, float]] = []

        # Build index (strict)
        skipped_files_no_events = 0
        skipped_files_no_trials = 0
        skipped_trials_no_button = 0

        for i, p in enumerate(self.files):
            ev_tsv = _events_tsv_for(p)
            if not ev_tsv.exists():
                skipped_files_no_events += 1
                continue

            ev = pd.read_csv(ev_tsv, sep="\t")
            if "onset" not in ev.columns or "value" not in ev.columns:
                skipped_files_no_events += 1
                continue

            df = ev.copy()
            df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
            df = df.dropna(subset=["onset"]).sort_values("onset", kind="mergesort").reset_index(drop=True)

            
            trials = df[df["value"].eq("contrastTrial_start")].copy().reset_index(drop=True)
            if trials.empty or len(trials) < 2:
                skipped_files_no_trials += 1
                continue
            trials["next_onset"] = trials["onset"].shift(-1)
            trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)
            if trials.empty:
                skipped_files_no_trials += 1
                continue

            stimuli   = df[df["value"].isin(["left_target", "right_target"])].copy()
            responses = df[df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

            for _, tr in trials.iterrows():
                start = float(tr["onset"]); end = float(tr["next_onset"])

                # first stimulus within the trial
                stim_blk = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
                if stim_blk.empty:
                    continue

                for _, srow in stim_blk.iterrows():
                    stim_on = float(srow["onset"])
                    resp_blk = responses[(responses["onset"] >= stim_on) & (responses["onset"] < end)]
                    if resp_blk.empty:
                        skipped_trials_no_button += 1
                        continue
                    resp_on = float(resp_blk.iloc[0]["onset"])
                    rt = max(0.0, resp_on - stim_on)

                    if self.mode in {"poststim", "both"}:
                        t_on = stim_on + ANCHOR_SHIFT_AFTER_STIM
                        self.index.append((i, float(t_on), float(rt)))
                    if self.mode in {"pretrial", "both"}:
                        t_on_pre = stim_on - self.win_sec
                        self.index.append((i, float(t_on_pre), float(rt)))

            if self.preload:
                r_pre = _preprocess_recording_once(_read_raw(p), self.cfg)
                self._cache_pre[i] = r_pre

        if (verbose or "").upper() == "INFO":
            print(f"[CCDWindowDataset] built windows={len(self.index)} | "
                  f"skipped: no_events_files={skipped_files_no_events}, "
                  f"no_trial_files={skipped_files_no_trials}, "
                  f"trials_no_button={skipped_trials_no_button}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_pre(self, idx: int, path: Path) -> mne.io.BaseRaw:
        if idx in self._cache_pre:
            return self._cache_pre[idx]
        r_pre = _preprocess_recording_once(_read_raw(path), self.cfg)
        if self.preload:
            self._cache_pre[idx] = r_pre
        return r_pre

    def __getitem__(self, k: int):
        file_idx, t_on, rt = self.index[k]
        r_pre = self._load_pre(file_idx, self.files[file_idx])
        X = _crop_window_uv_len(r_pre, float(t_on), self.win_sec)
        X = _finalize_window(X, self.cfg)
        OUT_T_local = int(round(self.win_sec * OUT_HZ))
        assert X.shape == (MAX_CH, OUT_T_local), f"Window shape mismatch: {X.shape}"
        y = {"rt": torch.tensor(float(rt), dtype=torch.float32)}
        return torch.from_numpy(X), y

    def get_subject(self, k: int) -> str:
        file_idx, *_ = self.index[k]
        return self.subjects_for_file[file_idx]
