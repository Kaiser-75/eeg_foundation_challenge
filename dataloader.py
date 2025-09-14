from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne

# Suppress MNE info/warnings by default
mne.set_log_level("ERROR")

# =============================================================================
# Organization requirements / shared constants
# =============================================================================
MAX_CH: int    = 129           # keep/pad to 129 EEG channels
OUT_HZ: float  = 100.0         # resample to 100 Hz
WIN_SEC: float = 2.0           # 2.0-second windows for all tasks
OUT_T: int     = int(WIN_SEC * OUT_HZ)   # 200 samples @100 Hz

# Base filters
HP_CUTOFF: float = 0.5
LP_CUTOFF: float = 40.0
LINE_FREQ: int   = 60         

# Anchor for supervised CCD windows
ANCHOR_SHIFT_AFTER_STIM = 0.5   # seconds after stimulus onset

# =============================================================================
# Preprocess configuration (basic only)
# =============================================================================
@dataclass
class PreprocessConfig:
    l_freq: float = HP_CUTOFF
    h_freq: float = LP_CUTOFF
    line_freq: int = LINE_FREQ
    notch: bool = True
    avg_ref: bool = True
    resample_hz: float = OUT_HZ

    # Window-level post steps
    amp_clip_uv: Optional[float] = 600.0
    window_standardize: bool = True  # per-window, per-channel z-score

# =============================================================================
# Helpers
# =============================================================================
def _harmonics(base: int, sfreq: float, upto: int = 6) -> List[float]:
    ny = sfreq / 2.0
    return [base * k for k in range(1, upto + 1) if base * k < ny - 1e-6]

def _pick_eeg_129(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """EEG-only, then cap to 129 channels (deterministic order)."""
    with mne.use_log_level("ERROR"):
        r = raw.copy().pick_types(eeg=True, verbose="ERROR")
    if len(r.ch_names) == 0:
        raise ValueError("No EEG channels present in this recording.")
    n = min(MAX_CH, len(r.ch_names))
    with mne.use_log_level("ERROR"):
        return r.copy().pick(list(range(n)), verbose="ERROR")

def _sanitize_inplace(r: mne.io.BaseRaw) -> None:
    """
    Deterministic sanitization: replace any NaN/Inf in-place with channel median
    (or zero if entire channel is non-finite). Always applied.
    """
    data = r.get_data()  # volts
    finite = np.isfinite(data)
    if finite.all():
        return
    for ci in range(data.shape[0]):
        ch = data[ci]
        good = np.isfinite(ch)
        if not good.all():
            fill = np.median(ch[good]) if good.any() else 0.0
            ch[~good] = fill
    r._data[:] = data  

def _finalize_window(data_uv: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Amplitude clip and per-channel standardize (within the window)."""
    if cfg.amp_clip_uv is not None and cfg.amp_clip_uv > 0:
        np.clip(data_uv, -cfg.amp_clip_uv, cfg.amp_clip_uv, out=data_uv)
    if cfg.window_standardize:
        mean = np.mean(data_uv, axis=1, keepdims=True)
        std = np.std(data_uv, axis=1, keepdims=True) + 1e-6
        data_uv = (data_uv - mean) / std
    return data_uv.astype(np.float32, copy=False)

def _fix_channels(data: np.ndarray, target_ch: int = MAX_CH) -> np.ndarray:
    """Truncate/pad channels (deterministic order) to target_ch."""
    C, T = data.shape
    if C == target_ch:
        return data
    if C > target_ch:
        return data[:target_ch]
    pad = np.zeros((target_ch - C, T), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)

def _preprocess_recording_once(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    """
    Apply *basic* recording-level preprocessing exactly once:
      1) EEG-only → cap to 129
      2) Average reference
      3) Notch @ mains harmonics (60 Hz by default)
      4) Band-pass 0.5–40 Hz
      5) Resample to 100 Hz
      6) Sanitize NaN/Inf deterministically
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

def _crop_window_uv(r_pre: mne.io.BaseRaw, tmin: float, tmax: float) -> np.ndarray:
    """
    Crop a [tmin, tmax) window, convert to µV, and enforce (MAX_CH, OUT_T).
    Padding/truncation is deterministic.
    """
    t0, t1 = float(r_pre.times[0]), float(r_pre.times[-1])
    tmin_ = max(t0, float(tmin))
    tmax_ = min(t1, float(tmax))
    if tmax_ <= tmin_:
        return np.zeros((MAX_CH, OUT_T), dtype=np.float32)
    with mne.use_log_level("ERROR"):
        seg = r_pre.copy().crop(tmin=tmin_, tmax=tmax_, include_tmax=False, verbose="ERROR")
    data = seg.get_data() * 1e6  # µV
    C, T = data.shape
    if T < OUT_T:
        pad = np.zeros((C, OUT_T - T), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=1)
    elif T > OUT_T:
        data = data[:, :OUT_T]
    data = _fix_channels(data, target_ch=MAX_CH)
    return data

def _parse_subject_id(path: Path) -> str:
    # Prefer folder part "sub-XXXX"
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    # Fallback: scan filename
    m = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return m.group(1) if m else "sub-UNKNOWN"

# =============================================================================
# Unlabeled SuS windows (for SSL)
# =============================================================================
class SuSWindowDataset(Dataset):
    """
    Unlabeled 2.0 s windows (1.0 s stride) from surroundSupp/surroundSuppression tasks.
    Basic preprocessing only
    """

    def __init__(
        self,
        base_dir: Path | str,
        releases: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        preprocess: Optional[PreprocessConfig] = None,
        preload: bool = True,
        stride_sec: float = 1.0,
        verbose: str = "INFO",
    ):
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.max_files = max_files
        self.cfg = preprocess or PreprocessConfig()
        self.preload = preload
        self.stride = float(stride_sec)
        self.verbose = verbose

        # Discover EEG files
        pats = [
            "**/eeg/*task-surroundSupp*_eeg.bdf",
            "**/eeg/*task-surroundSupp*_eeg.edf",
            "**/eeg/*task-surroundSupp*_eeg.set",
            "**/eeg/*task-surroundSupp*_eeg.vhdr",
            "**/eeg/*task-surroundSupp*_eeg.fif",
            "**/eeg/*task-surroundSuppression*_eeg.bdf",
            "**/eeg/*task-surroundSuppression*_eeg.edf",
            "**/eeg/*task-surroundSuppression*_eeg.set",
            "**/eeg/*task-surroundSuppression*_eeg.vhdr",
            "**/eeg/*task-surroundSuppression*_eeg.fif",
        ]
        files: List[Path] = []
        for p in pats:
            files.extend(self.base_dir.glob(p))
        files = sorted(set(files))

        if self.releases:
            def _in_release(path: Path) -> bool:
                return any(r in {s.upper() for s in path.parts} for r in self.releases)
            files = [f for f in files if _in_release(f)]

        if max_files and max_files > 0:
            files = files[:max_files]

        self.files: List[Path] = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}

        if (verbose or "").upper() == "INFO":
            if len(files) == 0:
                print(f"[SuSWindowDataset] No SuS EEG found under {self.base_dir}")
            else:
                print(f"[SuSWindowDataset] Found {len(files)} files. Example(s):")
                for ex in files[:8]:
                    print("   -", ex)

        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float]] = []  # (file_idx, start_time)

        # Fast indexing using raw header only
        for i, p in enumerate(self.files):
            raw = self._read_raw(p)
            sfreq = float(raw.info["sfreq"])
            dur = float(raw.n_times) / max(sfreq, 1e-6)
            n = max(0, int((dur - WIN_SEC) // self.stride) + 1)
            for w in range(n):
                self.index.append((i, float(w) * self.stride))

    # ---- raw readers ----
    def _read_raw(self, path: Path) -> mne.io.BaseRaw:
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

    def _load_preprocessed(self, idx: int, path: Path) -> mne.io.BaseRaw:
        if idx in self._cache_pre:
            return self._cache_pre[idx]
        raw = self._read_raw(path)
        r_pre = _preprocess_recording_once(raw, self.cfg)
        if self.preload:
            self._cache_pre[idx] = r_pre
        return r_pre

    # ---- torch Dataset API ----
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        file_idx, t_on = self.index[k]
        r_pre = self._load_preprocessed(file_idx, self.files[file_idx])
        X = _crop_window_uv(r_pre, float(t_on), float(t_on) + WIN_SEC)
        X = _finalize_window(X, self.cfg)
        assert X.shape == (MAX_CH, OUT_T), f"Window shape mismatch: {X.shape}"
        return torch.from_numpy(X), None

    def get_subject(self, k: int) -> str:
        file_idx, _ = self.index[k]
        return self.subjects_for_file[file_idx]

# =============================================================================
# CCD helpers 
# =============================================================================
def _find_ccd_files(base_dir: Path) -> List[Path]:
    pats = [
        "**/eeg/*task-contrastChangeDetection*_eeg.bdf",
        "**/eeg/*task-contrastChangeDetection*_eeg.edf",
        "**/eeg/*task-contrastChangeDetection*_eeg.set",
        "**/eeg/*task-contrastChangeDetection*_eeg.vhdr",
        "**/eeg/*task-contrastChangeDetection*_eeg.fif",
    ]
    files: List[Path] = []
    for p in pats:
        files.extend(Path(base_dir).glob(p))
    return sorted(set(files))

def _events_tsv_for(path: Path) -> Path:
    if "_eeg." not in path.name:
        raise ValueError(f"CCD path does not look like EEG file: {path.name}")
    ev = path.with_name(path.name.replace("_eeg", "_events").rsplit(".", 1)[0] + ".tsv")
    if not ev.exists():
        raise FileNotFoundError(f"Events file not found for {path.name}: {ev}")
    return ev

def _build_trials(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-trial rows with stimulus onset, response onset, RT, and correctness.
    Strict: requires 'feedback' ∈ {smiley_face, sad_face}. If 'feedback' column
    is missing → return empty DF (caller will SKIP and LOG).
    """
    req_cols = {"onset", "value", "event_code"}
    if not req_cols.issubset(set(events.columns)):
        return pd.DataFrame(columns=["stimulus_onset","response_onset","rt_from_stimulus","correct"])

    if "feedback" not in events.columns:
        return pd.DataFrame(columns=["stimulus_onset","response_onset","rt_from_stimulus","correct"])

    df = events.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials    = df[df["value"].eq("contrastTrial_start")].copy().reset_index(drop=True)
    stimuli   = df[df["value"].isin(["left_target", "right_target"])].copy()
    responses = df[df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"]); end = float(tr["next_onset"])

        stim_blk = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        if stim_blk.empty:
            continue
        stim_on = float(stim_blk.iloc[0]["onset"])

        resp_blk = responses[(responses["onset"] >= stim_on) & (responses["onset"] < end)]
        if resp_blk.empty:
            continue
        resp_on = float(resp_blk.iloc[0]["onset"])

        fb = resp_blk.iloc[0].get("feedback", None)
        if fb not in ("smiley_face", "sad_face"):
            continue
        correct = 1 if fb == "smiley_face" else 0

        rows.append({
            "stimulus_onset": stim_on,
            "response_onset": resp_on,
            "rt_from_stimulus": resp_on - stim_on,
            "correct": correct,
        })

    return pd.DataFrame(rows, columns=["stimulus_onset","response_onset","rt_from_stimulus","correct"])

# =============================================================================
# CCD — supervised windows with labels (for downstream finetuning)
# =============================================================================
class CCDWindowDataset(Dataset):
    """
    Supervised CCD 2.0 s windows:
      mode='poststim' : window anchored at stimulus_onset + 0.5 s
      mode='pretrial' : window ending at stimulus_onset (WIN_SEC before)
      mode='both'     : include both anchors

    Labels:
      - rt  (reaction time from stimulus, seconds, float32)
      - hit (1/0 correctness from 'feedback')

    Skips any file that lacks 'feedback' and logs summary (verbose='INFO').
    """
    def __init__(self,
                 base_dir: Path | str,
                 releases: Optional[List[str]] = None,
                 max_files: Optional[int] = None,
                 mode: str = "pretrial",
                 preprocess: Optional[PreprocessConfig] = None,
                 preload: bool = True,
                 verbose: str = "ERROR"):
        assert mode in {"poststim", "pretrial", "both"}
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.max_files = max_files
        self.mode = mode
        self.cfg = preprocess or PreprocessConfig()
        self.preload = preload
        self.verbose = verbose

        files: List[Path] = _find_ccd_files(self.base_dir)

        # Optional release filter
        if self.releases:
            def _in_release(path: Path) -> bool:
                return any(r in {s.upper() for s in path.parts} for r in self.releases)
            files = [f for f in files if _in_release(f)]

        # Optional cap
        if max_files and max_files > 0:
            files = files[:max_files]

        self.files: List[Path] = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}

        if (verbose or "").upper() == "INFO":
            if len(files) == 0:
                print(f"[CCDWindowDataset] No CCD EEG found under {self.base_dir}")
            else:
                print(f"[CCDWindowDataset] Using {len(files)} files. Example(s):")
                for ex in files[:8]:
                    print("   -", ex)

        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float, float, int]] = []  # (file_idx, t_on, rt, hit)
        self.index_modes: List[str] = []                      # "poststim" or "pretrial"

        # ---- Build index w
        skipped_files_no_feedback = 0
        skipped_files_unreadable  = 0
        skipped_trials_unknown_fb = 0
        example_missing: List[Path] = []

        for i, p in enumerate(self.files):
            evp = _events_tsv_for(p)
            try:
                ev = pd.read_csv(evp, sep="\t")
            except Exception:
                skipped_files_unreadable += 1
                if len(example_missing) < 10:
                    example_missing.append(p)
                continue

            tr = _build_trials(ev)  # empty if no feedback column or no valid trials
            if tr.empty:
                if "feedback" not in ev.columns:
                    skipped_files_no_feedback += 1
                    if len(example_missing) < 10:
                        example_missing.append(p)
                continue

            stims = tr["stimulus_onset"].astype(float).to_numpy()
            rts   = tr["rt_from_stimulus"].astype(float).to_numpy()
            hits  = tr["correct"].astype(float).to_numpy()

            if np.isnan(hits).any():
                n_bad = int(np.isnan(hits).sum())
                skipped_trials_unknown_fb += n_bad
                keep = ~np.isnan(hits)
                stims, rts, hits = stims[keep], rts[keep], hits[keep]

            for stim_on, rt, hit in zip(stims, rts, hits.astype(int)):
                if self.mode in {"poststim", "both"}:
                    t_on = float(stim_on) + ANCHOR_SHIFT_AFTER_STIM
                    self.index.append((i, t_on, float(rt), int(hit)))
                    self.index_modes.append("poststim")
                if self.mode in {"pretrial", "both"}:
                    t_on_pre = float(stim_on) - WIN_SEC
                    self.index.append((i, t_on_pre, float(rt), int(hit)))
                    self.index_modes.append("pretrial")

            if preload:
                _ = self._load_preprocessed(i, p)

        if (verbose or "").upper() == "INFO":
            print(f"[CCDWindowDataset] Built {len(self.index)} windows from {len(self.files)} EEG files (mode='{self.mode}').")
            if skipped_files_no_feedback or skipped_files_unreadable or skipped_trials_unknown_fb:
                print("[CCDWindowDataset] Skips → "
                      f"no_feedback_files={skipped_files_no_feedback}, "
                      f"unreadable_files={skipped_files_unreadable}, "
                      f"unknown_feedback_trials={skipped_trials_unknown_fb}")
                if example_missing:
                    print("  Examples of skipped files (up to 10):")
                    for ex in example_missing[:10]:
                        print("   -", ex)

    # ---- readers / preprocess ----
    def _read_raw(self, path: Path) -> mne.io.BaseRaw:
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

    def _load_preprocessed(self, idx: int, path: Path) -> mne.io.BaseRaw:
        if idx in self._cache_pre:
            return self._cache_pre[idx]
        raw = self._read_raw(path)
        r_pre = _preprocess_recording_once(raw, self.cfg)
        if self.preload:
            self._cache_pre[idx] = r_pre
        return r_pre

    # ---- torch Dataset API ----
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        file_idx, t_on, rt, hit = self.index[k]
        r_pre = self._load_preprocessed(file_idx, self.files[file_idx])
        X = _crop_window_uv(r_pre, float(t_on), float(t_on) + WIN_SEC)
        X = _finalize_window(X, self.cfg)
        y = {
            "rt":  torch.tensor(float(rt),  dtype=torch.float32),
            "hit": torch.tensor(int(hit),   dtype=torch.int64),
        }
        assert X.shape == (MAX_CH, OUT_T), f"Window shape mismatch: {X.shape}"
        return torch.from_numpy(X), y

    def get_subject(self, k: int) -> str:
        file_idx, *_ = self.index[k]
        return self.subjects_for_file[file_idx]


