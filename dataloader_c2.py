from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne

mne.set_log_level("ERROR")

# =============================================================================
# Shared constants 
# =============================================================================
MAX_CH: int    = 129
OUT_HZ: float  = 100.0
WIN_SEC: float = 2.0
OUT_T: int     = int(WIN_SEC * OUT_HZ)  # 200 samples @100 Hz

HP_CUTOFF: float = 0.5
LP_CUTOFF: float = 40.0
LINE_FREQ: int   = 60

# For CCD anchors
ANCHOR_SHIFT_AFTER_STIM = 0.5  # for poststim supervised if used

# Allowed EEG file extensions 
EEG_EXTS = {".bdf", ".edf", ".set", ".vhdr", ".fif"}


# =============================================================================
# Preprocess configuration
# =============================================================================
@dataclass
class PreprocessConfig:
    l_freq: float = HP_CUTOFF
    h_freq: float = LP_CUTOFF
    line_freq: int = LINE_FREQ
    notch: bool = True
    avg_ref: bool = True
    resample_hz: float = OUT_HZ

    # Window post-steps
    amp_clip_uv: Optional[float] = 800.0      
    window_standardize: bool = True           # per-window, per-channel z-score


# =============================================================================
# Helpers
# =============================================================================
def _keep_eeg_files(paths: List[Path]) -> List[Path]:
    return [p for p in paths if p.suffix.lower() in EEG_EXTS]

def _glob_many(base: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        out.extend(base.glob(p))
    return sorted(set(_keep_eeg_files(out)))

def _harmonics(base: int, sfreq: float, upto: int = 6) -> List[float]:
    ny = sfreq / 2.0
    return [base * k for k in range(1, upto + 1) if base * k < ny - 1e-6]

def _pick_eeg_129(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    with mne.use_log_level("ERROR"):
        r = raw.copy().pick_types(eeg=True, verbose="ERROR")
    if len(r.ch_names) == 0:
        raise ValueError("No EEG channels present.")
    n = min(MAX_CH, len(r.ch_names))
    with mne.use_log_level("ERROR"):
        return r.copy().pick(list(range(n)), verbose="ERROR")

def _sanitize_inplace(r: mne.io.BaseRaw) -> None:
    data = r.get_data()  # volts
    finite = np.isfinite(data)
    if finite.all():
        return
    for ci in range(data.shape[0]):
        ch = data[ci]
        good = np.isfinite(ch)
        fill = np.median(ch[good]) if good.any() else 0.0
        ch[~good] = fill
    r._data[:] = data

def _preprocess_recording_once(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
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
    C, T = data.shape
    if C == target_ch:
        return data
    if C > target_ch:
        return data[:target_ch]
    pad = np.zeros((target_ch - C, T), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)

def _finalize_window(data_uv: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.amp_clip_uv is not None and cfg.amp_clip_uv > 0:
        np.clip(data_uv, -cfg.amp_clip_uv, cfg.amp_clip_uv, out=data_uv)
    if cfg.window_standardize:
        mean = np.mean(data_uv, axis=1, keepdims=True)
        std = np.std(data_uv, axis=1, keepdims=True) + 1e-6
        data_uv = (data_uv - mean) / std
    return data_uv.astype(np.float32, copy=False)

def _crop_window_uv(r_pre: mne.io.BaseRaw, tmin: float, tmax: float) -> np.ndarray:
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
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    m = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return m.group(1) if m else "sub-UNKNOWN"


# =============================================================================
# File discovery
# =============================================================================
def _filter_by_releases(files: List[Path], releases: Optional[List[str]]) -> List[Path]:
    if not releases:
        return files
    R = {r.upper() for r in releases}
    keep = []
    for f in files:
        parts = {p.upper() for p in f.parts}
        if any(r in parts for r in R):
            keep.append(f)
    return keep

def _find_rs_files(base: Path) -> List[Path]:
    return _glob_many(base, [
        "**/eeg/*task-RestingState*_eeg.*"
    ])

def _find_mw_files(base: Path) -> List[Path]:
    pats = []
    for film in ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]:
        pats += [f"**/eeg/*task-{film}*_eeg.*"]
    return _glob_many(base, pats)

def _find_sus_files(base: Path) -> List[Path]:
    return _glob_many(base, [
        "**/eeg/*task-surroundSupp*_eeg.*",
        "**/eeg/*task-surroundSuppression*_eeg.*",
    ])

def _find_ccd_files(base: Path) -> List[Path]:
    return _glob_many(base, [
        "**/eeg/*task-contrastChangeDetection*_eeg.*"
    ])

def _find_sl_files(base: Path) -> List[Path]:
    return _glob_many(base, [
        "**/eeg/*task-seqLearning6target*_eeg.*",
        "**/eeg/*task-seqLearning8target*_eeg.*",
    ])

def _find_sys_files(base: Path) -> List[Path]:
    return _glob_many(base, [
        "**/eeg/*task-symbolSearch*_eeg.*"
    ])


# =============================================================================
# CCD event utils
# =============================================================================
def _events_tsv_for(path: Path) -> Path:
    return path.with_name(path.name.replace("_eeg", "_events").rsplit(".", 1)[0] + ".tsv")

def _extract_ccd_stim_onsets(ev: pd.DataFrame) -> List[float]:
    if "onset" not in ev.columns or "value" not in ev.columns:
        return []
    df = ev.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset")
    stim = df[df["value"].isin(["left_target", "right_target"])]
    return stim["onset"].astype(float).tolist()

def _build_ccd_trials(ev: pd.DataFrame) -> pd.DataFrame:
    """Return trials with stimulus_onset, response_onset, rt, and correct (if feedback present)."""
    need_cols = {"onset", "value"}
    if not need_cols.issubset(set(ev.columns)):
        return pd.DataFrame(columns=["stimulus_onset","response_onset","rt_from_stimulus","correct"])
    df = ev.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)

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
        if isinstance(fb, str):
            correct = 1 if fb == "smiley_face" else (0 if fb == "sad_face" else np.nan)
        else:
            correct = np.nan

        rows.append({
            "stimulus_onset": stim_on,
            "response_onset": resp_on,
            "rt_from_stimulus": resp_on - stim_on,
            "correct": correct,
        })
    return pd.DataFrame(rows, columns=["stimulus_onset","response_onset","rt_from_stimulus","correct"])


# =============================================================================
# Low-level raw reader
# =============================================================================
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


# =============================================================================
# SSL dataset (foundation pretraining across all tasks)
#   - RS/MW/SuS/SL/SyS: sliding windows
#   - CCD: pretrial windows from stimulus onsets (no feedback dependency)
# =============================================================================
class SSLWindowDataset(Dataset):
    def __init__(
        self,
        base_dir: Path | str,
        releases: Optional[List[str]] = None,
        preprocess: Optional[PreprocessConfig] = None,
        stride_sec: float = 1.0,
        preload: bool = False,
        verbose: str = "INFO",
    ):
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.cfg = preprocess or PreprocessConfig()
        self.stride = float(stride_sec)
        self.preload = preload
        self.verbose = verbose

        # discover files per task
        files_by_task: Dict[str, List[Path]] = {
            "RS":  _find_rs_files(self.base_dir),
            "MW":  _find_mw_files(self.base_dir),
            "SuS": _find_sus_files(self.base_dir),
            "CCD": _find_ccd_files(self.base_dir),
            "SL":  _find_sl_files(self.base_dir),
            "SyS": _find_sys_files(self.base_dir),
        }
        if self.releases:
            for k in files_by_task:
                files_by_task[k] = _filter_by_releases(files_by_task[k], self.releases)

        # flatten
        self.files_by_task = files_by_task
        self.files: List[Path] = []
        self.task_for_file: Dict[int, str] = {}
        for t, flist in files_by_task.items():
            for p in flist:
                self.task_for_file[len(self.files)] = t
                self.files.append(p)

        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}

        # caches and indices
        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float]] = []  # (file_idx, start_time)
        self.task_of_index: List[str] = []

        # Build indices:
        # 1) Non-CCD: slide across duration
        for i, p in enumerate(self.files):
            t = self.task_for_file[i]
            if t == "CCD":
                continue
            raw = _read_raw(p)
            if self.preload:
                self._cache_pre[i] = _preprocess_recording_once(raw, self.cfg)
                r_pre = self._cache_pre[i]
                dur = float(r_pre.n_times) / float(r_pre.info["sfreq"])
            else:
                sf = float(raw.info["sfreq"]); dur = float(raw.n_times) / max(sf, 1e-6)
            n = max(0, int((dur - WIN_SEC) // self.stride) + 1)
            for w in range(n):
                self.index.append((i, float(w) * self.stride))
                self.task_of_index.append(t)

        # 2) CCD: pretrial anchors from stimulus onsets (no feedback)
        ccd_files = self.files_by_task["CCD"]
        added_ccd = 0
        zero_stim_files = 0
        for p in ccd_files:
            i = self.files.index(p)
            evp = _events_tsv_for(p)
            ev  = pd.read_csv(evp, sep="\t")
            stims = _extract_ccd_stim_onsets(ev)
            if len(stims) == 0:
                zero_stim_files += 1
                continue
            for stim_on in stims:
                t_on = float(stim_on) - WIN_SEC
                if t_on < 0:
                    continue
                self.index.append((i, t_on))
                self.task_of_index.append("CCD")
                added_ccd += 1
            if self.preload:
                self._cache_pre[i] = _preprocess_recording_once(_read_raw(p), self.cfg)

        if (self.verbose or "").upper() == "INFO":
            counts: Dict[str, int] = {}
            for t in ["RS","MW","SuS","CCD","SL","SyS"]:
                counts[t] = sum(1 for tt in self.task_of_index if tt == t)
            print(f"[SSLWindowDataset] windows={len(self.index)} | per-task {counts}")
            if zero_stim_files > 0:
                print(f"[SSLWindowDataset][CCD] files with no usable stimuli (skipped): {zero_stim_files}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> torch.Tensor:
        file_idx, t_on = self.index[k]
        if file_idx in self._cache_pre:
            r_pre = self._cache_pre[file_idx]
        else:
            r_pre = _preprocess_recording_once(_read_raw(self.files[file_idx]), self.cfg)
            if self.preload:
                self._cache_pre[file_idx] = r_pre
        X = _crop_window_uv(r_pre, float(t_on), float(t_on) + WIN_SEC)
        X = _finalize_window(X, self.cfg)
        return torch.from_numpy(X)

    def get_subject(self, k: int) -> str:
        file_idx, _ = self.index[k]
        return self.subjects_for_file[file_idx]

    def get_task(self, k: int) -> str:
        return self.task_of_index[k]


# =============================================================================
# CCD supervised dataset (for Challenge-1 finetune)
#   - mode='poststim' : window at stimulus + 0.5 s
#   - mode='pretrial' : window ending at stimulus (WIN_SEC before)
#   - mode='both'     : include both
#   Labels:
#     y['rt']  = reaction time (sec) from stimulus→response (float32)
#     y['hit'] = 1/0 if feedback present (else omitted from selection; we keep only trials with response)
# =============================================================================
class CCDWindowDataset(Dataset):
    def __init__(
        self,
        base_dir: Path | str,
        releases: Optional[List[str]] = None,
        mode: str = "poststim",
        preprocess: Optional[PreprocessConfig] = None,
        preload: bool = False,
        verbose: str = "INFO",
    ):
        assert mode in {"poststim","pretrial","both"}
        self.base_dir = Path(base_dir)
        self.releases = [r.upper() for r in releases] if releases else None
        self.mode = mode
        self.cfg = preprocess or PreprocessConfig()
        self.preload = preload
        self.verbose = verbose

        files = _find_ccd_files(self.base_dir)
        files = _filter_by_releases(files, self.releases)
        self.files = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, p in enumerate(self.files)}

        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float, float, int]] = []  # (file_idx, t_on, rt, hit)

        skipped_no_trials = 0
        for i, p in enumerate(self.files):
            ev = pd.read_csv(_events_tsv_for(p), sep="\t")
            trials = _build_ccd_trials(ev)  # requires response to be present
            if trials.empty:
                skipped_no_trials += 1
                continue

            stims = trials["stimulus_onset"].astype(float).to_numpy()
            rts   = trials["rt_from_stimulus"].astype(float).to_numpy()
            # hit: if feedback missing, set to 0 (benign)
            if "correct" in trials.columns:
                hits = trials["correct"].to_numpy()
                hits = np.where(np.isnan(hits), 0, hits).astype(int)
            else:
                hits = np.zeros_like(rts, dtype=int)

            for stim_on, rt, hit in zip(stims, rts, hits):
                if self.mode in {"poststim","both"}:
                    t_on = float(stim_on) + ANCHOR_SHIFT_AFTER_STIM
                    self.index.append((i, t_on, float(rt), int(hit)))
                if self.mode in {"pretrial","both"}:
                    t_on2 = float(stim_on) - WIN_SEC
                    self.index.append((i, t_on2, float(rt), int(hit)))

            if self.preload:
                self._cache_pre[i] = _preprocess_recording_once(_read_raw(p), self.cfg)

        if (self.verbose or "").upper() == "INFO":
            print(f"[CCDWindowDataset] windows={len(self.index)} | files={len(self.files)} | skipped_no_trials={skipped_no_trials}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        file_idx, t_on, rt, hit = self.index[k]
        if file_idx in self._cache_pre:
            r_pre = self._cache_pre[file_idx]
        else:
            r_pre = _preprocess_recording_once(_read_raw(self.files[file_idx]), self.cfg)
            if self.preload:
                self._cache_pre[file_idx] = r_pre
        X = _crop_window_uv(r_pre, float(t_on), float(t_on) + WIN_SEC)
        X = _finalize_window(X, self.cfg)
        y = {"rt": torch.tensor(float(rt), dtype=torch.float32),
             "hit": torch.tensor(int(hit),  dtype=torch.long)}
        return torch.from_numpy(X), y

    def get_subject(self, k: int) -> str:
        file_idx, *_ = self.index[k]
        return self.subjects_for_file[file_idx]


# =============================================================================
def make_ssl_dataset(
    base_dir: Path | str,
    releases: Optional[List[str]] = None,
    preprocess: Optional[PreprocessConfig] = None,
    stride_sec: float = 1.0,
    preload: bool = False,
    verbose: str = "INFO",
) -> SSLWindowDataset:
    return SSLWindowDataset(
        base_dir=base_dir,
        releases=releases,
        preprocess=preprocess,
        stride_sec=stride_sec,
        preload=preload,
        verbose=verbose,
    )

def make_ccd_supervised_dataset(
    base_dir: Path | str,
    releases: Optional[List[str]] = None,
    mode: str = "poststim",
    preprocess: Optional[PreprocessConfig] = None,
    preload: bool = False,
    verbose: str = "INFO",
) -> CCDWindowDataset:
    return CCDWindowDataset(
        base_dir=base_dir,
        releases=releases,
        mode=mode,
        preprocess=preprocess,
        preload=preload,
        verbose=verbose,
    )
