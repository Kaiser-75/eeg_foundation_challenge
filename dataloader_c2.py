from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
import math
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne

# =========================
# Global constants (fixed)
# =========================
mne.set_log_level("ERROR")

MAX_CH: int    = 129           # keep/pad to 129 EEG channels
OUT_HZ: float  = 100.0         # resample to 100 Hz
WIN_SEC: float = 2.0           # 2.0-second windows for all tasks
OUT_T: int     = int(WIN_SEC * OUT_HZ)   # 200 samples @100 Hz

HP_CUTOFF: float = 0.5
LP_CUTOFF: float = 40.0
LINE_FREQ: int   = 60

# For CCD trial parsing
ANCHOR_SHIFT_AFTER_STIM = 0.5  # seconds after stimulus onset (not used here for SSL/C2)

EEG_EXTS = [".bdf", ".edf", ".set", ".vhdr", ".fif"]

TASK_PATTERNS: Dict[str, List[str]] = {
    "RS":  ["RestingState"],
    "MW":  ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"],
    "SuS": ["surroundSupp", "surroundSuppression"],
    "CCD": ["contrastChangeDetection"],
    "SL":  ["seqLearning6target", "seqLearning8target"],
    "SyS": ["symbolSearch"],
}

# =========================
# Preprocess configuration
# =========================
@dataclass
class PreprocessConfig:
    l_freq: float = HP_CUTOFF
    h_freq: float = LP_CUTOFF
    line_freq: int = LINE_FREQ
    notch: bool = True
    avg_ref: bool = True
    resample_hz: float = OUT_HZ
    amp_clip_uv: Optional[float] = 800.0
    window_standardize: bool = True  # per-window, per-channel z-score


def _harmonics(base: int, sfreq: float, upto: int = 6) -> List[float]:
    ny = sfreq / 2.0
    return [base * k for k in range(1, upto + 1) if base * k < ny - 1e-6]

def _parse_subject_id(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    m = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    return m.group(1)

def _events_tsv_for(path: Path) -> Path:
    assert "_eeg." in path.name, f"EEG filename must contain '_eeg': {path.name}"
    return path.with_name(path.name.replace("_eeg", "_events").rsplit(".", 1)[0] + ".tsv")

def _glob_task_files(base_dir: Path, keywords: List[str]) -> List[Path]:
    files: List[Path] = []
    for kw in keywords:
        for ext in EEG_EXTS:
            files.extend(Path(base_dir).glob(f"**/eeg/*task-{kw}*_eeg{ext}"))
    files = sorted(set(files))
    return files

def _read_raw(path: Path) -> mne.io.BaseRaw:
    suf = path.suffix.lower()
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
    raise AssertionError(f"Unsupported EEG file type: {suf}")

def _pick_eeg_129(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    with mne.use_log_level("ERROR"):
        r = raw.copy().pick_types(eeg=True, verbose="ERROR")
    assert len(r.ch_names) > 0, "No EEG channels present in this recording."
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

def _finalize_window(data_uv: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.amp_clip_uv is not None and cfg.amp_clip_uv > 0:
        np.clip(data_uv, -cfg.amp_clip_uv, cfg.amp_clip_uv, out=data_uv)
    if cfg.window_standardize:
        mean = np.mean(data_uv, axis=1, keepdims=True)
        std = np.std(data_uv, axis=1, keepdims=True) + 1e-6
        data_uv = (data_uv - mean) / std
    return data_uv.astype(np.float32, copy=False)

def _fix_channels(data: np.ndarray, target_ch: int = MAX_CH) -> np.ndarray:
    C, T = data.shape
    if C == target_ch:
        return data
    if C > target_ch:
        return data[:target_ch]
    pad = np.zeros((target_ch - C, T), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)

def _preprocess_recording_once(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    with mne.use_log_level("ERROR"):
        r = _pick_eeg_129(raw).copy().load_data()

        if cfg.avg_ref:
            r.set_eeg_reference("average", verbose="ERROR")

        if cfg.notch and cfg.line_freq > 0:
            harms = _harmonics(cfg.line_freq, r.info["sfreq"])
            if len(harms) > 0:
                r.notch_filter(harms, picks=None, verbose="ERROR")

        r.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks=None, verbose="ERROR")

        if cfg.resample_hz and abs(r.info["sfreq"] - cfg.resample_hz) > 1e-6:
            r.resample(cfg.resample_hz, npad="auto")

    _sanitize_inplace(r)
    return r

def _crop_window_uv(r_pre: mne.io.BaseRaw, tmin: float, tmax: float) -> np.ndarray:
    t0, t1 = float(r_pre.times[0]), float(r_pre.times[-1])
    tmin_ = max(t0, float(tmin))
    tmax_ = min(t1, float(tmax))
    if tmax_ <= tmin_:
        data = np.zeros((len(r_pre.ch_names), OUT_T), dtype=np.float32)
        return _fix_channels(data, target_ch=MAX_CH)
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

# =========================
# CCD trial parsing
# =========================
def _build_ccd_trials(events: pd.DataFrame) -> pd.DataFrame:
    req_cols = {"onset", "value", "event_code"}
    assert req_cols.issubset(set(events.columns)), f"CCD events missing required columns: {req_cols}"
    assert "feedback" in events.columns, "CCD events must include 'feedback' column."

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
        if len(stim_blk) == 0:
            continue
        stim_on = float(stim_blk.iloc[0]["onset"])

        resp_blk = responses[(responses["onset"] >= stim_on) & (responses["onset"] < end)]
        if len(resp_blk) == 0:
            continue
        resp_on = float(resp_blk.iloc[0]["onset"])

        fb = resp_blk.iloc[0]["feedback"] if "feedback" in resp_blk.columns else None
        if fb not in ("smiley_face", "sad_face"):
            continue
        correct = 1 if fb == "smiley_face" else 0

        rows.append({
            "stimulus_onset": stim_on,
            "response_onset": resp_on,
            "rt_from_stimulus": resp_on - stim_on,
            "correct": correct,
        })

    cols = ["stimulus_onset","response_onset","rt_from_stimulus","correct"]
    return pd.DataFrame(rows, columns=cols)

# =========================
# SSL: unlabeled windows across tasks (includes CCD pretrial only)
# =========================
@dataclass
class AllTasksConfig:
    base_dir: Path
    preprocess: PreprocessConfig
    stride_sec: float = 1.0
    preload: bool = False
    include_RS: bool = True
    include_MW: bool = True
    include_SuS: bool = True
    include_SL: bool = True
    include_SyS: bool = True
    include_CCD_pretrial: bool = True

class AllTasksWindows(Dataset):
    """
    Unlabeled 2.0 s windows (1.0 s stride) from multiple tasks for SSL.
    Includes CCD pretrial windows only (no poststim).
    Returns: torch.FloatTensor [C, T]
    """
    def __init__(self, cfg: AllTasksConfig):
        self.cfg = cfg
        self.base_dir = Path(cfg.base_dir)

        files: List[Tuple[str, Path]] = []
        if cfg.include_RS:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["RS"]):  files.append(("RS", p))
        if cfg.include_MW:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["MW"]):  files.append(("MW", p))
        if cfg.include_SuS:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["SuS"]): files.append(("SuS", p))
        if cfg.include_SL:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["SL"]):  files.append(("SL", p))
        if cfg.include_SyS:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["SyS"]): files.append(("SyS", p))
        if cfg.include_CCD_pretrial:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["CCD"]): files.append(("CCD_pre", p))

        assert len(files) > 0, "No EEG files found for selected tasks."

        self.files: List[Tuple[str, Path]] = files
        self.subjects_for_file: Dict[int, str] = {i: _parse_subject_id(p) for i, (_, p) in enumerate(self.files)}
        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        self.index: List[Tuple[int, float]] = []  # (file_idx, t_on)

        stride = float(cfg.stride_sec)
        for i, (task, p) in enumerate(self.files):
            if task == "CCD_pre":
                evp = _events_tsv_for(p)
                ev = pd.read_csv(evp, sep="\t")
                tr = _build_ccd_trials(ev)
                assert len(tr) > 0, f"CCD file has no valid trials (feedback missing or parse failed): {p}"
                stims = tr["stimulus_onset"].astype(float).to_numpy()
                for stim_on in stims:
                    t_on = float(stim_on) - float(WIN_SEC)  # pretrial window
                    self.index.append((i, t_on))
            else:
                raw = _read_raw(p)
                sf = float(raw.info["sfreq"])
                dur = float(raw.n_times) / max(1e-6, sf)
                n = max(0, int((dur - WIN_SEC) // stride) + 1)
                for w in range(n):
                    self.index.append((i, float(w) * stride))

            if cfg.preload:
                _ = self._load_preprocessed(i, p)

        assert len(self.index) > 0, "No windows indexed. Check base_dir and task flags."

    def _load_preprocessed(self, idx: int, path: Path) -> mne.io.BaseRaw:
        if idx in self._cache_pre:
            return self._cache_pre[idx]
        raw = _read_raw(path)
        r_pre = _preprocess_recording_once(raw, self.cfg.preprocess)
        if self.cfg.preload:
            self._cache_pre[idx] = r_pre
        return r_pre

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> torch.Tensor:
        file_idx, t_on = self.index[k]
        _, p = self.files[file_idx]
        r_pre = self._load_preprocessed(file_idx, p)
        X = _crop_window_uv(r_pre, float(t_on), float(t_on) + float(WIN_SEC))
        X = _finalize_window(X, self.cfg.preprocess)
        assert X.shape == (MAX_CH, OUT_T), f"Window shape mismatch: {X.shape}"
        return torch.from_numpy(X)

    def get_subject(self, k: int) -> str:
        file_idx, _ = self.index[k]
        return self.subjects_for_file[file_idx]

# =========================
# C2: subject-centric dataset (one item = one subject pack)
# =========================
@dataclass
class SubjectPackConfig:
    base_dir: Path
    manifest_csv: Path
    preprocess: PreprocessConfig
    stride_sec: float = 1.0
    max_windows_per_subject: int = 512
    preload: bool = False
    include_RS: bool = True
    include_MW: bool = True
    include_SuS: bool = True
    include_SL: bool = True
    include_SyS: bool = True
    include_CCD_pretrial: bool = True

class SubjectPackDataset(Dataset):
    """
    One sample = one subject:
      returns (windows [N,129,200], demo [3], targets [4], subject_id str)

    - windows are pooled across selected tasks
    - CCD uses pretrial anchor windows only
    - max N per subject capped by config
    - age normalization to be set via set_age_normalization(mean, std)
    """
    def __init__(self, cfg: SubjectPackConfig):
        self.cfg = cfg
        self.base_dir = Path(cfg.base_dir)
        self.manifest_csv = Path(cfg.manifest_csv)

        assert self.base_dir.exists(), f"Base directory not found: {self.base_dir}"
        assert self.manifest_csv.exists(), f"Manifest CSV not found: {self.manifest_csv}"

        df = pd.read_csv(self.manifest_csv)
        must = ["subject_id", "age", "sex", "handedness", "p", "internalizing", "externalizing", "attention"]
        assert set(must).issubset(set(df.columns)), f"Manifest missing columns. Need: {must}"
        self.manifest = df.copy()
        self.manifest["subject_id"] = self.manifest["subject_id"].astype(str)

        # Discover EEG files per subject
        file_by_subj: Dict[str, List[Tuple[str, Path]]] = {}

        def add_files(flag_name: str, key: str):
            if getattr(cfg, f"include_{flag_name}"):
                for p in _glob_task_files(self.base_dir, TASK_PATTERNS[key]):
                    sid = _parse_subject_id(p)
                    file_by_subj.setdefault(sid, []).append((flag_name if flag_name != "CCD_pretrial" else "CCD_pre", p))

        add_files("RS", "RS")
        add_files("MW", "MW")
        add_files("SuS", "SuS")
        add_files("SL", "SL")
        add_files("SyS", "SyS")
        if cfg.include_CCD_pretrial:
            for p in _glob_task_files(self.base_dir, TASK_PATTERNS["CCD"]):
                sid = _parse_subject_id(p)
                file_by_subj.setdefault(sid, []).append(("CCD_pre", p))

        manifest_subjects = set(self.manifest["subject_id"].tolist())
        self.subject_ids: List[str] = []
        for sid in sorted(file_by_subj.keys()):
            if sid in manifest_subjects and len(file_by_subj[sid]) > 0:
                self.subject_ids.append(sid)

        assert len(self.subject_ids) > 0, "No subjects matched between manifest and EEG files."

        # Build flattened file list and owner map
        self.files: List[Tuple[str, Path]] = []
        self.file_owner_subj: List[str] = []
        for sid in self.subject_ids:
            for (task, p) in file_by_subj[sid]:
                self.files.append((task, p))
                self.file_owner_subj.append(sid)

        # Build window indices per subject
        self.subj_windows: Dict[str, List[Tuple[int, float]]] = {}
        stride = float(cfg.stride_sec)
        for i, (task, p) in enumerate(self.files):
            sid = self.file_owner_subj[i]
            if task == "CCD_pre":
                evp = _events_tsv_for(p)
                ev = pd.read_csv(evp, sep="\t")
                tr = _build_ccd_trials(ev)
                assert len(tr) > 0, f"CCD file has no valid trials (feedback missing or parse failed): {p}"
                stims = tr["stimulus_onset"].astype(float).to_numpy()
                for stim_on in stims:
                    t_on = float(stim_on) - float(WIN_SEC)
                    self.subj_windows.setdefault(sid, []).append((i, t_on))
            else:
                raw = _read_raw(p)
                sf = float(raw.info["sfreq"])
                dur = float(raw.n_times) / max(1e-6, sf)
                n = max(0, int((dur - WIN_SEC) // stride) + 1)
                for w in range(n):
                    self.subj_windows.setdefault(sid, []).append((i, float(w) * stride))

        for sid in self.subject_ids:
            assert sid in self.subj_windows and len(self.subj_windows[sid]) > 0, f"Subject {sid} has zero windows."

        # Preload preprocessed raws if requested
        self._cache_pre: Dict[int, mne.io.BaseRaw] = {}
        if cfg.preload:
            for i, (_, p) in enumerate(self.files):
                _ = self._load_preprocessed(i, p)

        # Demographics & targets maps
        self._demo: Dict[str, Tuple[float, int, int]] = {}
        self._targ: Dict[str, Tuple[float, float, float, float]] = {}
        for _, row in self.manifest.iterrows():
            sid = str(row["subject_id"])
            age = float(row["age"])
            sex_raw = str(row["sex"]).strip().upper()
            hand_raw = str(row["handedness"]).strip().upper()
            sex_bin = 1 if sex_raw in ("F", "FEMALE") else 0
            if hand_raw in ("L", "LEFT"):
                hand_code = -1
            elif hand_raw in ("R", "RIGHT"):
                hand_code = +1
            else:
                hand_code = 0
            self._demo[sid] = (age, sex_bin, hand_code)
            self._targ[sid] = (
                float(row["p"]),
                float(row["internalizing"]),
                float(row["externalizing"]),
                float(row["attention"]),
            )

        self.age_mean: Optional[float] = None
        self.age_std: Optional[float] = None

    # ----- public API -----
    def set_age_normalization(self, mean: float, std: float) -> None:
        assert std > 0.0, "age_std must be > 0"
        self.age_mean = float(mean)
        self.age_std = float(std)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _load_preprocessed(self, file_idx: int, path: Path) -> mne.io.BaseRaw:
        if file_idx in self._cache_pre:
            return self._cache_pre[file_idx]
        raw = _read_raw(path)
        r_pre = _preprocess_recording_once(raw, self.cfg.preprocess)
        if self.cfg.preload:
            self._cache_pre[file_idx] = r_pre
        return r_pre

    def _demo_vec(self, sid: str) -> np.ndarray:
        age, sex_bin, hand_code = self._demo[sid]
        if self.age_mean is None or self.age_std is None:
            age_z = float(age)
        else:
            age_z = (float(age) - float(self.age_mean)) / float(self.age_std)
        return np.asarray([age_z, float(sex_bin), float(hand_code)], dtype=np.float32)

    def _targets_vec(self, sid: str) -> np.ndarray:
        return np.asarray(self._targ[sid], dtype=np.float32)

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        wlist = self.subj_windows[sid]

        # Deterministic cap (downsample windows evenly)
        if len(wlist) > self.cfg.max_windows_per_subject:
            step = max(1, math.floor(len(wlist) / self.cfg.max_windows_per_subject))
            wlist = wlist[::step][: self.cfg.max_windows_per_subject]

        Xs: List[np.ndarray] = []
        for (file_idx, t_on) in wlist:
            _, p = self.files[file_idx]
            r_pre = self._load_preprocessed(file_idx, p)
            X = _crop_window_uv(r_pre, float(t_on), float(t_on) + float(WIN_SEC))
            X = _finalize_window(X, self.cfg.preprocess)
            assert X.shape == (MAX_CH, OUT_T), f"Window shape mismatch: {X.shape} for {p.name}"
            Xs.append(X)

        X = np.stack(Xs, axis=0)  # [N, C, T]
        demo = self._demo_vec(sid)
        targ = self._targets_vec(sid)

        return (
            torch.from_numpy(X),    # [N, 129, 200]
            torch.from_numpy(demo), # [3]
            torch.from_numpy(targ), # [4]
            sid,                    # subject id (string)
        )

# =========================
# Collate for SubjectPack
# =========================
def collate_subject_pack(batch):
    assert len(batch) >= 1, "Empty batch."
    if len(batch) == 1:
        return batch[0]
    Xs, demos, targs, sids = [], [], [], []
    for (X, d, y, sid) in batch:
        Xs.append(X)      # [Ni, C, T]
        demos.append(d)   # [3]
        targs.append(y)   # [4]
        sids.append(sid)
    return Xs, torch.stack(demos, dim=0), torch.stack(targs, dim=0), sids

# =========================
# Convenience builders
# =========================
def make_ssl_dataset(
    base_dir: Path | str = Path(os.environ.get("EEG_BASE_DIR", "competition_data")),
    preprocess: Optional[PreprocessConfig] = None,
    stride_sec: float = 1.0,
    preload: bool = False,
) -> AllTasksWindows:
    cfg = AllTasksConfig(
        base_dir=Path(base_dir),
        preprocess=preprocess or PreprocessConfig(),
        stride_sec=stride_sec,
        preload=preload,
        include_RS=True, include_MW=True, include_SuS=True,
        include_SL=True, include_SyS=True, include_CCD_pretrial=True,
    )
    return AllTasksWindows(cfg)

def make_subject_dataset(
    manifest_csv: Path | str = Path(os.environ.get("EEG_BASE_DIR", "competition_data")) / "manifest_c2.csv",
    base_dir: Path | str = Path(os.environ.get("EEG_BASE_DIR", "competition_data")),
    preprocess: Optional[PreprocessConfig] = None,
    stride_sec: float = 1.0,
    max_windows_per_subject: int = 512,
    preload: bool = False,
) -> SubjectPackDataset:
    cfg = SubjectPackConfig(
        base_dir=Path(base_dir),
        manifest_csv=Path(manifest_csv),
        preprocess=preprocess or PreprocessConfig(),
        stride_sec=stride_sec,
        max_windows_per_subject=max_windows_per_subject,
        preload=preload,
        include_RS=True, include_MW=True, include_SuS=True,
        include_SL=True, include_SyS=True, include_CCD_pretrial=True,
    )
    return SubjectPackDataset(cfg)
