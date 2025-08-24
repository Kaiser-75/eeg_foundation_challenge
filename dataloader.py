# ============================================================
# CHANGE THIS PATH 
DATASET_ROOT = r"E:\EEG_foundation\dataset"
# ============================================================

from pathlib import Path
import warnings
import mne
from mne_bids import BIDSPath, read_raw_bids

# Supported EEG file extensions
EEG_EXTS = {".edf", ".set", ".fif", ".vhdr"}

def list_recordings(tasks=None, dataset_root=None):
    """
    Return a sorted list of file paths for all recordings under all hbn_bids_* folders.
    If tasks is provided (e.g., ["surroundSupp", "contrastChangeDetection"]),
    only return files whose name contains 'task-<name>'.
    """
    root = Path(dataset_root or DATASET_ROOT)
    files = []
    for f in root.glob("hbn_bids_*/*/eeg/*"):
        if f.suffix.lower() in EEG_EXTS:
            if tasks:
                name_l = f.name.lower()
                if not any(("task-" + t).lower() in name_l for t in tasks):
                    continue
            files.append(f)
    files.sort()
    return files

def parse_bids_from_filename(filepath: Path):
    """
    Parse subject, task (and optionally session/run if present) from a BIDS-like filename,
    and infer the BIDS root for read_raw_bids.
    """
    name_parts = {}
    for piece in filepath.name.split("_"):
        if "-" in piece:
            k, v = piece.split("-", 1)
            # strip extension fragments
            name_parts[k] = v.split(".")[0]
    subject = name_parts.get("sub", filepath.parent.parent.name.replace("sub-",""))
    task    = name_parts.get("task", None)
    session = name_parts.get("ses", None)
    run     = name_parts.get("run", None)
    bids_root = filepath.parents[2]  
    return subject, task, session, run, bids_root

def load_raw(filepath, suppress_set_warning=True):
    """
    Load a single recording as an MNE Raw object and return (raw, meta).
    No preprocessing is applied here.

    Note: For EEGLAB .set files, MNE preloads data into memory by design.
    """
    fp = Path(filepath)
    subject, task, session, run, bids_root = parse_bids_from_filename(fp)
    bids = BIDSPath(
        subject=subject,
        task=task,
        session=session,
        run=run,
        datatype="eeg",
        root=str(bids_root),
    )

    if suppress_set_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = read_raw_bids(bids, verbose=False)
    else:
        raw = read_raw_bids(bids, verbose=False)

    meta = {
        "filepath": str(fp),
        "release": bids_root.name,
        "subject": subject,
        "task": task,
        "session": session,
        "run": run,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
    }
    return raw, meta

def iter_recordings(tasks=None, dataset_root=None, suppress_set_warning=True):
    """
    Generator over recordings. Yields (raw, meta) for each file.
    """
    for fp in list_recordings(tasks=tasks, dataset_root=dataset_root):
        yield load_raw(fp, suppress_set_warning=suppress_set_warning)
