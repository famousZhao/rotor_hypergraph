"""Shared configuration for the rotor hypergraph paper experiments."""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
FIGURES = PROJ / "figures"
TABLES = PROJ / "tables"
CACHE = PROJ / "results" / "cache"
for _p in (RESULTS, FIGURES, TABLES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Raw data lives outside the repo (read-only sources).  Set ROTOR_DATA_ROOT
# to the directory that contains the rotor .mat files and the MFPT folder.
import os
_DATA_ROOT = Path(os.environ.get("ROTOR_DATA_ROOT", "../data"))
D_ROTOR180 = _DATA_ROOT / "180data_new_select_denoised.mat"
D_ROTOR360 = _DATA_ROOT / "360rotor_faults.mat"
D_MFPT = _DATA_ROOT / "MFPT"

# MFPT bearing set.  Baseline and the constant-load outer-race files were
# recorded at 97656 Hz, the variable-load files at 48828 Hz.  Everything is
# decimated to the lower rate so that no classifier can separate the classes
# by their sampling rate instead of by the fault.
MFPT_FS = 48828.0
MFPT_DECIM = 2
N_MFPT_POINTS = 2048
N_MFPT_PER_CLASS = 45
MFPT_CLASSES = ["Normal", "InnerRace", "OuterRace"]

# ---------------------------------------------------------------- signal
# The gas-turbine scaled rotor testbed standard document gives 3300 rpm and
# 10240 Hz.  The older scripts on disk assume 50 kHz; FS is kept here as a
# single knob so the setup section can be corrected in one place.
FS = 10240.0
N_POINTS = 2048

# Four measured classes, in the order used by the source .mat file.
CLASSES = ["Normal", "Rub", "Unbalance", "Misalignment"]
CLASSES_CN = ["正常", "碰摩", "不平衡", "不对中"]

# Eight-class set for the secondary (simulated coupling) study.
CLASSES8 = CLASSES + [
    "Rub+Unbalance",
    "Rub+Misalignment",
    "Unbalance+Misalignment",
    "Rub+Unbalance+Misalignment",
]

# ---------------------------------------------------------------- method defaults
# Faithful to the author's reference implementation (symbolic_entropy_filter.py).
SYMBOLS_P = 6           # number of symbols p
E_FACTOR = 1.2          # threshold e = E_FACTOR * std(x)
TAU_MAX = 150           # delay range 1..TAU_MAX
BUTTER_ORDER = 3
BUTTER_CUTOFF = 0.1

# Recurrence network
REC_PERCENTILE = 5.0    # recurrence rate (percentile of pairwise distances)

# NI-Louvain
REMOVE_RATIO = 0.75     # fraction of recurrence edges removed before clustering
OVERLAP_STRENGTH = 0.3

# VMD
VMD_K = 3
VMD_ALPHA = 2000.0

SEED = 20250929
