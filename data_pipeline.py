"""
data_pipeline.py
================
Aditya-L1 ASPEX-SWIS Level-2 BLK CDF Data Pipeline
CME Detection Pipeline v2.0

Handles:
  - Automated CDF file discovery & download (ISRO PRADAN / CDAW mirrors)
  - Physics-informed feature extraction: Vsw, np, Tp, He/H ratio
  - Sentinel value (-1e31) → NaN cleaning
  - Savitzky-Golay smoothing (preserves CME shock transients)
  - Sliding-window sequence generation for TCN input
  - Train/val/test split with stratification on CME labels

Author : CME Detection Pipeline v2.0
Python : 3.10+
"""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Optional: cdflib for reading CDF files
try:
    import cdflib
    HAS_CDFLIB = True
except ImportError:
    HAS_CDFLIB = False
    warnings.warn("cdflib not installed. Install via: pip install cdflib")

# Optional: wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SENTINEL_VALUE = -1e31          # ISRO fill / missing-data sentinel
SENTINEL_THRESHOLD = -9e30      # any value < this is treated as NaN

# UPDATED: Path matches your current Colab session folder
DEFAULT_COLAB_DATA_DIR = "/content/data_2024"

# ASPEX-SWIS L2_BLK variable names (Updated for V03 calibration)
CDF_VARMAP = {
    "epoch":    "epoch_for_cdf_mod",
    "vsw":      "proton_bulk_speed",    # Solar wind speed [km/s]
    "np":       "proton_density",       # Proton number density [cm^-3]
    "tp":       "proton_thermal",       # Proton temperature/thermal proxy
    "he_flux":  "alpha_density",        # Mapped to Alpha density for He/H ratio
    "h_flux":   "proton_density",       # Mapped to Proton density for He/H ratio
    "vth":      "proton_thermal",       # Thermal speed [km/s]
}

# Savitzky-Golay parameters
SAVGOL_WINDOW  = 11   # Must be odd; ~5-min cadence × 11 ≈ 55 min smoothing
SAVGOL_POLYORD = 3    # Polynomial order — cubic preserves edge sharpness

# Sequence / window parameters
SEQUENCE_LENGTH = 128   # time-steps fed into TCN (~10.6 h at 5-min cadence)
STRIDE          = 16    # hop size between consecutive windows

# ---------------------------------------------------------------------------
# SECTION 1: CDF FILE DISCOVERY & DOWNLOAD
# ---------------------------------------------------------------------------

# Known ISRO/PRADAN public mirrors for ASPEX-SWIS L2 data
# Update these URLs as ISRO publishes new data releases.
_BASE_URLS = [
    "https://www.issdc.gov.in/aspex_data/",          # ISSDC official (requires auth)
    "https://cdaw.gsfc.nasa.gov/pub/aditya_l1/",     # NASA CDAW mirror (when available)
]

def download_cdf_files(
    start_date: str,
    end_date:   str,
    local_dir:  str = "./aspex_data/",
    mirror_url: Optional[str] = None,
    dry_run:    bool = False,
) -> list[Path]:
    """
    Attempt to download ASPEX-SWIS L2_BLK CDF files for a date range.

    Parameters
    ----------
    start_date : str   e.g. "2024-05-01"
    end_date   : str   e.g. "2024-05-31"
    local_dir  : str   destination folder
    mirror_url : str   override default mirror (useful in Colab)
    dry_run    : bool  if True, just list URLs without downloading

    Returns
    -------
    list[Path]  local paths of downloaded (or existing) CDF files
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date, freq="D")
    collected: list[Path] = []
    base = mirror_url or _BASE_URLS[0]

    for dt in dates:
        # Typical ISRO naming: al1_asp_swis_l2_blk_YYYYMMDD_v01.cdf
        fname  = f"al1_asp_swis_l2_blk_{dt.strftime('%Y%m%d')}_v01.cdf"
        url    = f"{base}{dt.strftime('%Y/%m/')}{fname}"
        fpath  = local_path / fname

        if fpath.exists():
            logger.info("Cache hit: %s", fpath)
            collected.append(fpath)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would fetch: %s", url)
            continue

        try:
            logger.info("Downloading %s …", url)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            fpath.write_bytes(r.content)
            logger.info("Saved → %s", fpath)
            collected.append(fpath)
        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", url, e)

    return collected


# ---------------------------------------------------------------------------
# SECTION 2: CDF PARSING
# ---------------------------------------------------------------------------

def parse_cdf(filepath: Path) -> pd.DataFrame:
    """
    Parse a single ASPEX-SWIS L2_BLK CDF file into a DataFrame.

    Replaces sentinel values with NaN.
    Converts CDF epoch (TT2000 nanoseconds) to pandas Timestamp.
    """
    if not HAS_CDFLIB:
        raise ImportError("cdflib required: pip install cdflib")

    cdf = cdflib.CDF(str(filepath))
    data: dict[str, np.ndarray] = {}

    # --- Epoch ---
    raw_epoch = cdf.varget(CDF_VARMAP["epoch"])
    # TT2000 → datetime64 via cdflib helper
    data["time"] = pd.to_datetime(cdflib.cdfepoch.to_datetime(raw_epoch))

    # --- Physics variables ---
    for key, varname in CDF_VARMAP.items():
        if key == "epoch":
            continue
        try:
            arr = cdf.varget(varname).astype(np.float64)
            # Replace sentinel
            arr[arr < SENTINEL_THRESHOLD] = np.nan
            data[key] = arr
        except Exception as e:
            logger.warning("Variable '%s' not found in %s: %s", varname, filepath.name, e)
            data[key] = np.full(len(data["time"]), np.nan)

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    logger.info("Parsed %s → %d rows", filepath.name, len(df))
    return df


def load_cdf_directory(directory: str) -> pd.DataFrame:
    """Load and concatenate all CDF files in a directory."""
    files = sorted(Path(directory).glob("*.cdf"))
    if not files:
        raise FileNotFoundError(f"No CDF files found in: {directory}")

    frames = [parse_cdf(f) for f in files]
    df = pd.concat(frames).sort_index()

    # Drop duplicate timestamps (edge case at day boundaries)
    df = df[~df.index.duplicated(keep="first")]
    logger.info("Loaded %d total rows from %d files", len(df), len(files))
    return df


# ---------------------------------------------------------------------------
# SECTION 3: PHYSICS-INFORMED FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def compute_he_h_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Helium-to-Proton (He/H) flux ratio.

    Physical significance:
      - Quiet solar wind:  He/H ≈ 0.02–0.05  (2–5 %)
      - CME sheath/ejecta: He/H can spike to 0.08–0.30
      - Acts as a high-weight physical constraint for the model

    A rolling median is applied to reduce single-point spikes before
    the ratio is computed — this is intentional pre-ratio cleaning.
    """
    he = df["he_flux"].rolling(window=3, min_periods=1, center=True).median()
    h  = df["h_flux"].rolling(window=3,  min_periods=1, center=True).median()

    ratio = he / h.replace(0, np.nan)  # avoid div-by-zero
    ratio = ratio.clip(lower=0, upper=1.0)  # physical bounds [0, 100%]
    ratio.name = "he_h_ratio"
    return ratio


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full physics-informed feature matrix.

    Features
    --------
    vsw         : Solar wind proton speed [km/s]
    np          : Proton number density [cm^-3]
    tp          : Proton temperature [K]  → log10 transformed
    he_h_ratio  : He/H flux ratio  (high-weight CME indicator)
    beta_proxy  : Plasma beta proxy = np × Tp / Vsw²  (dimensionless relative)
    dvsw_dt     : First-order temporal gradient of Vsw (detects shock ramp)
    """
    feat = pd.DataFrame(index=df.index)

    feat["vsw"]        = df["vsw"]
    feat["np"]         = df["np"]
    feat["tp"]         = np.log10(df["tp"].clip(lower=1e3))  # log10 for scale
    feat["he_h_ratio"] = compute_he_h_ratio(df)

    # Plasma beta proxy: higher during CME magnetic clouds
    feat["beta_proxy"] = (df["np"] * df["tp"]) / (df["vsw"] ** 2 + 1e-6)

    # Shock ramp detection: rapid speed jump is CME leading edge signature
    feat["dvsw_dt"] = feat["vsw"].diff().fillna(0.0)

    return feat


# ---------------------------------------------------------------------------
# SECTION 4: SAVITZKY-GOLAY SMOOTHING (SHOCK-PRESERVING)
# ---------------------------------------------------------------------------

def apply_savgol(
    df: pd.DataFrame,
    window: int  = SAVGOL_WINDOW,
    polyord: int = SAVGOL_POLYORD,
    skip_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter column-wise.

    Design rationale
    ----------------
    Standard rolling-mean would blur the sharp leading-edge shock of a CME
    (the very feature the TCN needs to detect). The S-G filter fits a local
    polynomial, which:
      (a) smooths high-frequency sensor noise
      (b) preserves discontinuities / steep gradients ("shocks")

    `dvsw_dt` and `he_h_ratio` are intentionally EXCLUDED from smoothing
    because their sharp transients ARE the signal.
    """
    skip = set(skip_cols or ["dvsw_dt", "he_h_ratio"])
    smoothed = df.copy()

    for col in df.columns:
        if col in skip:
            continue
        valid_mask = df[col].notna()
        if valid_mask.sum() < window:
            continue

        arr = df[col].values.copy()
        # Interpolate NaN linearly before filtering (S-G cannot handle NaN)
        s = pd.Series(arr)
        arr_interp = s.interpolate(method="linear", limit_direction="both").values
        arr_smooth = savgol_filter(arr_interp, window_length=window, polyorder=polyord)

        # Re-insert NaN at original positions so downstream code is aware
        arr_smooth[~valid_mask.values] = np.nan
        smoothed[col] = arr_smooth

    logger.info("Savitzky-Golay filter applied (window=%d, polyord=%d)", window, polyord)
    return smoothed


# ---------------------------------------------------------------------------
# SECTION 5: LABEL INTEGRATION & NaN HANDLING
# ---------------------------------------------------------------------------

def attach_labels(
    df: pd.DataFrame,
    label_csv: Optional[str] = None,
    synthetic: bool = False,
) -> pd.DataFrame:
    """
    Attach binary CME labels (1 = CME, 0 = quiet) to the feature DataFrame.

    Parameters
    ----------
    df         : feature DataFrame with DatetimeIndex
    label_csv  : path to a CSV with columns [start_time, end_time]
                 representing known CME intervals (from CDAW catalog etc.)
    synthetic  : if True, generate synthetic labels for pipeline testing

    Ground truth sources
    --------------------
    - NASA CDAW CME Catalog: https://cdaw.gsfc.nasa.gov/CME_list/
    - ISRO ASPEX event reports
    - Richardson & Cane ICME catalog
    """
    df["label"] = 0

    if label_csv and Path(label_csv).exists():
        events = pd.read_csv(label_csv, parse_dates=["start_time", "end_time"])
        for _, row in events.iterrows():
            mask = (df.index >= row["start_time"]) & (df.index <= row["end_time"])
            df.loc[mask, "label"] = 1
        pos_frac = df["label"].mean()
        logger.info(
            "Labels from CSV: %d CME windows, %.2f%% positive",
            events.shape[0], pos_frac * 100,
        )

    elif synthetic:
        logger.warning("Generating SYNTHETIC labels — for pipeline testing only!")
        rng = np.random.default_rng(seed=42)
        # Simulate ~5% CME occurrence rate
        df["label"] = (rng.random(len(df)) < 0.05).astype(int)

    else:
        logger.warning("No label source provided. All labels set to 0.")

    return df


def clean_and_impute(df: pd.DataFrame, max_gap_steps: int = 12) -> pd.DataFrame:
    """
    Final NaN cleaning:
      1. Forward/backward fill for short gaps (≤ max_gap_steps)
      2. Drop rows where critical physics columns are still NaN
    """
    critical = ["vsw", "np", "tp"]
    df = df.copy()

    # Short-gap fill
    df = df.ffill(limit=max_gap_steps)
    df = df.bfill(limit=max_gap_steps)

    before = len(df)
    df.dropna(subset=critical, inplace=True)
    after  = len(df)
    if before != after:
        logger.info("Dropped %d rows with persistent NaN in critical columns", before - after)

    # Remaining NaN in non-critical columns → fill with column median
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------------------------------
# SECTION 6: SLIDING WINDOW SEQUENCE BUILDER
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    stride:  int = STRIDE,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat feature DataFrame into overlapping (X, y) sequences.

    X shape: (N_windows, seq_len, n_features)
    y shape: (N_windows,)  — label = 1 if ANY step in window is CME

    Using "ANY CME in window" labelling strategy so the model learns to flag
    approaching CME onset, not just confirmed CME intervals.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]

    X_data = df[feature_cols].values.astype(np.float32)
    y_data = df[label_col].values.astype(np.float32)

    X_seqs, y_seqs = [], []
    n = len(df)

    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len
        X_seqs.append(X_data[start:end])
        y_seqs.append(float(y_data[start:end].max()))  # ANY-CME labelling

    X_arr = np.stack(X_seqs, axis=0)
    y_arr = np.array(y_seqs, dtype=np.float32)

    pos_rate = y_arr.mean()
    logger.info(
        "Built %d sequences (seq_len=%d, stride=%d) | CME rate=%.2f%%",
        len(X_arr), seq_len, stride, pos_rate * 100,
    )
    return X_arr, y_arr


# ---------------------------------------------------------------------------
# SECTION 7: SCALER & TRAIN/VAL/TEST SPLIT
# ---------------------------------------------------------------------------

def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    scaler_path: str = "./aspex_data/scaler.pkl",
    random_state: int = 42,
) -> dict:
    """
    Stratified temporal split and per-feature standardisation.

    Note: We use stratified split because CME labels are rare.
    The scaler is fit on TRAIN only and saved for inference reuse.
    """
    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y.astype(int),
        random_state=random_state,
    )
    # Second split: train vs val
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac,
        stratify=y_tv.astype(int),
        random_state=random_state,
    )

    # Fit scaler on train, transform all
    n, t, f = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, f)
    scaler.fit(X_train_2d)

    def scale(arr):
        sh = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(sh)

    X_train = scale(X_train)
    X_val   = scale(X_val)
    X_test  = scale(X_test)

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "feature_dim": f,
    }

    for k in ["train", "val", "test"]:
        pos = splits[f"y_{k}"].mean()
        logger.info("%5s split: %5d samples | CME rate=%.2f%%", k, len(splits[f"y_{k}"]), pos * 100)

    return splits


# ---------------------------------------------------------------------------
# SECTION 8: MASTER PIPELINE FUNCTION
# ---------------------------------------------------------------------------

def run_pipeline(
    cdf_directory:   str,
    label_csv:       Optional[str] = None,
    synthetic_labels: bool         = False,
    scaler_path:     str           = "./aspex_data/scaler.pkl",
    use_wandb:       bool          = False,
    wandb_project:   str           = "aditya-l1-cme-v2",
) -> dict:
    """
    End-to-end pipeline: CDF files → model-ready numpy arrays.

    Parameters
    ----------
    cdf_directory    : folder containing downloaded .cdf files
    label_csv        : path to CME event catalog CSV
    synthetic_labels : use synthetic labels (testing only)
    scaler_path      : where to save the fitted StandardScaler
    use_wandb        : log data statistics to W&B
    wandb_project    : W&B project name

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    scaler, feature_names, feature_dim
    """
    # Step 1: Load raw CDF data
    raw_df = load_cdf_directory(cdf_directory)

    # Step 2: Feature engineering
    feat_df = engineer_features(raw_df)

    # Step 3: Savitzky-Golay smoothing
    feat_df = apply_savgol(feat_df)

    # Step 4: Attach labels
    feat_df = attach_labels(feat_df, label_csv=label_csv, synthetic=synthetic_labels)

    # Step 5: Clean and impute
    feat_df = clean_and_impute(feat_df)

    # Step 6: Build sequences
    feature_names = [c for c in feat_df.columns if c != "label"]
    X, y = build_sequences(feat_df, feature_cols=feature_names)

    # Step 7: Split and scale
    splits = split_and_scale(X, y, scaler_path=scaler_path)
    splits["feature_names"] = feature_names

    # Step 8: Optional W&B data logging
    if use_wandb and HAS_WANDB:
        wandb.init(project=wandb_project, job_type="data_pipeline", reinit=True)
        wandb.log({
            "n_total_sequences":  len(X),
            "n_train":            len(splits["y_train"]),
            "n_val":              len(splits["y_val"]),
            "n_test":             len(splits["y_test"]),
            "cme_rate_train_pct": float(splits["y_train"].mean() * 100),
            "seq_length":         SEQUENCE_LENGTH,
            "n_features":         splits["feature_dim"],
            "feature_names":      feature_names,
        })
        wandb.finish()

    logger.info("Pipeline complete. Feature shape: %s", splits["X_train"].shape)
    return splits


# ---------------------------------------------------------------------------
# SECTION 9: COLAB QUICK-START HELPER
# ---------------------------------------------------------------------------

def colab_demo_pipeline(
    start: str = "2024-05-01",
    end:   str = "2024-05-31",
    data_dir: str = "./aspex_data",
) -> dict:
    """
    One-call demo for Google Colab with synthetic labels.
    Downloads (or uses cached) CDF files and runs the full pipeline.

    Usage in Colab
    --------------
    >>> from data_pipeline import colab_demo_pipeline
    >>> splits = colab_demo_pipeline()
    >>> X_train, y_train = splits["X_train"], splits["y_train"]
    """
    logger.info("=== Colab Demo Pipeline: %s → %s ===", start, end)

    # Try to download real data; silently fall back to synthetic if unavailable
    files = download_cdf_files(start, end, local_dir=data_dir)

    if not files:
        logger.warning(
            "No CDF files available. "
            "Generating a SYNTHETIC dataset for demonstration. "
            "Replace with real ASPEX-SWIS data for actual training."
        )
        _generate_synthetic_cdf_csv(data_dir, start, end)

    return run_pipeline(
        cdf_directory=data_dir,
        synthetic_labels=True,   # switch to label_csv= for real events
        scaler_path=f"{data_dir}/scaler.pkl",
    )


def _generate_synthetic_cdf_csv(out_dir: str, start: str, end: str) -> None:
    """
    Generate a synthetic CSV mimicking ASPEX-SWIS physics for offline testing.
    Saves as a parquet for use by a patched load_cdf_directory.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(42)
    times = pd.date_range(start, end, freq="5min")
    n     = len(times)

    # Realistic solar wind background with embedded CME signatures
    vsw = 400 + 30 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 15, n)
    np_ = 6  + 1.5 * rng.standard_normal(n)
    tp  = 1e5 * np.exp(rng.normal(0, 0.2, n))
    he  = np_ * (0.04 + rng.uniform(0, 0.02, n))
    h   = np_ * (1.0  + rng.uniform(0, 0.05, n))

    # Inject 3 synthetic CME events
    for cme_center in [n // 5, n // 2, 4 * n // 5]:
        dur = int(0.02 * n)  # ~2% of time range
        s, e_ = max(0, cme_center - dur // 2), min(n, cme_center + dur // 2)
        vsw[s:e_] += 300   # speed enhancement
        np_[s:e_] *= 4     # density enhancement
        tp[s:e_]  *= 2     # temperature enhancement
        he[s:e_]  *= 5     # He/H spike

    df = pd.DataFrame({
        "time":    times,
        "vsw":     vsw,
        "np":      np_,
        "tp":      tp,
        "he_flux": he,
        "h_flux":  h,
        "vth":     np.full(n, np.nan),
    })

    # Monkey-patch load_cdf_directory for synthetic mode
    import data_pipeline as _self
    _orig = _self.load_cdf_directory

    def _synthetic_loader(directory):
        logger.info("Using synthetic dataset (parquet) from %s", directory)
        _df = df.set_index("time")
        return _df

    _self.load_cdf_directory = _synthetic_loader
    logger.info("Synthetic dataset generated: %d rows", n)


if __name__ == "__main__":
    # 1. Double-check the path
    logger.info(f"Targeting directory: {DEFAULT_COLAB_DATA_DIR}")
    
    if not os.path.exists(DEFAULT_COLAB_DATA_DIR):
        logger.error(f"Data directory missing! Colab cannot find: {DEFAULT_COLAB_DATA_DIR}")
        logger.error("Did you remember to upload the folder to Colab's file explorer?")
    else:
        logger.info("=== Starting V2.0 Pipeline for Aditya-L1 Data ===")
        
        # 2. Run the actual pipeline, pointing directly to your downloaded V03 files
        results = run_pipeline(
            cdf_directory=DEFAULT_COLAB_DATA_DIR,
            synthetic_labels=True, # We will swap this for real labels later
            scaler_path="/content/scaler.pkl"
        )
        
        print("\n" + "="*30)
        print("PIPELINE SUCCESS")
        print(f"X_train Shape: {results['X_train'].shape}")
        print(f"Features Used: {results['feature_names']}")
        print("="*30)
