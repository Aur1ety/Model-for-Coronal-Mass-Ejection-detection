"""
predict_streamlit.py
====================
Lightweight Inference Module + Streamlit Dashboard
Aditya-L1 ASPEX-SWIS CME Detection Pipeline v2.0

This module is intentionally DECOUPLED from training code.
It imports only what it needs for real-time inference, making it
trivially importable in:
  - Streamlit apps
  - Hugging Face Spaces (Gradio/Streamlit)
  - FastAPI endpoints
  - Jupyter notebooks

Run the Streamlit app:
    $ streamlit run predict_streamlit.py

Or import the inference function alone:
    >>> from predict_streamlit import CMEInferenceEngine
    >>> engine = CMEInferenceEngine("./checkpoints/best_model.pt", "./aspex_data/scaler.pkl")
    >>> prob = engine.predict_latest(df_raw)

Author : CME Detection Pipeline v2.0
Python : 3.10+ | Streamlit 1.x | PyTorch 2.x
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib

# Streamlit — optional at import time for non-UI uses
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import sibling modules
from data_pipeline import (
    engineer_features, apply_savgol, clean_and_impute,
    SEQUENCE_LENGTH, CDF_VARMAP,
)
from model_factory import load_model, CMEDetectorTCN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("predict_streamlit")

# ---------------------------------------------------------------------------
# SECTION 1: INFERENCE ENGINE (framework-agnostic)
# ---------------------------------------------------------------------------

class CMEInferenceEngine:
    """
    Self-contained inference engine for CME detection.

    Encapsulates:
      - Model loading & device management
      - Scaler loading
      - Data preprocessing (same pipeline as training)
      - Sliding-window prediction over arbitrary-length input
      - Alert thresholding

    Parameters
    ----------
    checkpoint_path : path to best_model.pt
    scaler_path     : path to scaler.pkl (fitted during training)
    threshold       : probability threshold for CME alert (default 0.5)
    device          : 'cpu', 'cuda', or None (auto-detect)
    """

    def __init__(
        self,
        checkpoint_path: str,
        scaler_path:     str,
        threshold:       float = 0.5,
        device:          Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.device    = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load scaler
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        self.scaler: object = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)

        # Load model
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.model: CMEDetectorTCN = load_model(checkpoint_path, self.device)
        self.model.eval()

        # Infer feature count from scaler
        self.n_features: int = self.scaler.mean_.shape[0]
        logger.info(
            "CMEInferenceEngine ready | device=%s | features=%d | threshold=%.2f",
            self.device, self.n_features, threshold,
        )

    def preprocess(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Apply the same preprocessing pipeline used during training.

        raw_df must contain raw ASPEX-SWIS columns:
          vsw, np, tp, he_flux, h_flux  (same as CDF_VARMAP keys)

        Returns numpy array of shape (n_steps, n_features), scaled.
        """
        feat_df = engineer_features(raw_df)
        feat_df = apply_savgol(feat_df)
        feat_df = clean_and_impute(feat_df)

        feature_cols = [c for c in feat_df.columns]
        X = feat_df[feature_cols].values.astype(np.float32)

        # Scale using the TRAINING scaler (no re-fitting!)
        X_scaled = self.scaler.transform(X)
        return X_scaled, feat_df.index

    def predict_sequence(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Run sliding-window inference over a scaled feature array.

        Returns
        -------
        probs : np.ndarray of shape (n_windows,)
                probability of CME for each window
        """
        n = len(X_scaled)
        if n < SEQUENCE_LENGTH:
            # Pad with zeros if insufficient history (cold-start case)
            pad = np.zeros((SEQUENCE_LENGTH - n, self.n_features), dtype=np.float32)
            X_scaled = np.vstack([pad, X_scaled])

        windows = []
        for start in range(0, len(X_scaled) - SEQUENCE_LENGTH + 1, 1):
            windows.append(X_scaled[start: start + SEQUENCE_LENGTH])

        X_tensor = torch.tensor(np.stack(windows), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy()

        return probs

    def predict_latest(self, raw_df: pd.DataFrame) -> dict:
        """
        Predict CME probability for the LATEST data window.

        This is the primary entry-point for real-time dashboards.

        Parameters
        ----------
        raw_df : DataFrame with raw ASPEX-SWIS columns

        Returns
        -------
        dict:
          - probability : float [0, 1] — CME probability for latest window
          - alert       : bool — True if probability >= threshold
          - alert_level : str — "NONE" | "WATCH" | "WARNING" | "ALERT"
          - timestamp   : pd.Timestamp of latest data point
        """
        X_scaled, timestamps = self.preprocess(raw_df)
        probs = self.predict_sequence(X_scaled)
        latest_prob = float(probs[-1])

        return {
            "probability": latest_prob,
            "alert":       latest_prob >= self.threshold,
            "alert_level": self._alert_level(latest_prob),
            "timestamp":   timestamps[-1],
        }

    def predict_timeseries(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a full probability time-series for the entire input DataFrame.
        Useful for retrospective analysis and dashboard plots.

        Returns DataFrame with columns: [probability, alert, alert_level]
        aligned to the input timestamps.
        """
        X_scaled, timestamps = self.preprocess(raw_df)
        probs = self.predict_sequence(X_scaled)

        # Align probabilities to timestamps
        # Each prob[i] corresponds to the LAST time-step of window[i]
        aligned_idx = timestamps[SEQUENCE_LENGTH - 1:]
        # If lengths mismatch due to padding, truncate
        min_len = min(len(probs), len(aligned_idx))
        probs   = probs[:min_len]
        idx     = aligned_idx[:min_len]

        result_df = pd.DataFrame({
            "probability": probs,
            "alert":       probs >= self.threshold,
            "alert_level": [self._alert_level(p) for p in probs],
        }, index=idx)

        return result_df

    @staticmethod
    def _alert_level(prob: float) -> str:
        """Map probability to NOAA-inspired alert level."""
        if prob < 0.30:   return "NONE"
        if prob < 0.50:   return "WATCH"
        if prob < 0.70:   return "WARNING"
        return                   "ALERT"


# ---------------------------------------------------------------------------
# SECTION 2: DEMO DATA GENERATOR (for dashboard with no live feed)
# ---------------------------------------------------------------------------

def generate_demo_data(
    n_hours: int = 72,
    inject_cme: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic ASPEX-SWIS data for dashboard demo.
    Injects a CME signature at t = 2/3 of the time range.
    """
    rng   = np.random.default_rng(seed)
    freq  = "5min"
    end   = pd.Timestamp.now().floor("5min")
    start = end - pd.Timedelta(hours=n_hours)
    times = pd.date_range(start, end, freq=freq)
    n     = len(times)

    vsw = 420 + 20 * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 12, n)
    np_ = 5.5 + rng.normal(0, 0.8, n)
    tp  = 1.1e5 * np.exp(rng.normal(0, 0.15, n))
    he  = np_ * (0.04 + rng.uniform(0, 0.01, n))
    h   = np_ * (1.0  + rng.uniform(0, 0.03, n))

    if inject_cme:
        cme_start = int(0.65 * n)
        cme_end   = int(0.75 * n)
        vsw[cme_start:cme_end] += 350
        np_[cme_start:cme_end] *= 5
        tp[cme_start:cme_end]  *= 2.5
        he[cme_start:cme_end]  *= 6

    return pd.DataFrame({
        "vsw": vsw, "np": np_, "tp": tp,
        "he_flux": he, "h_flux": h,
        "vth": np.full(n, np.nan),
    }, index=times)


# ---------------------------------------------------------------------------
# SECTION 3: STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    """
    Full Streamlit dashboard for CME detection.

    Features:
      - Model & data configuration in sidebar
      - Real-time gauge for latest CME probability
      - 72-hour time-series plot with colour-coded alert zones
      - Physics parameter sub-plots (Vsw, np, Tp, He/H)
      - Downloadable prediction CSV
    """
    if not HAS_STREAMLIT:
        raise ImportError("Streamlit not installed: pip install streamlit")

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Aditya-L1 CME Detector",
        page_icon="☀️",
        layout="wide",
    )

    st.title("☀️ Aditya-L1 ASPEX-SWIS | CME Detection Dashboard v2.0")
    st.caption("Temporal Convolutional Network — Physics-Informed Solar Wind Analysis")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        checkpoint = st.text_input(
            "Model Checkpoint (.pt)",
            value="./checkpoints/best_model.pt",
        )
        scaler_path = st.text_input(
            "Scaler Path (.pkl)",
            value="./aspex_data/scaler.pkl",
        )
        threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)
        demo_mode = st.checkbox("Use Synthetic Demo Data", value=True)
        inject_cme = st.checkbox("Inject CME into Demo", value=True)
        n_hours    = st.slider("Data Window (hours)", 24, 168, 72, 24)

        st.divider()
        st.markdown("**Model Info**")
        st.markdown("""
- Architecture: TCN with dilated residual blocks
- Input: Vsw, np, Tp, He/H ratio, β proxy, ΔVsw/Δt
- Sequence length: 128 steps (~10.6 h)
- Trained on: ASPEX-SWIS L2-BLK CDF
        """)

    # ── Load engine (cached) ─────────────────────────────────────────────────
    @st.cache_resource
    def load_engine(ckpt, scl, thr):
        try:
            return CMEInferenceEngine(ckpt, scl, threshold=thr)
        except FileNotFoundError as e:
            return None, str(e)

    if not demo_mode:
        engine = load_engine(checkpoint, scaler_path, threshold)
        if engine is None:
            st.error("Could not load model. Check paths in sidebar.")
            st.stop()
    else:
        engine = None  # will use demo predictions

    # ── Load data ─────────────────────────────────────────────────────────────
    if demo_mode:
        raw_df = generate_demo_data(n_hours=n_hours, inject_cme=inject_cme)
        st.info("🔬 Running in synthetic demo mode. Toggle off in sidebar to use real data.")
    else:
        st.warning("Live CDF ingestion: connect your ASPEX-SWIS data feed here.")
        raw_df = generate_demo_data(n_hours=n_hours, inject_cme=False)

    # ── Compute predictions (demo or real) ───────────────────────────────────
    if engine is not None:
        pred_df  = engine.predict_timeseries(raw_df)
        latest   = engine.predict_latest(raw_df)
    else:
        # Demo: use synthetic probability curve for visualisation
        n = len(raw_df)
        rng  = np.random.default_rng(99)
        prob = 0.05 + 0.03 * np.abs(np.sin(np.linspace(0, 10 * np.pi, n))) + rng.uniform(0, 0.05, n)
        if inject_cme:
            s, e = int(0.65 * n), int(0.75 * n)
            t = np.linspace(0, np.pi, e - s)
            prob[s:e] = 0.3 + 0.65 * np.sin(t)
        prob = np.clip(prob, 0, 1)
        idx  = raw_df.index
        pred_df = pd.DataFrame({
            "probability": prob,
            "alert": prob >= threshold,
            "alert_level": [CMEInferenceEngine._alert_level(p) for p in prob],
        }, index=idx)
        latest = {
            "probability": float(prob[-1]),
            "alert": float(prob[-1]) >= threshold,
            "alert_level": CMEInferenceEngine._alert_level(float(prob[-1])),
            "timestamp": idx[-1],
        }

    # ── TOP ROW: KPIs ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    prob_pct = latest["probability"] * 100
    level    = latest["alert_level"]
    colour   = {"NONE": "🟢", "WATCH": "🟡", "WARNING": "🟠", "ALERT": "🔴"}[level]

    with col1:
        st.metric("CME Probability", f"{prob_pct:.1f}%",
                  delta=f"{prob_pct - 5:.1f}% vs quiet baseline")
    with col2:
        st.metric("Alert Level", f"{colour} {level}")
    with col3:
        st.metric("Last Update", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))
    with col4:
        cme_periods = pred_df["alert"].sum()
        st.metric("CME Windows (72h)", int(cme_periods))

    st.divider()

    # ── MAIN PLOT: Probability time-series ────────────────────────────────────
    if HAS_PLOTLY:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.35, 0.22, 0.22, 0.21],
            subplot_titles=[
                "CME Detection Probability", "Solar Wind Speed (Vsw) [km/s]",
                "Proton Density (np) [cm⁻³]", "He/H Flux Ratio",
            ],
            vertical_spacing=0.06,
        )

        # --- Row 1: Probability ---
        x = pred_df.index

        # Alert zones shading
        for threshold_line, colour_zone, label in [
            (0.30, "rgba(255,255,0,0.12)", "Watch"),
            (0.50, "rgba(255,165,0,0.15)", "Warning"),
            (0.70, "rgba(255,50,50,0.18)",  "Alert"),
        ]:
            fig.add_hrect(
                y0=threshold_line, y1=1.0,
                fillcolor=colour_zone, line_width=0,
                annotation_text=label, annotation_position="top right",
                row=1, col=1,
            )

        fig.add_trace(go.Scatter(
            x=x, y=pred_df["probability"],
            mode="lines", name="CME Probability",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ), row=1, col=1)

        fig.add_hline(y=threshold, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold ({threshold})", row=1, col=1)

        # --- Row 2: Vsw ---
        fig.add_trace(go.Scatter(
            x=raw_df.index, y=raw_df["vsw"],
            mode="lines", name="Vsw",
            line=dict(color="#f0a500", width=1.2),
        ), row=2, col=1)

        # --- Row 3: np ---
        fig.add_trace(go.Scatter(
            x=raw_df.index, y=raw_df["np"],
            mode="lines", name="np",
            line=dict(color="#6be5b4", width=1.2),
        ), row=3, col=1)

        # --- Row 4: He/H ---
        he_h = raw_df["he_flux"] / (raw_df["h_flux"] + 1e-9)
        fig.add_trace(go.Scatter(
            x=raw_df.index, y=he_h,
            mode="lines", name="He/H ratio",
            line=dict(color="#d070ff", width=1.2),
        ), row=4, col=1)
        fig.add_hline(y=0.08, line_dash="dot", line_color="rgba(208,112,255,0.6)",
                      annotation_text="CME threshold", row=4, col=1)

        fig.update_layout(
            height=750,
            template="plotly_dark",
            paper_bgcolor="rgba(10,10,20,0.95)",
            plot_bgcolor="rgba(15,15,30,0.95)",
            showlegend=False,
            margin=dict(l=60, r=20, t=40, b=40),
            font=dict(color="#e0e0e0"),
        )
        fig.update_yaxes(range=[0, 1], row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.line_chart(pred_df["probability"])

    # ── DATA TABLE & DOWNLOAD ────────────────────────────────────────────────
    with st.expander("📊 Raw Prediction Data"):
        st.dataframe(pred_df.tail(50).style.background_gradient(
            subset=["probability"], cmap="RdYlGn_r"
        ))

    csv_data = pred_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_data,
        file_name="cme_predictions.csv",
        mime="text/csv",
    )

    # ── ABOUT ────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this pipeline"):
        st.markdown("""
### Aditya-L1 ASPEX-SWIS CME Detection Pipeline v2.0

**Data Source**: ISRO ASPEX-SWIS Level-2 BLK CDF files  
**Training Period**: May 2024 (Solar Maximum peak) + early 2026

**Physics Features**:
| Feature | Description | CME Signature |
|---------|-------------|---------------|
| Vsw | Solar wind speed | +200–500 km/s jump at shock |
| np | Proton density | 3–10× enhancement in sheath |
| Tp | Proton temperature (log) | Enhancement at shock, depletion in cloud |
| He/H ratio | Helium-to-proton flux ratio | 0.08–0.30 in CME ejecta (vs ~0.04 quiet) |
| β proxy | np × Tp / Vsw² | High in cloud, low in sheath |
| ΔVsw/Δt | Speed gradient | Sharp positive spike at leading edge |

**Model**: TCN (Temporal Convolutional Network)
- Dilated causal convolutions (no future leakage)
- Residual blocks with skip connections
- Effective receptive field: ~85 hours
- Loss: Weighted BCE (pos_weight ≈ 19 for 5% CME rate)

**Signal Processing**: Savitzky-Golay filter (window=11, poly=3)  
→ Smooths sensor noise while **preserving shock transients**
        """)


# ---------------------------------------------------------------------------
# SECTION 4: FASTAPI ENDPOINT STUB (for Hugging Face Spaces / production)
# ---------------------------------------------------------------------------

"""
To deploy as a REST API on Hugging Face Spaces (FastAPI):

    from fastapi import FastAPI
    from pydantic import BaseModel
    from predict_streamlit import CMEInferenceEngine
    import pandas as pd

    app    = FastAPI(title="CME Detector API")
    engine = CMEInferenceEngine("./checkpoints/best_model.pt", "./aspex_data/scaler.pkl")

    class SolarWindInput(BaseModel):
        timestamps: list[str]
        vsw:        list[float]
        np:         list[float]
        tp:         list[float]
        he_flux:    list[float]
        h_flux:     list[float]

    @app.post("/predict")
    def predict(data: SolarWindInput):
        df = pd.DataFrame({...}, index=pd.to_datetime(data.timestamps))
        result = engine.predict_latest(df)
        return result

    # Run: uvicorn predict_streamlit:app --reload
"""


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly: launch the Streamlit dashboard
    # $ python predict_streamlit.py   →  triggers 'streamlit run' automatically
    import subprocess, sys
    if HAS_STREAMLIT:
        run_dashboard()
    else:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        print("Then: streamlit run predict_streamlit.py")
