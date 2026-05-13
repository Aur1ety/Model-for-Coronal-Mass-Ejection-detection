# Aditya-L1 ASPEX-SWIS CME Detection Pipeline v2.0

Autonomous Coronal Mass Ejection detection using in-situ solar wind 
plasma measurements from ISRO's Aditya-L1 spacecraft.

## Results
| Metric | Value |
|--------|-------|
| Validation F1 | 0.3101 (TCN) / 0.3182 (ensemble) |
| Blind test peak P(CME) | 0.8707 |
| Blind test verdict | CME DETECTED — May 9, 2026 03:10 UTC |
| Physics corroboration | He/H = 0.2965, Vsw = 598 km/s |

## Architecture
- **TCN** (Temporal Convolutional Network) — shock specialist
- **TCAN** (TCN + Self-Attention) — long-range precursor detector  
- **Ensemble** — 70% TCN + 30% TCAN

## Features (8 physics-informed)
1. Solar wind speed Vsw [km/s]
2. Proton density np [cm⁻³]
3. Proton temperature log₁₀(Tp) [K]
4. He/H flux ratio (CME ejecta indicator)
5. Plasma beta proxy
6. ΔVsw/Δt (shock ramp detector)
7. dV/dt (smoothed speed gradient)
8. dn/dt (smoothed density gradient)

## Data
- Source: ISRO ASPEX-SWIS Level-2 BLK CDF files
- Training period: May 2024 (G5 storm), 2025 quiet baseline, 
  Jan-Feb 2026 active noise
- Blind test: May 7-10, 2026

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
from predict_streamlit import CMEInferenceEngine
engine = CMEInferenceEngine("saved_models/tcn_optimised_v2.pth", 
                             "scalers/scaler_8feat.pkl")
result = engine.predict_latest(raw_df)
print(result)
```

## Future Work
Adding Bz magnetometer data from DSCOVR/Wind is expected to push 
F1 above 0.70 by providing the magnetic flux rope signature that 
plasma data alone cannot detect.
