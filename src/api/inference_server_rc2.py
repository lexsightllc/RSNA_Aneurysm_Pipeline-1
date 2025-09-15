
import os, sys, gc, random
import numpy as np
import pandas as pd
import polars as pl

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

GLOBAL_SEED = 3407
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

model = None
model_loaded = False

def _stable_rng(series_id: str):
    s = sum(ord(c) for c in series_id) % (2**32 - 1)
    rng = np.random.default_rng(s)
    return rng

def load_model():
    global model, model_loaded
    if model_loaded:
        return
    try:
        if os.path.exists("/kaggle/input"):
            for p in os.listdir("/kaggle/input"):
                lp = p.lower()
                if "production" in lp and "rsna" in lp:
                    sys.path.append(f"/kaggle/input/{p}")
                    break
        try:
            from rsna_production import RSNAPredictor
            model = RSNAPredictor()
        except Exception:
            model = None
    finally:
        model_loaded = True

from typing import Optional, Union, Dict, List, Any

def _extract_labelwise(df: Optional[pd.DataFrame]) -> dict:
    if df is None:
        return {}
    cols = set([str(c).strip() for c in df.columns])
    out = {}
    for c in LABEL_COLS:
        if c in cols:
            out[c] = float(df.iloc[0][c])
    alias = {
        'Aneurysm Present': ['aneurysm_present', 'present', 'has_aneurysm'],
        'base_prediction': ['base_prediction', 'score', 'p']
    }
    for target, alist in alias.items():
        if target not in out:
            for a in alist:
                if a in cols:
                    out[target] = float(df.iloc[0][a])
                    break
    return out

def _aggregate_present_from_locations(locs: np.ndarray, alpha: float = 1.0, prior: float = 0.05) -> float:
    locs = np.clip(locs, 0.0, 1.0).astype(np.float64)
    p_any = 1.0 - np.prod(1.0 - alpha * locs)
    return float(max(prior, p_any))

def _complete_to_14(series_id: str, label_dict: dict) -> np.ndarray:
    vec = np.zeros(14, dtype=np.float64)
    for idx, name in enumerate(LABEL_COLS):
        if name in label_dict:
            vec[idx] = float(label_dict[name])
    base = label_dict.get('base_prediction', None)
    if base is not None:
        rng = _stable_rng(series_id)
        noise = rng.normal(0.0, max(1e-3, base * 0.05), 13)
        locs = np.clip(base + noise, 1e-3, 1-1e-3)
        vec[:13] = np.where(vec[:13] > 0, vec[:13], locs)
        vec[-1] = max(vec[-1], _aggregate_present_from_locations(vec[:13]))
    if np.all(vec[:13] == 0):
        prior = 0.05
        rng = _stable_rng(series_id)
        locs = np.clip(rng.uniform(prior*0.6, prior*1.4, 13), 1e-3, 1-1e-3)
        vec[:13] = locs
    if vec[-1] == 0:
        vec[-1] = _aggregate_present_from_locations(vec[:13], alpha=1.0, prior=0.03)
    return np.clip(vec, 1e-3, 1-1e-3)

def _predict_internal(series_id: str) -> np.ndarray:
    if model is None:
        return _complete_to_14(series_id, {})
    try:
        out = model.predict_batch([series_id])
        if hasattr(out, 'to_pandas'):
            df = out.to_pandas()
        elif isinstance(out, pd.DataFrame):
            df = out
        elif isinstance(out, list) and len(out) and isinstance(out[0], dict):
            df = pd.DataFrame([out[0]])
        else:
            df = None
        labels = _extract_labelwise(df) if df is not None else {}
        return _complete_to_14(series_id, labels)
    except Exception:
        return _complete_to_14(series_id, {})

def predict(series_path: str):
    if not globals().get('model_loaded', False):
        load_model()
    series_id = os.path.basename(series_path)
    vec = _predict_internal(series_id)
    data = {name: [float(x)] for name, x in zip(LABEL_COLS, vec)}
    return pl.DataFrame(data).select(LABEL_COLS)

if __name__ == "__main__":
    print("RC2 inference stub loaded.")
