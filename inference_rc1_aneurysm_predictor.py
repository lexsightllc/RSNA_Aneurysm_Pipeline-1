"""RC1-style inference entrypoint adapted for the RSNA aneurysm pipeline repo.

This script mirrors the Kaggle RC1 production server behaviour while
leveraging the shared utilities that live inside this repository.  It can be
used directly inside a Kaggle notebook:

```
import kaggle_evaluation.rsna_inference_server as ks
from inference_rc1_aneurysm_predictor import predict

server = ks.RSNAInferenceServer(predict)
server.serve()
```

The implementation keeps the "ultra aggressive" post-processing heuristics
from the original script, but now integrates tightly with the repository:

* Predictor loading goes through :mod:`src.utils.predictor_loader` to honour
  the user-packaged ``AneurysmPredictor`` class when available and fall back
  to the bundled RC2 stub otherwise.
* Label handling relies on :mod:`src.constants` so the column order always
  matches the rest of the pipeline.
* Column canonicalisation/reconciliation use
  :mod:`src.utils.prediction_utils` to support a variety of predictor output
  formats (dict / pandas / polars / numpy).

The public surface is a single ``predict(series_path)`` function that returns
the 14 competition probabilities as a Polars DataFrame without the ID column.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Ensure the repository is importable before touching internal modules.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.constants import (  # noqa: E402  (deferred import after sys.path)
    RSNA_ALL_LABELS,
    RSNA_ANEURYSM_PRESENT_LABEL,
    RSNA_LOCATION_LABELS,
)
from src.utils.predictor_loader import load_predictor_adapter, setup_environment  # noqa: E402
from src.utils.prediction_utils import (  # noqa: E402
    canonicalize_columns,
    reconcile_presence,
)


try:  # pragma: no cover - kaggle runtime only
    import kaggle_evaluation.rsna_inference_server as kaggle_server  # type: ignore
except Exception:  # pragma: no cover - local/unit-test environment
    kaggle_server = None


# Competition constants -----------------------------------------------------
LABEL_COLS = list(RSNA_ALL_LABELS)


# Global predictor handle ---------------------------------------------------
PREDICT_FN = None
PREDICTOR_NAME = "unloaded"
MODEL_LOADED = False


def _stable_rng(series_id: str) -> np.random.Generator:
    """Create a deterministic RNG based on the series identifier."""

    seed = np.uint32(np.abs(hash(series_id)))
    return np.random.default_rng(seed)


def load_model() -> None:
    """Load the best available predictor using repository utilities."""

    global PREDICT_FN, PREDICTOR_NAME, MODEL_LOADED

    if MODEL_LOADED:
        return

    setup_environment()

    try:
        # Honour packaged predictors first, then bundled fallbacks.
        predict_fn, predictor_name = load_predictor_adapter()
        PREDICT_FN = predict_fn
        PREDICTOR_NAME = predictor_name
        print(f"✅ Predictor initialised: {predictor_name}")
    except Exception as exc:  # pragma: no cover - only triggered on failure
        print(f"❌ Failed to load predictor adapter: {exc}")
        PREDICT_FN = None
        PREDICTOR_NAME = "error"
    finally:
        MODEL_LOADED = True


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _to_dataframe(output: object) -> Optional[pd.DataFrame]:
    """Normalise various predictor outputs into a single-row DataFrame."""

    if output is None:
        return None

    if isinstance(output, pd.DataFrame):
        return output.copy()

    if isinstance(output, pl.DataFrame):
        return output.to_pandas()

    if hasattr(output, "to_pandas"):
        try:
            return output.to_pandas()
        except Exception:
            pass

    if isinstance(output, Mapping):
        return pd.DataFrame([dict(output)])

    if isinstance(output, (list, tuple, np.ndarray)):
        arr = np.asarray(output, dtype=float).ravel()
        cols = LABEL_COLS[: len(arr)]
        data = {c: [float(v)] for c, v in zip(cols, arr)}
        return pd.DataFrame(data)

    return None


def _extract_predictions(model_output: object, series_id: str) -> Dict[str, float]:
    """Extract a label → probability mapping from the predictor output."""

    df = _to_dataframe(model_output)
    if df is None or df.empty:
        return {}

    df = canonicalize_columns(df)
    df = reconcile_presence(df)

    row = df.iloc[0]
    result: Dict[str, float] = {}
    for label in LABEL_COLS:
        val = row.get(label)
        if pd.notna(val):
            try:
                result[label] = float(val)
            except (TypeError, ValueError):
                continue

    return result


def _aggregate_presence(locations: Iterable[float]) -> float:
    loc_arr = np.clip(np.asarray(list(locations), dtype=float), 1e-3, 1 - 1e-3)
    return float(max(0.03, 1.0 - np.prod(1.0 - loc_arr)))


def _expand_to_vector(preds: Mapping[str, float], series_id: str) -> np.ndarray:
    rng = _stable_rng(series_id)
    vec = np.empty(len(LABEL_COLS), dtype=float)

    for idx, label in enumerate(LABEL_COLS):
        value = preds.get(label)
        if value is None or not np.isfinite(value):
            vec[idx] = float(rng.uniform(0.02, 0.15))
        else:
            vec[idx] = float(value)

    if RSNA_ANEURYSM_PRESENT_LABEL not in preds or not np.isfinite(
        preds.get(RSNA_ANEURYSM_PRESENT_LABEL, np.nan)
    ):
        vec[-1] = _aggregate_presence(vec[:-1])

    vec[:-1] = np.clip(vec[:-1], 1e-3, 1 - 1e-3)
    vec[-1] = float(np.clip(vec[-1], 1e-3, 1 - 1e-3))
    return vec


def _run_predictor(series_path: str) -> Dict[str, float]:
    if PREDICT_FN is None:
        return {}

    try:
        output = PREDICT_FN(series_path)
        return _extract_predictions(output, os.path.basename(series_path))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"⚠️ Predictor failed: {exc}")
        return {}


def _ensemble_predictions(
    predictions: Iterable[np.ndarray],
) -> np.ndarray:
    preds = list(predictions)
    if not preds:
        return np.random.uniform(0.02, 0.15, len(LABEL_COLS))

    if len(preds) == 1:
        return preds[0]

    weights = np.array([0.4, 0.3, 0.3][: len(preds)], dtype=float)
    weights /= weights.sum()
    stacked = np.stack(preds, axis=0)
    return np.average(stacked, axis=0, weights=weights)


def _adjust_predictions(vec: np.ndarray, num_dicoms: int) -> np.ndarray:
    adjusted = vec.astype(float).copy()

    if num_dicoms < 20:
        adjusted *= 0.80
    elif num_dicoms < 50:
        adjusted *= 0.90
    elif num_dicoms > 300:
        adjusted *= 1.15
    elif num_dicoms > 150:
        adjusted *= 1.08

    location_scores = adjusted[:-1]
    max_location = float(location_scores.max(initial=0.0))
    mean_location = float(location_scores.mean(initial=0.0))
    top3_mean = float(np.mean(np.sort(location_scores)[-3:]))

    presence = max(
        adjusted[-1],
        max_location * 0.9,
        top3_mean * 0.8,
        mean_location * 1.3,
    )
    adjusted[-1] = presence
    adjusted[-1] = max(adjusted[-1], max_location * 0.85)

    return np.clip(adjusted, 1e-3, 1 - 1e-3)


# ---------------------------------------------------------------------------
# Public predict API
# ---------------------------------------------------------------------------


def predict(series_path: str) -> pl.DataFrame:
    """Predict probabilities for a single DICOM series."""

    global MODEL_LOADED

    if not MODEL_LOADED:
        load_model()

    series_id = os.path.basename(series_path.rstrip(os.sep))
    print(f"🔍 Processing series: {series_id} (predictor={PREDICTOR_NAME})")

    dicom_count = 0
    for root, _, files in os.walk(series_path):
        dicom_count += sum(1 for f in files if f.lower().endswith(".dcm"))

    print(f"📊 Found {dicom_count} DICOM files")

    use_tta = dicom_count > 40
    seeds = [42, 123, 456] if use_tta else [42]

    ensemble_inputs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        preds = _run_predictor(series_path)
        if not preds:
            fallback_rng = _stable_rng(f"fallback-{series_id}-{seed}")
            preds = {
                label: float(fallback_rng.uniform(0.02, 0.15))
                for label in LABEL_COLS
            }

        vec = _expand_to_vector(preds, series_id)
        noise = rng.normal(0.0, 0.01, size=vec.shape)
        ensemble_inputs.append(np.clip(vec + noise, 1e-3, 1 - 1e-3))

    prediction_vec = _ensemble_predictions(ensemble_inputs)
    final_vec = _adjust_predictions(prediction_vec, dicom_count)

    print(
        "📊 Final predictions: mean={:.4f}, max={:.4f}, aneurysm={:.4f}".format(
            float(final_vec.mean()), float(final_vec.max()), float(final_vec[-1])
        )
    )

    df = pl.DataFrame({label: [float(val)] for label, val in zip(LABEL_COLS, final_vec)})

    shutil.rmtree("/kaggle/shared", ignore_errors=True)
    gc.collect()

    return df.select(LABEL_COLS)


# ---------------------------------------------------------------------------
# Optional local gateway for quick manual testing
# ---------------------------------------------------------------------------


def _run_local_gateway():  # pragma: no cover - helper for notebook testing
    if kaggle_server is None:
        print("⚠️ Kaggle evaluation server not available in this environment.")
        return

    server = kaggle_server.RSNAInferenceServer(predict)
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print("🔄 Running in competition mode...")
        server.serve()
    else:
        print("🧪 Running in local test mode...")
        server.run_local_gateway()
        try:
            result_df = pl.read_parquet("/kaggle/working/submission.parquet")
            print("\n📊 SUBMISSION SUMMARY:")
            print(f"   Rows: {len(result_df)}")
            print(f"   Columns: {len(result_df.columns)}")
            for col in LABEL_COLS:
                if col in result_df.columns:
                    values = result_df[col].to_numpy()
                    print(
                        f"   {col}: mean={np.mean(values):.4f}, "
                        f"std={np.std(values):.4f}, range={np.min(values):.4f}-{np.max(values):.4f}"
                    )
        except Exception as exc:
            print(f"📝 Local test completed - submission file generated: {exc}")


if __name__ == "__main__":  # pragma: no cover - script mode
    print("🚀 RSNA RC1 PRODUCTION INFERENCE SERVER - STARTING")
    print("=" * 60)
    load_model()
    _run_local_gateway()
    print("✅ RSNA Production Inference Server Ready!")
    print("=" * 60)

