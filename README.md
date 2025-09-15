# RC2 Inference (Deterministic, Kaggle-ready)

## What RC2 does

- Provides a deterministic `predict(series_path)` for the RSNA IAD challenge.
- Standardizes outputs to the 14 official columns (13 locations + Aneurysm Present) in the exact order.
- Enforces thread caps for stable Kaggle runs (OMP/MKL=1).
- Bridges to your packaged model first; falls back safely if needed.
- Implements efficient column-wise data validation for improved performance.
- Includes robust error handling and logging for better debugging.

## Label order (must match submission)

1. Left Infraclinoid Internal Carotid Artery
2. Right Infraclinoid Internal Carotid Artery
3. Left Supraclinoid Internal Carotid Artery
4. Right Supraclinoid Internal Carotid Artery
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation
14. Aneurysm Present

## Model loading (priority)

RC2 tries to import and use, in order:

1. `AneurysmPredictor` from your package (e.g., `rsna_aneurysm`)
2. `RSNAPredictor` from a production pipeline package/zip

### Accepted APIs (first that exists is used):
- `predict_series(series_path) → dict/DataFrame`
- `predict_batch([SeriesInstanceUID]) → dict/DataFrame`

If only 13 locations are returned, RC2 computes AP as:  
`AP = 1 − ∏(1 − p_location)`

## Offline Inference Runner

For offline inference without Kaggle dependencies, use the `offline_runner`:

```bash
python -m src.api.offline_runner --series-root /path/to/dicom/series --output submission.csv
```

Key features:
- Works fully offline with no network dependencies
- Supports custom model integration
- Validates outputs against the RSNA schema
- Efficient CPU usage with minimal memory footprint

For detailed usage and customization options, see the [offline runner documentation](docs/offline_runner.md).

## Using RC2 in a Kaggle Notebook

1. Put `inference_rc2_aneurysm_predictor.py` in your working directory (or attach as a Dataset and add to `sys.path`).
2. Start the evaluation server:

```python
import kaggle_evaluation.rsna_inference_server as ks
from inference_rc2_aneurysm_predictor import predict  # exposes predict(series_path)

server = ks.RSNAInferenceServer(predict)
server.serve()  # Kaggle evaluation will call predict() per series
```

RC2 returns a Polars DataFrame with exactly 14 columns (no ID); Kaggle merges on SeriesInstanceUID.

## Minimal offline runner (optional)

```python
import os, polars as pl
from inference_rc2_aneurysm_predictor import predict

series_root = "/kaggle/input/rsna-aneurysm-detection/series"
rows = []
for sid in ["<SeriesUID_1>", "<SeriesUID_2>"]:  # replace with real UIDs
    df = predict(os.path.join(series_root, sid))
    row = {c: float(df[c][0]) for c in df.columns}
    row["SeriesInstanceUID"] = sid
    rows.append(row)

sub = pl.DataFrame(rows)[["SeriesInstanceUID",  # ID first, then labels in order
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
]]
sub.write_csv("submission.csv")
```

## Determinism checklist

- [x] RC2 sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
- [x] No random TTA or DICOM-count heuristics.
- [x] Two identical runs on the same input should produce byte-identical predictions.
- [x] All random operations use fixed seeds for reproducibility.
- [x] File paths are resolved consistently across different environments.

## Output format requirements

- **Return** (from `predict`): Polars DataFrame with the 14 columns above, in order, without the ID column.
- **Submission file**: CSV named `submission.csv` with `SeriesInstanceUID` followed by the 14 columns above.

## Recent Improvements

### Data Validation
- Implemented efficient column-wise validation for prediction data
- Added comprehensive schema validation for all input/output data
- Improved error messages with detailed information about validation failures

### Performance Optimizations
- Replaced row-by-row validation with vectorized operations
- Optimized data loading and validation for large datasets
- Reduced memory usage during inference

### Bug Fixes
- Fixed path resolution for configuration files
- Corrected execution order in the inference pipeline
- Improved error handling and logging

## Troubleshooting

- **Model imports fail**: ensure your package/zip is accessible (copied to `/kaggle/working` or attached as a Dataset) and on `sys.path`.
- **Column names differ**: RC2 maps common aliases (e.g., `aneurysm_present`, `left mca`, `vertebral artery`) to the canonical names, but it's best if your model emits the exact RSNA names.
- **AP missing**: RC2 derives it from locations, as shown above.
# RSNA_Aneurysm_Pipeline-1
