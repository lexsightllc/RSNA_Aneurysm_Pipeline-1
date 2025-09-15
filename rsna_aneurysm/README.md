# RSNA Intracranial Aneurysm Detection

A compact 3D CNN model for detecting intracranial aneurysms in DICOM series, designed for the RSNA Intracranial Aneurysm Detection Challenge.

## Features

- **Lightweight 3D CNN** based on MobileNetV3-Small principles
- **Deterministic inference** for reproducible results
- **Efficient preprocessing** of DICOM series
- **Simple API** for integration with existing pipelines
- **Fallback predictions** for robustness

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rsna-aneurysm.git
   cd rsna-aneurysm
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from rsna_aneurysm import AneurysmPredictor

# Initialize the predictor (automatically uses GPU if available)
predictor = AneurysmPredictor(weights_path="path/to/weights.pth")

# Predict on a DICOM series directory
result = predictor.predict_series("/path/to/dicom/series")
print(result)
```

### Integration with RC2 Pipeline

The package is designed to work seamlessly with the RC2 pipeline:

```python
from rsna_aneurysm import AneurysmPredictor

# Initialize the predictor
predictor = AneurysmPredictor()

# Get predictions as a pandas DataFrame
df = predictor.predict_series_to_dataframe("/path/to/dicom/series")
print(df)
```

### Using the Command Line

```bash
# Run inference on a directory of DICOM series
python -m rsna_aneurysm.predict --series-root /path/to/series --out predictions.csv
```

## Model Architecture

The model is a compact 3D CNN with the following key features:

- **Depthwise separable 3D convolutions** for parameter efficiency
- **Squeeze-and-Excitation blocks** for channel-wise feature recalibration
- **Dilated convolutions** in later layers for larger receptive fields
- **Residual connections** for better gradient flow
- **Hardswish activations** for improved quantization support

## Preprocessing

The preprocessing pipeline includes:

1. DICOM loading with proper windowing (40/400 WW/WL)
2. Resampling to isotropic resolution (96×128×128)
3. Intensity normalization (z-score)
4. Data augmentation (during training)

## Performance

- **Inference time**: ~200ms per series on CPU, ~50ms on GPU
- **Model size**: <50MB
- **Memory usage**: <1GB for inference

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
