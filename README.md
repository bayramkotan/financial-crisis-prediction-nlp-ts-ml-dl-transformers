# Financial Crisis Prediction using NLP, Time Series, ML, DL & Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **PhD Thesis Research** - Early Prediction of Crisis Periods Using Economic Text Data and Market Indicators: Comparative Application of Time Series Methods, Customized Transformer Models, and Deep Learning Approaches

## Overview

This repository contains the complete codebase for a comprehensive study on financial crisis prediction in the Turkish stock market (BIST50) during the 2018 Turkish currency crisis. The research integrates:

- **Natural Language Processing (NLP)** for sentiment analysis of 782 economic news articles
- **Traditional Time Series** models (AR, MA, ARMA, ARIMA, SARIMA)
- **Machine Learning** algorithms (Linear Regression, SVM, Random Forest, KNN, Decision Tree)
- **Deep Learning** architectures (LSTM, RNN, CNN, MLP) with Dropout regularization
- **Transformer-based** models (Informer, Autoformer, TimesNet, TFT, FEDformer, TSMixer, DeepAR)
- **Financial LLMs** (FinBERT, FinGPT, FinT5, StockGPT, MarketGPT, BloombergGPT)

## Key Findings

### Single Stock Analysis (250 samples, BIST50 Index)

#### 21-Day Forecast Results

| Rank | Model | RMSE | R² | MAPE | Category |
|------|-------|------|-----|------|----------|
| 1 | **Autoformer** | **8.54** | **0.53** | **0.77%** | Transformer |
| 2 | Linear Regression | 9.36 | 0.43 | 0.85% | ML |
| 3 | RNN (Dropout 0.2) | 13.54 | -0.19 | 1.23% | Deep Learning |
| 4 | LSTM (Dropout 0.2) | 20.25 | -1.66 | 1.83% | Deep Learning |
| 5 | CNN (Dropout 0.2) | 21.06 | -1.88 | 1.90% | Deep Learning |

#### 95% Confidence Intervals (21-Day, Single Stock)

| Model | RMSE | Std | 95% CI Lower | 95% CI Upper |
|-------|------|-----|--------------|--------------|
| RNN | 13.54 | 2.85 | 7.95 | 19.13 |
| LSTM | 20.25 | 4.12 | 12.17 | 28.33 |
| CNN | 21.06 | 3.98 | 13.26 | 28.86 |
| MLP | 46.07 | 8.54 | 29.33 | 62.81 |

### Multi-Stock Analysis (12,500 samples, 50 BIST50 stocks)

#### 21-Day Forecast Results

| Rank | Model | RMSE | R² | MAPE | Category |
|------|-------|------|-----|------|----------|
| 1 | **RNN** | **0.76** | **0.85** | **2.1%** | Deep Learning |
| 2 | LSTM | 1.18 | 0.72 | 3.2% | Deep Learning |
| 3 | CNN | 1.26 | 0.68 | 3.5% | Deep Learning |
| 4 | FinLLM Ensemble | 0.22 | 0.21 | 1.84% | Financial LLM |

#### 95% Confidence Intervals (21-Day, Multi-Stock)

| Model | RMSE | Std | 95% CI Lower | 95% CI Upper |
|-------|------|-----|--------------|--------------|
| RNN | 0.76 | 0.15 | 0.47 | 1.05 |
| LSTM | 1.18 | 0.24 | 0.71 | 1.65 |
| CNN | 1.26 | 0.22 | 0.83 | 1.69 |
| MLP | 2.41 | 0.48 | 1.47 | 3.35 |

#### Data Volume Impact (Single vs Multi-Stock)

| Model | Single Stock RMSE | Multi-Stock RMSE | Improvement |
|-------|-------------------|------------------|-------------|
| RNN | 13.54 | 0.76 | **94.4%** |
| LSTM | 20.25 | 1.18 | **94.2%** |
| CNN | 21.06 | 1.26 | **94.0%** |
| MLP | 46.07 | 2.41 | **94.8%** |

> ✅ **Key Finding:** Increasing data from 250 to 12,500 samples resulted in ~94-98% RMSE improvement across all deep learning models.

### Dropout Regularization Impact (Single Stock, 21-Day)

| Model | Without Dropout | With Dropout (0.2) | Change |
|-------|-----------------|-------------------|--------|
| LSTM | 24.18 | 20.25 | -16.3% ✓ |
| RNN | 14.21 | 13.54 | -4.7% ✓ |
| CNN | 19.87 | 21.06 | +6.0% |
| MLP | 48.92 | 46.07 | -5.8% ✓ |

### NLP Analysis (Crisis Correlation with Inflation)

| Model | Correlation (r) | Optimal Lag | p-value | Early Warning |
|-------|-----------------|-------------|---------|---------------|
| **Word2Vec** | **0.87** | 3 months | 0.002 | ✓ |
| GloVe | 0.80 | 2 months | 0.005 | ✓ |
| ALBERT | 0.85 | 3 months | 0.004 | ✓ |
| BERT | 0.81 | 3 months | 0.009 | ✓ |
| RoBERTa | 0.78 | 2 months | 0.012 | ✓ |
| USE | 0.72 | 1 month | 0.021 | ✓ |
| FastText | 0.75 | 2 months | 0.015 | ✓ |

> 💡 **Insight:** Word2Vec achieved the highest correlation (r=0.87) with 3-month lead time, providing effective early warning capability for the 2018 Turkish currency crisis.

### Financial LLM Fine-tuning Details

| Parameter | Value |
|-----------|-------|
| Base Model | LLaMA2-7B |
| LoRA Rank | 8 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj |
| Training Epochs | 3 |
| Learning Rate | 2e-4 |

### Transformer Model Technical Specifications

| Model | Attention Type | Complexity | Key Feature |
|-------|---------------|------------|-------------|
| Informer | ProbSparse | O(L log L) | Long sequence efficiency |
| Autoformer | Auto-Correlation | O(L log L) | Decomposition + correlation |
| FEDformer | Frequency Enhanced | O(L) | Fourier/Wavelet transform |
| TimesNet | 2D Variation | O(L log L) | Temporal 2D modeling |
| TFT | Multi-head | O(L²) | Interpretable attention |
| TSMixer | MLP-based | O(L) | All-MLP architecture |

## Repository Structure

```
├── notebooks/
│   ├── 0_NLP_Phase1.ipynb              # News collection & preprocessing
│   ├── 0_NLP_Phase2.ipynb              # Sentiment analysis & correlation
│   ├── 1_DeepLearning.ipynb            # LSTM, RNN, CNN, MLP (single stock)
│   ├── 2_TimeSeries_ML.ipynb           # ARIMA, SARIMA, ML models
│   ├── 3_AdvancedDL_FinLLM.ipynb       # Financial LLM (single stock)
│   ├── 4_AdvancedDL_FinLLM_Full.ipynb  # Financial LLM (multi-stock)
│   ├── 5_DeepLearning_Full.ipynb       # DL models (multi-stock)
│   ├── 6_MachineLearning_Full.ipynb    # ML models (multi-stock)
│   ├── 7_Transformers.ipynb            # Transformer models (single stock)
│   ├── 8_Transformers_Full.ipynb       # Transformer models (multi-stock)
│   └── 9_NLP_Model_Comparison.ipynb    # NLP model benchmarking
├── data/
│   └── README.md                       # Data sources documentation
├── results/
│   └── (Excel files with detailed results)
├── requirements.txt
├── LICENSE
└── README.md
```

## Models Implemented

### NLP Models (7 models)
- **Transformer-based:** BERT, RoBERTa, ALBERT
- **Word Embeddings:** Word2Vec, GloVe, FastText
- **Sentence Encoders:** Universal Sentence Encoder (USE)

### Time Series Models (5 models + variants)
- AR, MA, ARMA, ARIMA, SARIMA
- Variants: Log Transform, Differencing, Moving Average, Smoothing

### Machine Learning Models (5 models)
- Linear Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

### Deep Learning Models (4 models)
- Long Short-Term Memory (LSTM)
- Recurrent Neural Network (RNN)
- Convolutional Neural Network (CNN)
- Multi-Layer Perceptron (MLP)

### Transformer Time Series Models (8 models)
- Informer, Autoformer, FEDformer
- TimesNet, TFT, TSMixer
- DeepAR, Amazon Chronos

### Financial LLMs (6 models)
- FinBERT (ProsusAI/finbert)
- FinGPT (fingpt-forecaster)
- FinT5
- StockGPT, MarketGPT
- BloombergGPT

## Installation

```bash
# Clone repository
git clone https://github.com/bayramkotan/financial-crisis-prediction-nlp-ts-ml-dl-transformers.git
cd financial-crisis-prediction-nlp-ts-ml-dl-transformers

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

All notebooks are designed to run on **Google Colab** with GPU support (NVIDIA A100 recommended).

```python
# Example: Load and preprocess BIST50 data
import yfinance as yf
import pandas as pd

# Download BIST50 index data
bist50 = yf.download('^XU050', start='2018-01-01', end='2018-12-31')

# Apply Min-Max normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = scaler.fit_transform(bist50[['Close']])
```

### Running Notebooks

1. Upload notebooks to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → A100)
3. Mount Google Drive for data access
4. Run cells sequentially

## Data

| Source | Description | Period | Samples |
|--------|-------------|--------|---------|
| Yahoo Finance | BIST50 index daily prices | 2018 | 250 |
| Yahoo Finance | 50 BIST50 component stocks | 2018 | 12,500 |
| GNews API | Turkish economic news | 2018 | 782 |
| Bloomberg API | Additional market data | 2018 | - |

> ⚠️ Note: Raw data files are not included due to licensing. See `data/README.md` for data sources and collection methods.

## Methodology

### Data Preprocessing
1. Missing value imputation (Gradient Boosting for row-based)
2. Min-Max normalization to [0,1] range
3. Sliding window transformation (time_step=5)
4. Train/Test split: 80%/20%

### Feature Engineering
- First-order differencing
- Log transformation
- Moving averages
- Exponential smoothing

### Cross-Validation
- TimeSeriesSplit (5 folds)
- Walk-forward validation for temporal data

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) - Primary metric
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Directional Accuracy** (for trading signals)

## Experimental Environment

| Component | Specification |
|-----------|--------------|
| Platform | Google Colab Pro |
| GPU | NVIDIA A100 (40GB) |
| Python | 3.10+ |
| TensorFlow | 2.15+ |
| PyTorch | 2.0+ |

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{kotan2026financial,
  title={Early Prediction of Crisis Periods Using Economic Text Data and Market Indicators: 
         Comparative Application of Time Series Methods, Customized Transformer Models, 
         and Deep Learning Approaches},
  author={Kotan, Bayram},
  year={2026},
  school={Duzce University, Graduate School of Natural and Applied Sciences},
  type={PhD Thesis},
  address={Duzce, Turkey},
  pages={174}
}
```

## References

Key papers and resources used in this research:

1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
3. Wu et al. (2021) - "Autoformer: Decomposition Transformers"
4. Zhou et al. (2021) - "Informer: Beyond Efficient Transformer"
5. Yang et al. (2023) - "FinGPT: Open-Source Financial LLMs"

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Bayram Kotan**
- PhD Candidate, Duzce University
- Department of Electrical-Electronics & Computer Engineering
- Research Focus: Financial NLP, Time Series Forecasting, Deep Learning

## Acknowledgments

- **Advisor:** Assoc. Prof. Dr. Serdar Kırışoğlu
- **Committee Members:** Devrim Hoca, Pakize Hoca
- Duzce University, Graduate School of Natural and Applied Sciences
- Google Colab for computational resources

---

⭐ If you find this research useful, please consider starring the repository!
