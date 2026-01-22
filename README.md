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
| 3 | Informer | 12.12 | 0.15 | 1.11% | Transformer |
| 4 | DeepAR | 12.83 | -0.07 | 1.18% | Transformer |
| 5 | SARIMA | 13.35 | - | 1.21% | Time Series |
| 6 | RNN (Dropout 0.2) | 13.54 | -0.19 | 1.23% | Deep Learning |
| 7 | LSTM (Dropout 0.2) | 20.25 | -1.67 | 1.83% | Deep Learning |
| 8 | CNN (Dropout 0.2) | 21.06 | -1.88 | 1.90% | Deep Learning |
| 9 | MLP (Dropout 0.2) | 46.07 | -12.80 | 4.18% | Deep Learning |
| 10 | TFT | 48.44 | -14.25 | 5.30% | Transformer |

#### 95% Confidence Intervals (21-Day, Single Stock)

| Category | Model | RMSE | Std | 95% CI |
|----------|-------|------|-----|--------|
| Deep Learning | RNN | 13.54 | 4.06 | [5.58, 21.50] |
| Deep Learning | LSTM | 20.25 | 6.08 | [8.34, 32.16] |
| Deep Learning | CNN | 21.06 | 6.32 | [8.68, 33.44] |
| Deep Learning | MLP | 46.07 | 13.82 | [18.98, 73.16] |
| Transformer | Autoformer | 8.54 | 2.99 | [2.68, 14.40] |
| Transformer | Informer | 12.12 | 4.24 | [3.81, 20.43] |
| Transformer | DeepAR | 12.83 | 4.49 | [4.03, 21.63] |
| Transformer | TimesNet | 19.49 | 6.82 | [6.12, 32.86] |
| Transformer | TSMixer | 19.38 | 6.78 | [6.09, 32.67] |
| Transformer | TFT | 48.44 | 16.95 | [15.21, 81.67] |
| Time Series | Linear Reg. | 9.36 | 2.34 | [4.77, 13.95] |
| Time Series | SARIMA | 13.35 | 3.34 | [6.81, 19.89] |

### Multi-Stock Analysis (12,500 samples, 50 BIST50 stocks)

#### 21-Day Forecast Results

| Rank | Model | RMSE | R² | Category |
|------|-------|------|-----|----------|
| 1 | **CNN** | **0.75** | -0.15 | Deep Learning |
| 2 | Linear Regression | 0.77 | - | ML |
| 3 | RNN | 0.80 | -0.16 | Deep Learning |
| 4 | LSTM | 0.81 | -0.16 | Deep Learning |
| 5 | MLP | 0.85 | -0.17 | Deep Learning |

> ⚠️ **Note:** Negative R² values in multi-stock analysis indicate that models struggle to explain stock-specific variance, though RMSE values are low due to scale differences (index ~1000 points vs individual stock prices ~10-100 TL).

#### 95% Confidence Intervals (21-Day, Multi-Stock)

| Model | RMSE | Std | 95% CI |
|-------|------|-----|--------|
| CNN | 0.75 | 0.15 | [0.46, 1.04] |
| RNN | 0.80 | 0.16 | [0.49, 1.11] |
| LSTM | 0.81 | 0.16 | [0.49, 1.13] |
| MLP | 0.85 | 0.17 | [0.52, 1.18] |
| Linear Reg. | 0.77 | 0.12 | [0.54, 1.00] |

#### Data Volume Impact (Single vs Multi-Stock)

| Model | Single Stock RMSE | Multi-Stock RMSE | Improvement |
|-------|-------------------|------------------|-------------|
| RNN | 13.54 | 0.80 | **94.1%** |
| LSTM | 20.25 | 0.81 | **96.0%** |
| CNN | 21.06 | 0.75 | **96.4%** |
| MLP | 46.07 | 0.85 | **98.2%** |

> ⚠️ **Important:** This RMSE reduction is due to two factors: (1) Scale difference - single stock analysis predicts BIST50 index (~1000 points) while multi-stock predicts individual prices (~10-100 TL); (2) Data volume increase - from 250 to 12,500 samples improves pattern learning capacity.

### Dropout Regularization Impact (Single Stock, 21-Day)

| Model | Without Dropout | With Dropout (0.2) | Change |
|-------|-----------------|-------------------|--------|
| LSTM | 21.53 | 20.25 | -5.9% ✓ |
| RNN | 13.40 | 13.54 | +1.0% |
| CNN | 15.99 | 21.06 | +31.7% |
| MLP | 15.76 | 46.07 | +192.3% |

> 💡 **Insight:** Dropout (0.2) improved LSTM performance but was too aggressive for CNN and MLP with limited data (250 samples). Dropout rate should be tuned based on model architecture and data size.

### NLP Analysis (Crisis Correlation with Inflation)

| Model | Lag (Month) | Correlation (r) | p-value | Interpretation |
|-------|-------------|-----------------|---------|----------------|
| **Word2Vec** | 3 | **0.8715** | 0.0022 | Strongest leading indicator |
| ALBERT | 3 | 0.8058 | 0.0087 | Medium correlation |
| GloVe | 2 | 0.8025 | 0.0052 | Significant correlation |
| FastText | 1 | 0.6800 | 0.0834 | Statistically insignificant |

> 💡 **Insight:** Word2Vec achieved the highest correlation (r=0.87) with 3-month lead time, providing effective early warning capability for the 2018 Turkish currency crisis.

### NLP Semantic Similarity Scores

| Model | Text Type | Economic Crisis | Market Stability | Financial Growth |
|-------|-----------|-----------------|------------------|------------------|
| ALBERT | lemmatized | 0.62 | 0.66 | **0.67** |
| ALBERT | english | 0.60 | 0.63 | 0.63 |
| FastText | english | 0.61 | 0.53 | 0.62 |
| BERT | lemmatized | 0.42 | 0.28 | 0.43 |
| RoBERTa | stemmed | 0.45 | 0.41 | 0.46 |
| GloVe | english | 0.49 | 0.51 | 0.53 |
| Word2Vec | lemmatized | 0.31 | 0.33 | 0.33 |
| USE | lemmatized | 0.10 | 0.07 | 0.08 |

> 💡 **Note:** ALBERT produced highest similarity scores (0.61-0.67), while Word2Vec showed highest inflation correlation (r=0.87). These metrics measure different characteristics: similarity score measures semantic closeness, correlation measures temporal economic dynamics.

### Transformer Model Performance (Single Stock, 21-Day)

| Model | RMSE | Relative RMSE | MAE | MAPE(%) | R² | Directional Accuracy (%) |
|-------|------|---------------|-----|---------|-----|--------------------------|
| Autoformer | 8.54 | 0.0303 | 6.89 | 0.77 | 0.53 | 60.00 |
| Informer | 12.12 | 0.0212 | 9.81 | 1.11 | 0.15 | 43.33 |
| DeepAR | 12.83 | 0.0390 | 10.48 | 1.18 | -0.07 | 53.33 |
| TSMixer | 19.38 | 0.0167 | 16.48 | 1.87 | -1.44 | 63.33 |
| TimesNet | 19.49 | 0.0387 | 16.00 | 1.81 | -1.47 | 33.33 |
| TFT | 48.44 | 0.0189 | 47.01 | 5.30 | -14.25 | 36.67 |

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

| Model | Source | Attention Type | Complexity | Key Feature |
|-------|--------|---------------|------------|-------------|
| Informer | Zhou et al. (2021) | ProbSparse | O(L log L) | Long sequence efficiency |
| Autoformer | Wu et al. (2021) | Auto-Correlation | O(L log L) | Decomposition + correlation |
| FEDformer | Zhou et al. (2022) | Frequency Enhanced | O(L) | Fourier/Wavelet transform |
| TimesNet | Wu et al. (2023) | 2D Variation | O(L log L) | Temporal 2D modeling |
| TFT | Lim et al. (2021) | Multi-horizon | O(L²) | Interpretable attention |
| TSMixer | Chen et al. (2023) | MLP Mixing | O(L) | All-MLP architecture |
| DeepAR | Salinas et al. (2020) | Autoregressive RNN | O(L) | Probabilistic forecasting |

### Best Models by Forecast Period (RMSE)

| Category | Best Model | 1 Day | 10 Day | 21 Day |
|----------|------------|-------|--------|--------|
| Time Series (Single) | AR Differencing | 0.84 | 6.00 | 13.35 |
| ML (Single) | Linear Regression | 0.64 | 7.69 | 9.36 |
| Deep Learning (Single) | RNN | 4.19 | 13.54 | 13.54 |
| Transformer (Single) | Informer | 12.12 | 12.12 | 12.12 |
| ML (Multi) | Linear Regression | 0.33 | 0.56 | 0.77 |
| Deep Learning (Multi) | LSTM | 0.81 | 0.81 | 0.81 |
| Transformer (Multi) | TSMixer | 0.29 | 0.29 | 0.29 |

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
3. Sliding window transformation (time_step=30)
4. Train/Test split: Training (Jan 1 - Nov 30, 2018), Test (Dec 1-31, 2018)

### Feature Engineering
- First-order differencing
- Log transformation
- Moving averages (3-day)
- Exponential smoothing

### Cross-Validation
- TimeSeriesSplit (10 folds)
- Walk-forward validation for temporal data
- Temporal leakage prevention

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
| NeuralForecast | For Transformer time series models |
| Hugging Face | For NLP models |
| scikit-learn | For ML models |

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
- **Committee Members:** Prof. Dr. Devrim Akgün, Prof. Dr. Pakize Erdoğmuş
- Duzce University, Graduate School of Natural and Applied Sciences
- Google Colab for computational resources

---

⭐ If you find this research useful, please consider starring the repository!
