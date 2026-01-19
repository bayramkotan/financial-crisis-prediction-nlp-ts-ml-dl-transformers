# Financial Crisis Prediction using NLP, Time Series, ML, DL & Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **PhD Thesis Research** - Comparative Analysis of Time Series Methods, Customized Transformer Models, and Deep Learning Approaches for Early Prediction of Crisis Periods Using Economic Text Data and Market Indicators

## Overview

This repository contains the complete codebase for a comprehensive study on financial crisis prediction in the Turkish stock market (BIST50). The research integrates:

- **Natural Language Processing (NLP)** for sentiment analysis of economic news
- **Traditional Time Series** models (ARIMA, SARIMA)
- **Machine Learning** algorithms (SVM, Random Forest, KNN)
- **Deep Learning** architectures (LSTM, RNN, CNN, MLP)
- **Transformer-based** models (Informer, Autoformer, TimesNet, TFT, FEDformer)
- **Financial LLMs** (FinBERT, FinGPT, BloombergGPT ensemble)

## Key Findings

### Single Stock Analysis (250 samples, 21-day forecast)

| Metric | Best Model | Value | Category |
|--------|-----------|-------|----------|
| **R²** | Autoformer | **0.53** | Transformer |
| **RMSE** | Autoformer | **8.54** | Transformer |
| **MAPE** | Autoformer | **0.77%** | Transformer |

**Top 5 Models by R²:**
| Rank | Model | R² | RMSE | MAPE | Category |
|------|-------|-----|------|------|----------|
| 1 | Autoformer | 0.53 | 8.54 | 0.77% | Transformer |
| 2 | Linear Regression | 0.43 | 9.36 | 0.85% | Time Series/ML |
| 3 | RNN | 0.38 | 9.77 | 0.83% | Deep Learning |
| 4 | DeepAR | -0.07 | 12.83 | 1.18% | Transformer |
| 5 | Ensemble (6 LLM) | -21.13 | 58.35 | 5.28% | Financial LLM |

> ⚠️ **Note:** Financial LLM ensemble (FinBERT, FinGPT, FinT5, StockGPT, MarketGPT, BloombergGPT) performed poorly on single stock due to limited training data (250 samples).

### Multi-Stock Analysis (12,500 samples, 21-day forecast)

| Metric | Best Model | Value | Category |
|--------|-----------|-------|----------|
| **MAPE** | Ensemble (FinGPT+Tech+Sent) | **1.84%** | Financial LLM |
| **R²** | Ensemble (FinGPT+Tech+Sent) | **0.21** | Financial LLM |
| **RMSE** | Ensemble (FinGPT+Tech+Sent) | **0.22** | Financial LLM |

**Top 5 Models by MAPE:**
| Rank | Model | MAPE | RMSE | R² | Category |
|------|-------|------|------|-----|----------|
| 1 | **Ensemble (FinGPT+Tech+Sent)** | **1.84%** | 0.22 | **0.21** | Financial LLM |
| 2 | FinGPT (Multi-Stock) | 3.04% | 0.33 | -0.72 | Financial LLM |
| 3 | FEDformer | 3.83% | 0.29 | -1.61 | Transformer |
| 4 | TSMixer | 3.86% | 0.30 | -1.62 | Transformer |
| 5 | Informer | 4.35% | 0.35 | -2.35 | Transformer |

> ✅ **Key Finding:** Financial LLM ensemble achieved the **only positive R² (0.21)** in multi-stock analysis, demonstrating the value of combining sentiment features with technical indicators when sufficient data is available.

### NLP Analysis (Crisis Early Warning)

| Model | Correlation (r) | Lag | p-value | Significant |
|-------|-----------------|-----|---------|-------------|
| Word2Vec | **0.87** | 3 months | 0.002 | ✓ |
| GloVe | 0.80 | 2 months | 0.005 | ✓ |
| ALBERT | 0.81 | 3 months | 0.009 | ✓ |

> 💡 **Insight:** Classical word embeddings (Word2Vec, GloVe) outperformed transformer models (BERT, RoBERTa) for crisis prediction correlation, suggesting simpler models may better capture economic sentiment patterns.

## Repository Structure

```
├── notebooks/
│   ├── 0_NLP_Phase1.ipynb          # News data collection & preprocessing
│   ├── 0_NLP_Phase2.ipynb          # Sentiment analysis & correlation
│   ├── 1_DeepLearning.ipynb        # LSTM, RNN, CNN, MLP (single stock)
│   ├── 2_TimeSeries_ML.ipynb       # ARIMA, SARIMA, traditional ML
│   ├── 3_AdvancedDL_FinLLM.ipynb   # Financial LLM ensemble
│   ├── 4_AdvancedDL_FinLLM_Full.ipynb  # FinLLM (multi-stock)
│   ├── 5_DeepLearning_Full.ipynb   # DL models (multi-stock)
│   ├── 6_MachineLearning_Full.ipynb    # ML models (multi-stock)
│   ├── 7_Transformers.ipynb        # Transformer models (single stock)
│   ├── 8_Transformers_Full.ipynb   # Transformer models (multi-stock)
│   └── 9_NLP_Model_Comparison.ipynb    # NLP model benchmarking
├── data/
│   └── README.md                   # Data sources documentation
├── results/
│   └── (Excel files with detailed results)
├── requirements.txt
└── README.md
```

## Models Implemented

### NLP Models
- BERT, RoBERTa, ALBERT
- Word2Vec, GloVe, FastText
- Universal Sentence Encoder (USE)

### Time Series Models
- AR, MA, ARMA, ARIMA, SARIMA
- Fourier Transform

### Machine Learning Models
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Linear Regression

### Deep Learning Models
- Long Short-Term Memory (LSTM)
- Recurrent Neural Network (RNN)
- Convolutional Neural Network (CNN)
- Multi-Layer Perceptron (MLP)

### Transformer Models
- Informer
- Autoformer
- FEDformer
- TimesNet
- Temporal Fusion Transformer (TFT)
- TSMixer
- DeepAR

### Financial LLMs
- FinBERT, FinGPT, FinT5
- StockGPT, MarketGPT
- BloombergGPT

## Installation

```bash
# Clone repository
git clone https://github.com/bayramkotan/financial-crisis-prediction-nlp-ts-ml-dl-transformers.git
cd financial-crisis-prediction-nlp-ts-ml-dl-transformers

# Install dependencies
pip install -r requirements.txt
```

## Usage

All notebooks are designed to run on **Google Colab** with GPU support (A100 recommended).

1. Upload notebooks to Google Colab
2. Mount Google Drive for data access
3. Run cells sequentially

## Data

- **Market Data**: BIST50 index and component stocks (2018)
- **News Data**: 782 Turkish economic news articles via GNews API
- **Period**: January 2018 - December 2018 (Turkish currency crisis period)

> Note: Raw data files are not included due to licensing. See `data/README.md` for data sources.

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{kotan2026financial,
  title={Early Prediction of Crisis Periods Using Economic Text Data and Market Indicators: 
         Comparative Application of Time Series Methods, Customized Transformer Models, 
         and Deep Learning Approaches},
  author={Kotan, Bayram},
  year={2026},
  school={Duzce University},
  type={PhD Thesis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Bayram Kotan**
- PhD Candidate, Duzce University
- Research Focus: Financial NLP, Time Series Forecasting, Deep Learning

## Acknowledgments

- Advisor: Assoc. Prof. Dr. Serdar Kirisoglu
- Duzce University, Graduate School of Natural and Applied Sciences
