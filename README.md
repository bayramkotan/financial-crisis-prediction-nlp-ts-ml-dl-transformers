<p align="center">
  <img src="https://img.shields.io/badge/🇹🇷_BIST50-Financial_Crisis_Prediction-red?style=for-the-badge&labelColor=black" alt="BIST50"/>
</p>

<h1 align="center">
  🔮 Financial Crisis Prediction
  <br/>
  <sub>NLP • Time Series • Machine Learning • Deep Learning • Transformer TS • Financial LLM</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/🤗_Transformers-Latest-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-success?style=flat-square"/>
</p>

<p align="center">
  <b>📚 PhD Thesis Research</b><br/>
  <i>Early Prediction of Crisis Periods Using Economic Text Data and Market Indicators</i>
</p>

<p align="center">
  <a href="#-key-results">Results</a> •
  <a href="#-models">Models</a> •
  <a href="#%EF%B8%8F-installation">Installation</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🎯 Overview

> **Research Question:** Can we predict financial crises by combining news sentiment analysis with advanced ML/DL models?

This repository presents a **multi-layered early warning system** for the **2018 Turkish Currency Crisis**, analyzing the BIST50 stock market using **35+ models** across 6 categories.

<div align="center">
<table>
<tr>
<td width="50%">

### 📊 Dataset Summary

| Data | Details |
|:-----|:--------|
| 📰 **News Articles** | 782 Turkish economic news |
| 📈 **Single Stock** | BIST50 Index, 250 days |
| 📊 **Multi-Stock** | 50 stocks × 250 days = 12,500 samples |
| 📅 **Period** | January - December 2018 |

</td>
<td width="50%">

### 🏆 Best Results

| Scenario | Best Model | RMSE |
|:---------|:-----------|-----:|
| 📈 Single Stock (21d) | **Autoformer** | **8.54** |
| 📊 Multi-Stock (21d) | **FEDformer** | **0.29** |
| 📰 NLP Correlation | **Word2Vec** | **r=0.87** |

</td>
</tr>
</table>
</div>

---

## 📈 Key Results

### 🏅 Model Performance Leaderboard (21-Day Forecast, Single Stock)

<div align="center">

| Rank | Model | RMSE | R² | MAPE | Category |
|:----:|:------|-----:|:--:|-----:|:---------|
| 🥇 | **Autoformer** | **8.54** | **+0.53** | **0.77%** | Transformer TS |
| 🥈 | Linear Regression | 9.36 | +0.43 | 0.85% | Machine Learning |
| 🥉 | Informer | 12.12 | +0.05 | 1.11% | Transformer TS |
| 4 | DeepAR | 12.83 | -0.07 | 1.18% | Transformer TS |
| 5 | FinLLM Ensemble | 13.19 | -0.13 | 1.32% | Financial LLM |
| 6 | AR Fark Alma | 13.35 | -0.16 | 1.10% | Time Series |
| 7 | RNN | 13.54 | -0.19 | 1.24% | Deep Learning |

</div>

### 📊 Performance by Category

<div align="center">

**Single Stock (21-Day) - RMSE by Category:**

| Category | Best Model | RMSE |
|:---------|:-----------|-----:|
| 🏆 Transformer TS | Autoformer | 8.54 |
| Machine Learning | Linear Regression | 9.36 |
| Financial LLM | Ensemble | 13.19 |
| Time Series | AR Fark Alma | 13.35 |
| Deep Learning | RNN | 13.54 |

**Multi-Stock (21-Day) - RMSE by Category:**

| Category | Best Model | RMSE |
|:---------|:-----------|-----:|
| 🏆 Transformer TS | FEDformer | 0.29 |
| Financial LLM | Ensemble | 0.32 |
| Deep Learning | CNN | 0.75 |
| Machine Learning | Linear Reg | 0.77 |

</div>

### 🔍 NLP Early Warning System

Our **Word2Vec-based sentiment analysis** achieved remarkable predictive power:

<div align="center">

| Model | Lag | Correlation | p-value | Status |
|:------|:---:|:-----------:|:-------:|:------:|
| **Word2Vec** | 3 months | **r = 0.8715** | 0.0022 | 🏆 Best |
| Word2Vec | 2 months | r = 0.8621 | 0.0013 | ✅ |
| ALBERT | 3 months | r = 0.8058 | 0.0087 | ✅ |
| GloVe | 2 months | r = 0.8025 | 0.0052 | ✅ |
| USE | 3 months | r = 0.7443 | 0.0214 | ✅ |
| FastText | 0 months | r = 0.6800 | 0.0150 | ✅ |

</div>

> 💡 **Key Finding:** Word2Vec predicted inflation changes **3 months in advance** with 87% correlation!

---

## 🧠 Models

<div align="center">
<table>
<tr>
<td width="33%" valign="top">

### 📰 NLP Models (7)
```
├── 🤖 BERT
├── 🤖 ALBERT
├── 🤖 RoBERTa
├── 📝 Word2Vec
├── 📝 GloVe
├── 📝 FastText
└── 🔤 USE
```

</td>
<td width="33%" valign="top">

### 📈 Time Series (5+)
```
├── AR
├── MA
├── ARMA
├── ARIMA
├── SARIMA
└── + variants
    ├── Log Transform
    ├── Differencing
    └── Smoothing
```

</td>
<td width="33%" valign="top">

### 🤖 Machine Learning (5)
```
├── Linear Regression
├── SVM (RBF Kernel)
├── Random Forest
├── Decision Tree
└── KNN
```

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 🧬 Deep Learning (4)
```
├── LSTM
├── RNN
├── CNN
└── MLP
```

</td>
<td width="33%" valign="top">

### ⚡ Transformer TS (7)
```
├── Autoformer ⭐
├── Informer
├── FEDformer
├── TimesNet
├── TFT
├── TSMixer
└── DeepAR
```

</td>
<td width="33%" valign="top">

### 💰 Financial LLMs (3)
```
├── FinBERT
├── FinGPT
└── FinT5

Fine-tuned with LoRA:
├── rank: 8
├── alpha: 32
└── epochs: 3
```

</td>
</tr>
</table>
</div>

---

## 📉 Detailed Results

### 🎯 Transformer TS Models (21-Day Forecast, Single Stock)

<div align="center">

| Model | RMSE | MAE | R² | MAPE | Dir. Acc. | Key Feature |
|:------|-----:|----:|:--:|-----:|:---------:|:------------|
| 🥇 **Autoformer** | **8.54** | 6.89 | **+0.53** | 0.77% | 60.0% | Auto-Correlation |
| 🥈 Informer | 12.12 | 9.81 | +0.05 | 1.11% | 43.3% | ProbSparse Attention |
| 🥉 DeepAR | 12.83 | 10.48 | -0.07 | 1.18% | 53.3% | Probabilistic |
| TimesNet | 19.49 | 16.00 | -1.47 | 1.81% | 33.3% | 2D Variation |
| TSMixer | 19.38 | 16.48 | -1.44 | 1.87% | **63.3%** | All-MLP |
| TFT | 48.44 | 47.01 | -14.25 | 5.30% | 36.7% | Multi-horizon |

</div>

### 🧬 Deep Learning Models (21-Day Forecast)

<div align="center">
<table>
<tr>
<th>Model</th>
<th>Single Stock</th>
<th>Multi-Stock</th>
<th>Improvement</th>
</tr>
<tr>
<td><b>RNN</b></td>
<td>13.54 RMSE</td>
<td>0.80 RMSE</td>
<td>
<img src="https://img.shields.io/badge/-94.1%25-success?style=flat-square"/>
</td>
</tr>
<tr>
<td><b>LSTM</b></td>
<td>20.25 RMSE</td>
<td>0.81 RMSE</td>
<td>
<img src="https://img.shields.io/badge/-96.0%25-success?style=flat-square"/>
</td>
</tr>
<tr>
<td><b>CNN</b></td>
<td>21.06 RMSE</td>
<td>0.75 RMSE</td>
<td>
<img src="https://img.shields.io/badge/-96.4%25-success?style=flat-square"/>
</td>
</tr>
<tr>
<td><b>MLP</b></td>
<td>46.07 RMSE</td>
<td>0.85 RMSE</td>
<td>
<img src="https://img.shields.io/badge/-98.2%25-success?style=flat-square"/>
</td>
</tr>
</table>
</div>

> ⚠️ **Note:** RMSE reduction is due to: (1) Scale difference (index ~1000 pts vs stock prices ~10-100 TL), (2) Data volume increase (250 → 12,500 samples)

### 💰 Financial LLM Ensemble Results

<div align="center">

**Sentiment Scores:**

| Model | Score | Interpretation |
|:------|------:|:---------------|
| FinBERT | -0.087 | 📉 Bearish |
| FinGPT | +0.045 | ➡️ Neutral |
| FinT5 | -0.391 | 📉📉 Very Bearish |
| **Ensemble** | **-0.117** | **📉 Bearish** |

**Forecast Performance vs Naive Baseline:**

| Period | Ensemble RMSE | Naive RMSE | Improvement |
|:------:|:-------------:|:----------:|:-----------:|
| 1-Day | 2.42 | 4.54 | ✅ **47%** |
| 10-Day | 15.72 | 32.87 | ✅ **52%** |
| 21-Day | 13.19 | 36.74 | ✅ **64%** |

</div>

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph INPUT["📥 DATA SOURCES"]
        A["📰 NEWS<br/>782 articles"]
        B["📈 PRICE<br/>BIST50 Index"]
        C["📊 MACRO<br/>Inflation"]
    end
    
    subgraph PROCESS["⚙️ PROCESSING LAYERS"]
        D["🔤 NLP LAYER<br/>Word2Vec • ALBERT • GloVe"]
        E["📉 FORECAST LAYER<br/>Autoformer • LSTM/RNN • Linear Reg"]
        F["🤖 ENSEMBLE LAYER<br/>FinBERT • FinGPT • FinT5"]
    end
    
    subgraph OUTPUT["📤 OUTPUT"]
        G["🚨 EARLY WARNING SIGNAL<br/>Sentiment + Forecast + Technical Analysis"]
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    E --> G
    F --> G
    
    style INPUT fill:#e1f5fe
    style PROCESS fill:#fff3e0
    style OUTPUT fill:#ffebee
    style G fill:#ef5350,color:#fff
```

---

## 📁 Repository Structure

```
📦 financial-crisis-prediction
├── 📂 notebooks/
│   ├── 📓 0_NLP_Phase1.ipynb           # News collection & preprocessing
│   ├── 📓 0_NLP_Phase2.ipynb           # Sentiment analysis
│   ├── 📓 1_DeepLearning.ipynb         # LSTM, RNN, CNN, MLP
│   ├── 📓 2_TimeSeries_ML.ipynb        # ARIMA, SARIMA, ML
│   ├── 📓 3_AdvancedDL_FinLLM.ipynb    # FinLLM (single stock)
│   ├── 📓 4_AdvancedDL_FinLLM_Full.ipynb # FinLLM (multi-stock)
│   ├── 📓 5_DeepLearning_Full.ipynb    # DL (multi-stock)
│   ├── 📓 6_MachineLearning_Full.ipynb # ML (multi-stock)
│   ├── 📓 7_Transformers.ipynb         # Transformer TS (single)
│   ├── 📓 8_Transformers_Full.ipynb    # Transformer TS (multi)
│   └── 📓 9_NLP_Comparison.ipynb       # NLP benchmarking
├── 📂 data/
│   └── 📄 README.md                    # Data sources
├── 📂 results/
│   └── 📊 Model_Comparison.xlsx        # Detailed results
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/bayramkotan/financial-crisis-prediction-nlp-ts-ml-dl-transformers.git
cd financial-crisis-prediction-nlp-ts-ml-dl-transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Requirements

```
tensorflow>=2.15
torch>=2.0
transformers>=4.30
neuralforecast>=1.6
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
yfinance>=0.2
gensim>=4.3
```

---

## 🖥️ Environment

<div align="center">

| Component | Specification |
|:----------|:-------------|
| **Platform** | Google Colab Pro |
| **GPU** | NVIDIA A100 (40GB) |
| **Python** | 3.10+ |
| **Framework** | TensorFlow 2.15 / PyTorch 2.0 |

</div>

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@phdthesis{kotan2026financial,
  title     = {Early Prediction of Crisis Periods Using Economic Text Data 
               and Market Indicators: Comparative Application of Time Series 
               Methods, Customized Transformer Models, and Deep Learning Approaches},
  author    = {Kotan, Bayram},
  year      = {2026},
  school    = {Duzce University},
  type      = {PhD Thesis},
  pages     = {174}
}
```

---

## 📚 Key References

<div align="center">

| Paper | Authors | Year | Contribution |
|:------|:--------|:----:|:-------------|
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer architecture |
| Autoformer | Wu et al. | 2021 | Auto-correlation mechanism |
| Informer | Zhou et al. | 2021 | ProbSparse attention |
| FinBERT | Yang et al. | 2020 | Financial sentiment |
| FinGPT | Yang et al. | 2023 | Open-source financial LLM |

</div>

---

## 👤 Author

<div align="center">
<table>
<tr>
<td width="150">
<img src="https://img.shields.io/badge/PhD-Candidate-blue?style=for-the-badge"/>
</td>
<td>

**Bayram Kotan**  
📍 Duzce University, Turkey  
🎓 Department of Electrical-Electronics & Computer Engineering  
🔬 Research: Financial NLP, Time Series, Deep Learning

</td>
</tr>
</table>
</div>

### 🙏 Acknowledgments

- **Advisor:** Assoc. Prof. Dr. Serdar Kırışoğlu
- **Committee:** Prof. Dr. Resul Kara, Prof. Dr. Pakize Erdoğmuş, Prof. Dr. Devrim Akgün, Assoc. Prof. Dr. Murat İskefiyeli
- **Resources:** Google Colab, Hugging Face, NeuralForecast

---

<p align="center">
  <img src="https://img.shields.io/badge/Made_with-❤️-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/in-Turkey_🇹🇷-red?style=for-the-badge"/>
</p>

<p align="center">
  ⭐ <b>If you find this research useful, please star the repository!</b> ⭐
</p>

---

<p align="center">
  <sub>© 2026 Bayram Kotan. MIT License.</sub>
</p>
