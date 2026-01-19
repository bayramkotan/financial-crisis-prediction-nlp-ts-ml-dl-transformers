# Data Files

This directory contains all datasets used in the thesis research.

## Directory Structure

```
data/
├── raw/                    # Original unprocessed data
│   ├── BIST_50_2018_Data.csv
│   └── BorsaHaberleri_Neutr_2018-01-01_2018-12-31.csv
├── processed/              # Cleaned and processed data
│   ├── BIST_50_2018_Data_Full_Imputed.csv
│   ├── BorsaHaberleri_Neutr_2018-01-01_2018-12-31-ingilizce.csv
│   ├── BorsaHaberleri_Neutr_2018-01-01_2018-12-31-ingilizce-KokleriBulunmus.csv
│   └── BorsaHaberleri_Neutr_2018-01-01_2018-12-31-Degerlendirme_Sonuclari.xlsx
└── README.md
```

## File Descriptions

### Raw Data

| File | Description | Rows | Period |
|------|-------------|------|--------|
| `BIST_50_2018_Data.csv` | BIST50 stock prices (raw) | 250 | Jan-Dec 2018 |
| `BorsaHaberleri_Neutr_2018-01-01_2018-12-31.csv` | Turkish economic news articles | 782 | Jan-Dec 2018 |

### Processed Data

| File | Description | Processing |
|------|-------------|------------|
| `BIST_50_2018_Data_Full_Imputed.csv` | BIST50 with missing values imputed | Interpolation + Gradient Boosting |
| `BorsaHaberleri_...-ingilizce.csv` | News translated to English | Google Translate API |
| `BorsaHaberleri_...-ingilizce-KokleriBulunmus.csv` | Stemmed/Lemmatized English text | NLTK |
| `BorsaHaberleri_...-Degerlendirme_Sonuclari.xlsx` | NLP evaluation results | Sentiment scores |

## Data Details

### BIST50 Stock Data

- **Source**: Yahoo Finance
- **Stocks**: 50 component stocks of BIST50 index
- **Fields**: Date, Ticker_Open, Ticker_High, Ticker_Low, Ticker_Now (Close), Ticker_Volume
- **Period**: January 1, 2018 - December 31, 2018 (Turkish currency crisis year)
- **Trading Days**: 250

### News Data

- **Source**: GNews API
- **Language**: Turkish (original), English (translated)
- **Articles**: 782 unique news articles
- **Keywords searched**: 
  - borsa, hisse senedi, piyasa, yatirim, ekonomi
  - ticaret, finans, para piyasasi, sermaye piyasasi
  - doviz, endeks

### Preprocessing Steps

1. **Stock Data Imputation**:
   - Column-wise: Linear interpolation
   - Row-wise: Gradient Boosting Regression
   - Result: Zero missing values

2. **Text Processing**:
   - Translation: Turkish to English (Google Translate)
   - Tokenization: NLTK word_tokenize
   - Stemming: Porter Stemmer
   - Lemmatization: WordNet Lemmatizer

## Usage Example

```python
import pandas as pd

# Load stock data
stocks = pd.read_csv('data/processed/BIST_50_2018_Data_Full_Imputed.csv')

# Load news data
news = pd.read_csv('data/processed/BorsaHaberleri_Neutr_2018-01-01_2018-12-31-ingilizce.csv')

print(f"Stock data shape: {stocks.shape}")
print(f"News articles: {len(news)}")
```

## License

Data is provided for academic research purposes only.
