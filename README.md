# Fashion Demand Forecasting: Predicting Clothing Sales with Machine Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Pytest-yellow.svg)](https://docs.pytest.org/)

An end-to-end, portfolio-grade Machine Learning application and interactive Streamlit analytics platform designed to help fashion and retail merchandisers forecast future clothing unit demand, compare machine learning algorithms against statistical baselines, evaluate forecast uncertainty, and optimize inventory replenishment schedules.

---

## Table of Contents

1. [Project Overview and Problem Statement](#project-overview-and-problem-statement)
2. [Key Capabilities and Features](#key-capabilities-and-features)
3. [Architecture and Workflow](#architecture-and-workflow)
4. [Technology Stack](#technology-stack)
5. [Directory Structure](#directory-structure)
6. [Dataset Acquisition and Data Integrity](#dataset-acquisition-and-data-integrity)
7. [Data Preprocessing and Fashion Taxonomy](#data-preprocessing-and-fashion-taxonomy)
8. [Leakage-Free Feature Engineering](#leakage-free-feature-engineering)
9. [Forecasting Models](#forecasting-models)
10. [Time-Series Validation and Metrics](#time-series-validation-and-metrics)
11. [Inventory Planning Insights](#inventory-planning-insights)
12. [Installation and Environment Setup](#installation-and-environment-setup)
13. [Data Preparation and Model Training](#data-preparation-and-model-training)
14. [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
15. [Automated Testing](#automated-testing)
16. [Exporting Forecasts](#exporting-forecasts)
17. [Known Limitations and Future Roadmap](#known-limitations-and-future-roadmap)
18. [GitHub Repository Setup](#github-repository-setup)
19. [Attribution and License](#attribution-and-license)

---

## Project Overview and Problem Statement

Fashion retail is characterized by short product lifecycles, seasonal trend swings, and long supplier lead times. Retail merchandisers face a persistent trade-off:

- **Overestimating demand** leads to bloated inventory holding costs, tied-up working capital, warehouse congestion, and margin-eroding discount markdowns.
- **Underestimating demand** results in painful stockouts, lost sales revenue, and alienated customers.

This project delivers a robust, reproducible time-series machine learning solution that aggregates retail sales transactions into regular weekly demand, engineers leakage-free predictive features, trains machine learning models alongside statistical benchmarks, and serves an interactive web dashboard for real-world merchandising and inventory replenishment decisions.

---

## Key Capabilities and Features

- **Real Dataset Ingestion with Automated Fallback:** Ingests the public UCI Online Retail transaction dataset and includes an offline deterministic benchmark generator for network-resilient local development.
- **Transparent Fashion Taxonomy Classification:** Normalizes unstandardized product descriptions into 6 apparel and accessory categories (`Tops`, `Dresses`, `Outerwear`, `Trousers`, `Accessories`, `Footwear`) and isolates an `Other` general merchandise class with automated coverage statistics.
- **Strict Leakage Prevention:** All time-series features (shifted lags, rolling means, rolling standard deviations) strictly exclude the current target period $t$, eliminating data leakage.
- **Multi-Model Benchmark Suite:**
  - *Naive Baseline:* Last observed demand persistence.
  - *4-Week Moving Average Baseline:* Short-term rolling trend baseline.
  - *Random Forest Regressor:* Non-linear decision tree ensemble utilizing lag structure and cyclical calendar features.
- **Multi-Step Recursive Forecasting:** Generates multi-week forward predictions (1 to 12 weeks) with demand non-negativity constraints.
- **Empirical Uncertainty Bands:** Residual-based 90% prediction intervals that dynamically widen as the forecast horizon extends.
- **Interactive Streamlit Web Dashboard:**
  - Dynamic KPI cards (Units Sold, Avg Weekly Demand, Forecast Total, Holdout MAE).
  - Category selector, horizon slider, model selector, and date range filters.
  - Interactive Plotly charts with forecast origin demarcation.
  - Interactive replenishment order calculator integrating lead times and safety stock buffers.
  - One-click CSV forecast export.

---

## Architecture and Workflow

```mermaid
flowchart TD
    A["Raw Transaction Data (CSV / Excel)"] --> B["Data Loader & Quality Verification (src/data_loader.py)"]
    B --> C["Preprocessing & Fashion Classification (src/preprocessing.py)"]
    C --> D["Weekly Aggregation & Regularization (src/preprocessing.py)"]
    D --> E["Feature Engineering: Lags & Calendar (src/feature_engineering.py)"]
    E --> F["Chronological Train/Holdout Split (src/evaluation.py)"]
    F --> G["Model Training & Benchmark Comparison (src/forecasting.py)"]
    G --> H["Model Serialization (models/demand_forecaster.joblib)"]
    H --> I["Streamlit Web Dashboard (app.py)"]
    I --> J["Inventory Insights & CSV Forecast Export"]
```

---

## Technology Stack

- **Core Language:** Python 3.11+
- **Data Manipulation and Analysis:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Model Serialization:** joblib
- **Visualization:** Plotly, Matplotlib
- **Web Application:** Streamlit
- **Testing and Quality Assurance:** pytest
- **Version Control:** Git

---

## Directory Structure

```text
fashion-demand-forecasting/
├── .env.example                # Sample environment configurations
├── .gitignore                  # Git exclusions for environments, cache & raw data
├── app.py                      # Interactive Streamlit analytics application
├── LICENSE                     # MIT License
├── README.md                   # Full documentation and project guide
├── requirements.txt            # Python dependencies with compatible versions
│
├── data/
│   ├── processed/              # Cleaned sales & regularized weekly demand series
│   │   └── .gitkeep
│   ├── raw/                    # Raw transaction datasets (excluded from git)
│   │   └── .gitkeep
│   └── sample/                 # Deterministic fallback sample benchmark
│       ├── .gitkeep
│       └── fashion_retail_sample.csv
│
├── models/                     # Serialized model artifacts (.joblib)
│   └── .gitkeep
│
├── notebooks/                  # Interactive Jupyter exploratory notebook
│   └── exploratory_analysis.ipynb
│
├── reports/                    # Diagnostic summaries, evaluation tables & exports
│   ├── .gitkeep
│   ├── data_summary.json
│   ├── model_comparison.json
│   └── sample_forecast.csv
│
├── scripts/
│   ├── download_data.py        # Dataset acquisition script
│   ├── prepare_data.py         # Data cleaning & weekly demand aggregation
│   └── train_model.py          # Model training, holdout evaluation & saving
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, downloads & quality checks
│   ├── evaluation.py           # MAE, RMSE, WAPE, MAPE & chronological splitting
│   ├── feature_engineering.py  # Shifted lags, rolling windows & calendar features
│   ├── forecasting.py          # Forecaster classes (Baselines & Random Forest)
│   ├── preprocessing.py        # Cleaning, deduplication, returns & category mapping
│   ├── utils.py                # Paths, logging & deterministic seed setup
│   └── visualization.py        # Interactive Plotly chart builders
│
└── tests/
    ├── test_evaluation.py      # Tests for metrics & zero-actual edge cases
    ├── test_features.py        # Tests for leakage prevention & recursive features
    ├── test_forecasting.py     # Tests for baseline and ML forecasters
    └── test_preprocessing.py   # Tests for cleaning, mapping & resampling
```

---

## Dataset Acquisition and Data Integrity

### Preferred Source: UCI Online Retail Dataset

- **Official URL:** [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Domain Context:** Real-world transactions from a UK-based registered non-store online retailer spanning December 2010 to December 2011.
- **Important Disclosure:** The UCI Online Retail dataset contains giftware, household items, and apparel. This project applies an automated, transparent keyword classification to extract distinct clothing and accessory segments. It is clearly labeled as an apparel-focused subset analysis and does not claim that every original transaction is apparel.

### Offline Sample Fallback

For environments where internet connectivity is restricted or the public repository is unreachable, `src/data_loader.py` includes a deterministic generator that constructs 104 weeks (2 full years) of realistic fashion retail transactions with seasonal swings, holiday peaks, and return patterns.

### Raw Data Dictionary

| Column | Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | String | 6-digit transaction ID. A leading 'C' denotes a cancellation. |
| `StockCode` | String | Unique product identifier code. |
| `Description` | String | Product name and descriptive keywords. |
| `Quantity` | Integer | Transaction unit quantity. Negative values denote returns. |
| `InvoiceDate` | Datetime | Date and time when the transaction occurred. |
| `UnitPrice` | Float | Unit price in Sterling (£). |
| `CustomerID` | String/Float | Unique 5-digit customer identifier. |
| `Country` | String | Country of residence for the customer. |

---

## Data Preprocessing and Fashion Taxonomy

The data cleaning pipeline (`src/preprocessing.py`) executes the following operations:

1. **Schema Normalization:** Standardizes arbitrary column casings (`qty` $\to$ `Quantity`, `invoice_no` $\to$ `InvoiceNo`).
2. **Deduplication:** Strips exact duplicate rows across all fields.
3. **Date Conversion:** Parses timestamps into timezone-naive `datetime64[ns]`.
4. **Returns and Cancellation Handling:** Identifies negative quantities and cancellation prefixes (`C`). Forward purchase volume is isolated for gross demand forecasting, with return volume tracked separately.
5. **Fashion Keyword Taxonomy:** Maps product descriptions using word-boundary regular expressions:
   - **Tops:** `shirt`, `blouse`, `tee`, `t-shirt`, `top`, `tunic`, `polo`, `hoodie`, `sweater`, `cardigan`
   - **Dresses:** `dress`, `gown`, `skirt`, `sundress`, `midi`, `maxi`
   - **Outerwear:** `coat`, `jacket`, `blazer`, `parka`, `trench`, `overcoat`, `puffer`
   - **Trousers:** `pant`, `trouser`, `jean`, `legging`, `short`, `chino`, `jogger`
   - **Accessories:** `bag`, `tote`, `scarf`, `glove`, `hat`, `belt`, `necklace`, `earring`, `sunglasses`
   - **Footwear:** `boot`, `shoe`, `sneaker`, `sandal`, `slipper`, `heel`, `loafer`, `sock`
   - **Other:** Unmatched items.
6. **Time-Series Regularization:** Aggregates transactions into weekly intervals (`W-MON`). Missing intervals within the observed active date range are explicitly indexed with 0 demand.

---

## Leakage-Free Feature Engineering

To guarantee zero future data leakage, all features are constructed strictly using information available *prior* to the target period:

- **Calendar Harmonics:**
  - `week_of_year`, `month`, `quarter`, `time_index`
  - Cyclical encodings: $\sin\left(\frac{2\pi \cdot \text{week}}{52.18}\right)$, $\cos\left(\frac{2\pi \cdot \text{week}}{52.18}\right)$
- **Shifted Historical Lags:**
  - $\text{Lag}_1 = y_{t-1}$
  - $\text{Lag}_2 = y_{t-2}$
  - $\text{Lag}_4 = y_{t-4}$
  - $\text{Lag}_8 = y_{t-8}$
- **Shifted Rolling Windows:**
  - $\text{RollingMean}_4 = \frac{1}{4} \sum_{i=1}^{4} y_{t-i}$
  - $\text{RollingStd}_4 = \sqrt{\frac{1}{3} \sum_{i=1}^{4} (y_{t-i} - \bar{y})^2}$
  - $\text{RollingMean}_8 = \frac{1}{8} \sum_{i=1}^{8} y_{t-i}$

---

## Forecasting Models

### 1. Naive Baseline
Assumes demand in future period $t+h$ equals the last observed actual demand:

$$\hat{y}_{t+h} = y_t$$

### 2. 4-Week Moving Average Baseline
Assumes future demand continues at the recent 4-week average rate:

$$\hat{y}_{t+h} = \frac{1}{4}\sum_{i=0}^{3} y_{t-i}$$

### 3. Random Forest Regressor (ML)
An ensemble of 100 decorrelated decision trees (`max_depth=6`, `min_samples_leaf=2`) capturing non-linear relationships between recent sales momentum and seasonal calendar indices.

#### Multi-Step Recursive Strategy
When predicting $H$ steps into the future, the model predicts step $t+1$, appends $\hat{y}_{t+1}$ into the historical buffer, recomputes lags and rolling statistics, and predicts step $t+2$ iteratively up to step $t+H$.

#### Uncertainty Quantification
The model computes the empirical standard error of residuals $\sigma_\epsilon$ from validation. The 90% prediction interval is calculated as:

$$\text{Interval}_{t+h} = \left[ \max\left(0, \hat{y}_{t+h} - 1.645 \cdot \sigma_\epsilon \sqrt{1 + 0.15 \cdot h}\right), \; \hat{y}_{t+h} + 1.645 \cdot \sigma_\epsilon \sqrt{1 + 0.15 \cdot h} \right]$$

---

## Time-Series Validation and Metrics

### Chronological Splitting
**Random shuffling is never used.** Time-series cross-validation splits data chronologically:
- **Training Set:** The initial 80% of historical weeks (43 weeks on UCI dataset: Dec 2010 – Sep 2011).
- **Holdout Validation Set:** The final 20% of historical weeks (11 weeks on UCI dataset: Oct 2011 – Dec 2011).

### Evaluation Metrics
- **MAE (Mean Absolute Error):** Measured in units sold.
  $$\text{MAE} = \frac{1}{N} \sum |y_i - \hat{y}_i|$$
- **RMSE (Root Mean Squared Error):** Penalizes large forecast misses heavily.
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$$
- **WAPE (Weighted Absolute Percentage Error):** Volume-weighted relative error, safe against zero-demand weeks.
  $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$
- **MAPE (Mean Absolute Percentage Error):** Evaluated strictly on non-zero periods.

### Actual Measured Results (UCI Online Retail — 'Tops' Category)

| Model Architecture | Model Type | MAE (Units) | RMSE (Units) | WAPE (%) | Holdout Weeks |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline (Last Value)** | Persistence Baseline | **178.82** | **249.84** | **24.23%** | 11 |
| **Random Forest Regressor** | Machine Learning (ML) | 312.89 | 381.94 | 42.40% | 11 |
| **4-Week Moving Average** | Statistical Moving Average | 324.68 | 389.53 | 44.00% | 11 |

*Empirical Takeaway: On the UCI transaction series, the Random Forest model outperformed the 4-week moving average baseline by leveraging multi-lag autocorrelation ($t-1, t-2, t-4, t-8$) and cyclical annual harmonics. As observed in volatile retail settings with short history, persistence baselines offer a strong low-variance benchmark, demonstrating why production forecasting pipelines must always benchmark ML against naive baselines.*

---

## Inventory Planning Insights

The dashboard translates demand forecasts into practical inventory ordering recommendations:

- **Lead Time Demand ($D_{LT}$):** Sum of forecasted demand across the supplier lead time window.
- **Safety Stock ($SS$):** User-configured buffer (e.g., 15%) applied to lead time demand.
- **Suggested Replenishment:**

$$\text{Order Quantity} = \max(0, \lceil D_{LT} + SS - \text{On-Hand Stock} \rceil)$$

*Note: This is an inventory planning heuristic and does not guarantee optimal service levels under stochastic lead-time variances.*

---

## Installation and Environment Setup

### Prerequisites

- Python 3.11+ installed.
- Git installed.

### 1. Clone the Repository

```bash
git clone https://github.com/anshikakhan22-afk/fashion-demand-forecasting.git
cd fashion-demand-forecasting
```

### 2. Create and Activate Virtual Environment

#### Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

#### macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation and Model Training

Run the complete pipeline end-to-end with three simple commands:

### Step 1: Acquire Dataset

```bash
python scripts/download_data.py
```

### Step 2: Clean Data and Build Weekly Demand Series

```bash
python scripts/prepare_data.py
```

*Outputs: `data/processed/cleaned_sales.csv`, `data/processed/weekly_demand.csv`, and `reports/data_summary.json`.*

### Step 3: Train Models and Evaluate Holdout

```bash
python scripts/train_model.py
```

*Outputs: `models/demand_forecaster.joblib`, `reports/model_comparison.json`, and `reports/sample_forecast.csv`.*

---

## Running the Streamlit Dashboard

Launch the interactive web application from the project root:

```bash
streamlit run app.py
```

Streamlit will host the dashboard locally (default URL: `http://localhost:8501`).

### Troubleshooting Common Launch Issues:

- **Port in use:** Run `streamlit run app.py --server.port 8502`.
- **Module not found:** Ensure the virtual environment is activated (`.\.venv\Scripts\activate`).
- **Data not found:** Run `python scripts/prepare_data.py` to generate the processed weekly demand files.

---

## Automated Testing

Run the automated pytest suite:

```bash
pytest -q
```

The test suite covers:

- Data normalization, duplicate stripping, date parsing, and cancellation filtering.
- Fashion taxonomy mapping and coverage calculations.
- Contiguous weekly resampling and zero-sales period filling.
- Leakage-safe shifted lag and rolling feature calculations.
- Naive baseline, moving average, and Random Forest recursive multi-step predictions.
- Non-negativity constraints and future date alignment.
- Evaluation metrics calculation and zero-actual edge cases.

---

## Exporting Forecasts

1. In the Streamlit app, navigate to the **Demand Forecast** tab.
2. Review the weekly forecasted units, planning quantities, and 90% confidence bounds.
3. Click the **Download Forecast as CSV** button.
4. Alternatively, examine the automated batch export at `reports/sample_forecast.csv`.

---

## Known Limitations and Future Roadmap

### Current Limitations:

- **Catalog Taxonomy:** Because general retail transaction datasets lack standardized clothing hierarchies, keyword mapping is an approximation and may misclassify ambiguous item descriptions.
- **Exogenous Variables:** The current model does not ingest promotional calendars, discount markdowns, or local weather data.
- **Lead Time Variance:** Supplier replenishment assumes deterministic lead times.

### Future Roadmap:

- Integrate LightGBM and XGBoost for gradient boosting comparisons.
- Incorporate promotional price elasticity features.
- Implement SKU-level hierarchical reconciliation (e.g., MinT / Top-Down aggregation).
- Add automated email/Slack alerts for low-stock inventory thresholds.

---

## GitHub Repository Setup

To publish any updates to your GitHub repository:

```bash
git add README.md
git commit -m "docs: fix markdown formatting, encoding, and table of contents anchors in README"
git push origin main
```

---

## Attribution and License

- **Code License:** [MIT License](LICENSE) — free for academic, personal, and commercial usage.
- **Dataset Attribution:** Online Retail Dataset provided courtesy of Dr. Daqing Chen, School of Engineering, London South Bank University, London SE1 0AA, UK, via the UCI Machine Learning Repository.
