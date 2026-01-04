# Energy Market Causal Analytics (EMCA)

## Project Overview

The EMCA project is a modular, phase-based portfolio designed to bridge advanced mathematical, statistical, and machine learning concepts with real-world energy market data. Each phase is inspired by a foundational textbook and applies new concepts to energy market analytics, building both theoretical understanding and practical data science skills.

---

## Learning Goals

- Apply signal processing, time series analysis, econometrics, spectral analysis, machine learning, and Bayesian methods to real energy market data.
- Map abstract mathematical concepts onto noisy, high-frequency datasets from the energy sector.
- Develop skills in functional programming (for data integrity) and object-oriented programming (for orchestration and system state management).

---

## Data Types & Attributes

- **Market Data:**
  - Locational Marginal Prices (LMPs), demand, generation, outages.
  - Attributes: timestamp, region, price, demand, fuel mix.
- **Weather Data:**
  - Temperature, wind speed, solar irradiance.
  - Attributes: timestamp, location, measurement type.
- **Event/Policy Data:**
  - Regulatory changes, plant outages, market interventions.
  - Attributes: event time, description, affected assets/regions.

---

## Phase-by-Phase Roadmap

### Phase 1: Signals (Lathi Time-Domain)
- **Learning Goals:** Signal classification, sampling, convolution.
- **Data:** Demand/price time series.
- **Strategy:** Model grid response to demand shocks using convolution.

### Phase 2: Laplace & Z-Transforms
- **Learning Goals:** System stability, transfer functions, poles/zeros.
- **Data:** Sampled price/demand series.
- **Strategy:** Use Z-transform to analyze stability and response to shocks.

### Phase 3: Basic Time Series (Stoffer)
- **Learning Goals:** Stationarity, autocorrelation, SARIMA.
- **Data:** Multi-year demand/price series.
- **Strategy:** Forecast residual demand, decompose trends and seasonality.

### Phase 4: Econometrics (Wooldridge)
- **Learning Goals:** Endogeneity, IV regression, panel data.
- **Data:** Price, demand, weather, fuel prices.
- **Strategy:** Estimate causal effects using instrumental variables.

### Phase 5: Fourier Series & Transforms
- **Learning Goals:** Frequency analysis, periodicity, FFT.
- **Data:** Long time-series (years).
- **Strategy:** Identify dominant cycles/harmonics in prices.

### Phase 6: Spectral Time Series (Stoffer Advanced)
- **Learning Goals:** Spectral density, coherence.
- **Data:** Price, wind speed, demand.
- **Strategy:** Measure coherence between physical signals and market volatility.

### Phase 7: ML/DL & Nonparametrics
- **Learning Goals:** Sequential modeling (RNN/LSTM), Gaussian Processes.
- **Data:** High-frequency price/demand, frequency-domain features.
- **Strategy:** Predict future prices using deep learning and nonparametric models.

### Phase 8: Causal Sequential ML
- **Learning Goals:** Double Machine Learning, treatment effects.
- **Data:** Price, demand, event data (e.g., outages).
- **Strategy:** Perform counterfactual analysis using causal ML on sequential data.

### Phase 9: Bayesian Data Analysis
- **Learning Goals:** Bayesian inference, uncertainty quantification.
- **Data:** All previous datasets.
- **Strategy:** Apply Bayesian models to quantify uncertainty and update beliefs.

---

## Project Outcomes

- Demonstrate mastery of mathematical, statistical, and ML concepts in a real-world context.
- Build a portfolio of modular, well-documented analytical projects.
- Gain practical experience with real energy market and weather datasets.
- Develop a deep understanding of the interplay between physical systems and economic outcomes.

---

## Repository Structure (for reference)

```
EMCA_Project/
├── data/                  # Raw and processed datasets (CSV/Parquet)
├── src/
│   ├── core/              # Functional Programming Layer (Pure Math)
│   ├── orchestrator/      # OOP Layer (System State)
├── notebooks/             # Phase-by-Phase Exploration
├── tests/                 # Unit tests for functional core
└── README.md              # This Document
```
