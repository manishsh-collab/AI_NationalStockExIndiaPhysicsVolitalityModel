# AI National Stock Exchange India — Physics-Responsive Volatility Model

A research/demo project that combines simple physics-inspired kinematics with machine learning to forecast short-term market volatility for the NIFTY 50 index (^NSEI). The repository contains `main.py`, a single-file prototype which:

- Downloads historical NIFTY data via `yfinance` (with a robust fallback that generates synthetic data if download fails).
- Constructs "physics" signals (velocity, acceleration, jerk, snap) using log-returns and EMA-based stiffness/damping.
- Builds features (including physics signals and lagged volatility) and trains a gradient-boosted regressor to forecast future volatility.
- Prices short-dated options using Black–Scholes with actual vs predicted volatility.
- Visualizes results (option price comparisons, the "snap" signal, and volatility forecasts) and prints simple error metrics.

This project is intended as an illustrative experiment — not investment advice.

---

## Table of contents

- [Features](#features)
- [How it works (high-level)](#how-it-works-high-level)
- [Quickstart](#quickstart)
- [Configuration & parameters](#configuration--parameters)
- [Outputs](#outputs)
- [Caveats & notes](#caveats--notes)
- [Extending / Ideas](#extending--ideas)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Robust data ingestion with fallback synthetic data generation (prevents runtime crashes if `yfinance` fails).
- Physics-inspired signals:
  - EMA and EMA-based standard deviation for adaptive bandwidth.
  - Stiffness (k) and damping (c) derived from volatility & volume dynamics.
  - Kinematic integration that computes acceleration, jerk (3rd derivative), and snap (4th derivative).
- Feature engineering for volatility forecasting (rolling realized vol, lagged vols/returns).
- Model: scikit-learn `HistGradientBoostingRegressor` for predicting future annualized volatility.
- Simple Black–Scholes option pricing comparison using actual, predicted, and current volatility estimates.
- Visual outputs: option price series, snap signal, and volatility forecast.

---

## How it works (high-level)

1. Data ingestion
   - Attempts to download 10 years of historical data for `^NSEI` using `yfinance`.
   - If the download fails or the data is insufficient, the script generates synthetic daily price data for demonstration.

2. Physics engine
   - Computes EMA (span=10) and EMA-based std to derive an adaptive price bandwidth.
   - Calculates stiffness (`k`) from bandwidth and damping (`c`) from relative volume.
   - Uses log returns as driving forces and numerically integrates to get velocity, acceleration, jerk, and snap.
   - Produces physics-based signals (e.g., `Phys_Power`, `Signal_Jerk`, `Signal_Snap`).

3. Features & target
   - Target: future realized volatility over a 21-trading-day window (annualized) shifted forward.
   - Features include `k`, `Phys_Power`, physics signals, current realized vol, lagged vol, and lagged returns.

4. Modeling
   - Trains a `HistGradientBoostingRegressor` on the features and target using an 85/15 train/test split.
   - Predicts volatility on the test set.

5. Visualization & evaluation
   - Uses predicted vol vs actual vol to price short-dated options with Black–Scholes.
   - Plots the option price comparison, the snap signal, and the volatility forecast.
   - Prints mean absolute pricing errors for the standard (current vol) and AI-predicted vol approaches.

---

## Quickstart

Prerequisites
- Python 3.8+ (3.9/3.10 recommended)
- pip

Install dependencies (recommended within a virtual environment):
```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas yfinance matplotlib scipy scikit-learn
```

Run
```bash
python main.py
```

The script will attempt to download data; if that fails it will create synthetic data and continue. A plotting window (matplotlib) will appear showing the results. The console prints training/test split and simple error metrics.

---

## Configuration & parameters

All fast-editable settings are in `main.py`:

- `ticker` — default: `^NSEI`. Change to any supported yfinance ticker.
- Data period — currently set inside the download call to `period="10y"`.
- EMA span — currently 10 (used for quick reaction).
- `TRADING_DAYS` — window used for realized volatility and target horizon (default 21).
- Model hyperparameters — configured in the `HistGradientBoostingRegressor` constructor (learning rate, max_iter, max_depth, loss).
- Black–Scholes parameters:
  - `T` — time-to-expiry in years (default 30/365).
  - `r` — risk-free rate (default 0.07).

Suggested edits: modify these values directly in `main.py` for experimentation.

---

## Outputs

- Matplotlib figures:
  - Option price comparison (Ideal vs AI vs Standard).
  - Snap signal time series (physics 4th derivative).
  - Volatility forecast (actual vs predicted).
- Console:
  - Training/Testing sample counts
  - Mean absolute error for price approximations and a relative gain percent.

---

## Caveats & notes

- This is a prototype/educational demo:
  - Not designed for production trading.
  - No transaction costs, liquidity constraints, bid/ask spreads, or risk management are modeled.
  - The synthetic-data fallback is for demo continuity only — treat synthetic output accordingly.
- The model and physics heuristics are simplistic and provided to demonstrate a concept.
- Results depend heavily on data quality, the amount of history, and hyperparameter choices.
- If you run in a headless environment (e.g., remote server without display), either configure matplotlib to use a non-interactive backend or modify the script to save plots to files instead of calling `plt.show()`.

---

## Extending / Ideas

- Add a `requirements.txt` and CI tests.
- Persist trained models and add evaluation metrics (e.g., MAE, RMSE, calibration).
- Add cross-validation and a more diverse feature set (macro indicators, sector signals).
- Replace synthetic fallback with cached data or a local CSV for reproducible experiments.
- Wrap in a Jupyter notebook for interactive exploration and plotting.
- Add option Greeks and stress tests for the option-pricing component.
- Parameterize the script and expose a CLI/API for reproducible runs.

---

## Contributing

Contributions are welcome. For substantial changes, please open issues or pull requests describing the proposed improvement. When contributing, please:

- Keep changes well-documented.
- Add tests for critical logic where appropriate.
- If adding dependencies, explain the rationale in the PR.

---

## License

No license is provided in this repository. If you intend to reuse or publish code from here, add an explicit license (for example, MIT) by creating a `LICENSE` file.

---

File reference
- Main prototype: [main.py](https://github.com/manishsh-collab/AI_NationalStockExIndiaPhysicsVolitalityModel/blob/d22778760f732b361eb3f1ce17043934b1c4526d/main.py) (commit d227787)
