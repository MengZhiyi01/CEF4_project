
# Intraday Return Prediction in the Chinese Futures Market

This repository contains the final project for the Deep Learning course at Peking University HSBC Business School. The project focuses on using deep learning models to predict 5-minute intraday returns in the Chinese futures market.

## 📁 Project Structure

```

CEF4\_project/
│
├── config/                  # Configuration files for futures multipliers and margin ratios
│
├── data/                    # All data used in the project
│   ├── backtest/            # Backtesting results (processed data)
│   ├── input/               # Model input data
│   ├── output/              # Model output (predictions, plots, evaluations)
│   ├── processed/           # Preprocessed intermediate files
│   ├── raw/                 # Original raw data
│   └── sample/              # Sample futures data (e.g. AG2408)
│
├── experiments/
│   └── models/              # Saved models and tuning results
│       ├── ALSTM\_tuned/               # Regression models using ALSTM
│       ├── ALSTM\_tuned\_classification/ # Classification models using ALSTM
│       └── MASTER\_tuned\_classification/ # Classification models using MASTER
│
├── notebooks/              # Jupyter notebooks for model training and backtesting
│   ├── backtest\_alstm.ipynb
│   └── backtest\_master.ipynb
│
├── src/                    # Core source code for factor generation, modeling, and backtesting
│   ├── alstm\_*.py          # ALSTM training and visualization scripts
│   ├── master\_*.py         # MASTER model training and visualization scripts
│   ├── backtest\_generate\_trade.py   # Trade generation logic
│   ├── backtest\_analyze.py          # Performance analysis
│   ├── std\_tag\_generation.py        # Label generation
│   ├── Generate\_alpha158.ipynb      # Factor construction (Alpha158-style)
│   └── Generate\_qlib.py             # Qlib format conversion
│   └── data\preprocess.py             # data preprocessing for level 2 data & 1 min k bars
│
└── md.txt                  # Optional notes

```

## 📓 Notebooks

- `notebooks/backtest_alstm.ipynb`: Demonstrates backtesting based on predictions from the ALSTM model.
- `notebooks/backtest_master.ipynb`: Demonstrates backtesting based on predictions from the MASTER model.

## 🔧 Responsibilities

This project was a team collaboration with the following roles:

- **Factor generation, modeling, and training on the `main` branch**: *Xinlei Hao (郝心蕾)*
- **Factor generation, modeling, and training on the `master` branch**: *Haoyuan Wei (韦皓元)*
- **Raw data processing and backtesting framework**: *Jinzhan Lin (林劲展)*

## 🎓 Course Information

- **Course**: Deep Learning (2024)
- **Instructor**: Professor Xianhua Peng
- **Institution**: Peking University HSBC Business School
- **Objective**: Predict 5-minute intraday returns for major Chinese futures contracts using deep learning.

## 📈 Models

The project compares the performance of multiple architectures including:
- ALSTM (Attentive LSTM)
- MASTER (Transformer-based architecture)

Each model is trained to predict future return directions or magnitudes based on order book and aggregated feature data.

---

For any questions or suggestions, feel free to open an issue or contact the team members.
