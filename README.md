# Stock Analyzer

A Python tool for fetching, cleaning, and analyzing NSE stock data with technical indicators and candlestick pattern detection.

## Features
- Fetches bhavcopy data from NSE via [`nselib`](https://pypi.org/project/nselib/).
- Standardizes OHLCV data and adjusts for stock splits.
- Detects:
  - **Technical Indicators**: RSI, SMA (20/50/100/200), ADX, OBV, Bollinger Bands.
  - **Patterns**: Breakouts, squeezes, 52-week highs/lows, swing highs/lows, volume spikes.
  - **Candlestick Patterns** via [TA-Lib](https://mrjbq7.github.io/ta-lib/).
- Produces snapshot CSV reports for analysis.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/gitsha256/stockAnalyerV3.git
   cd stock-analyzer
2. Install dependencies:

pip install -r requirements.txt


⚠️ For Windows users: install TA-Lib-bin instead of TA-Lib:

pip install TA-Lib-bin

Usage

Run the analyzer:

python analyzer.py


You’ll be prompted to choose:

Fetch → Fetch raw data from NSE

Update → Update existing dataset with new data

Adjust → Adjust for stock splits

Analyze → Run full technical analysis

Outputs:

raw_data.csv → Raw fetched OHLCV data

data.csv → Adjusted OHLCV data

detected_splits.csv → Split log

snapshot.csv → Analysis report (latest snapshot)

Project Structure

analyzer.py → Main script

symbols.csv → Input list of symbols (+ optional SECTOR and HOLIDAYS)

workflow.log → Log file

Example
python analyzer.py
# Select option 1 → Fetch
# Enter start/end dates
# Results saved in raw_data.csv and data.csv

Notes

Requires internet access to fetch NSE data.

Needs TA-Lib properly installed. On Windows:

pip install TA-Lib-bin


For Linux/macOS:

sudo apt-get install ta-lib
pip install TA-Lib
