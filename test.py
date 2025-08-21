import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from nselib import capital_market
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands
try:
    import talib
    TA_LIB_AVAILABLE = True
except Exception:
    talib = None
    TA_LIB_AVAILABLE = False
import logging

# Configuration
CONFIG = {
    'SYMBOLS_FILE': 'symbols.csv',
    'RAW_DATA_FILE': 'raw_data.csv',
    'ADJUSTED_DATA_FILE': 'data.csv',
    'SPLITS_LOG_FILE': 'detected_splits.csv',
    'ANALYSIS_OUTPUT_FILE': 'snapshot.csv',
    'MAX_WORKERS': 6,
    'DEFAULT_LOOKBACK_DAYS': 252,
    'FETCH_INTERVAL': 'D',
    'REQUIRED_COLUMNS': ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbols'],
}

# Setup logging
def setup_logging(verbose=True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler('workflow.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# Modified load_symbols to handle holidays column
def load_symbols(filepath, logger):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.upper()
        if 'SYMBOL' not in df.columns:
            logger.error(f"'SYMBOL' column missing in {filepath}")
            return [], pd.DataFrame(), set()
        df['SYMBOL'] = df['SYMBOL'].astype(str).str.upper().str.strip().str.replace('.NS', '', regex=False).str.replace('-EQ', '', regex=False)
        symbols = df['SYMBOL'].dropna().unique().tolist()
        sector_df = df[['SYMBOL', 'SECTOR']].drop_duplicates().rename(columns={'SYMBOL': 'symbols'}) if 'SECTOR' in df.columns else pd.DataFrame()
        
        # Extract holidays
        holidays = set()
        if 'HOLIDAYS' in df.columns:
            holiday_list = df['HOLIDAYS'].dropna().astype(str).str.strip().str.split(';').explode().str.strip()
            for date_str in holiday_list:
                try:
                    holiday_date = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
                    if pd.notna(holiday_date):
                        holidays.add(holiday_date.date())
                except Exception as e:
                    logger.warning(f"Invalid holiday date format '{date_str}' in {filepath}: {e}")
        else:
            logger.warning(f"No 'HOLIDAYS' column found in {filepath}. Assuming no holidays.")
        
        logger.info(f"Loaded {len(symbols)} symbols and {len(holidays)} holiday dates")
        return symbols, sector_df, holidays
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return [], pd.DataFrame(), set()

# Modified get_nse_holiday_dates to use holidays from symbols.csv
def get_nse_holiday_dates(symbols_file, logger):
    _, _, holidays = load_symbols(symbols_file, logger)
    logger.info(f"Holidays loaded from {symbols_file}: {sorted(holidays)}")
    return holidays

def standardize_data(df, filepath='', logger=None):
    if df.empty:
        if logger: logger.warning(f"No data in {filepath or 'DataFrame'}")
        return df
    try:
        df.columns = df.columns.str.strip().str.lower()
        if not all(col in df.columns for col in CONFIG['REQUIRED_COLUMNS']):
            if logger: logger.error(f"Missing required columns in {filepath or 'DataFrame'}: {CONFIG['REQUIRED_COLUMNS']}")
            return pd.DataFrame()
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['symbols'] = df['symbols'].astype(str).str.upper().str.strip().str.replace('.NS', '', regex=False).str.replace('-EQ', '', regex=False)
        df = df.dropna(subset=['datetime', 'symbols']).sort_values(['symbols', 'datetime']).drop_duplicates(['symbols', 'datetime'], keep='last')
        if logger:
            logger.info(f"Standardized {len(df)} records from {filepath or 'DataFrame'}")
        return df
    except Exception as e:
        if logger: logger.error(f"Error standardizing data from {filepath}: {e}")
        return pd.DataFrame()

def fetch_and_format(trade_date, symbol_filter=None, logger=None):
    try:
        spot_data = capital_market.bhav_copy_with_delivery(trade_date=trade_date)
        logger.info(f"Raw data for {trade_date}: {spot_data.shape} rows, columns: {spot_data.columns.tolist()}")
        ohlcv_columns = [
            'SYMBOL', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'TTL_TRD_QNTY', 'DELIV_PER'
        ]
        available_cols = [col for col in ohlcv_columns if col in spot_data.columns]
        if not all(col in spot_data.columns for col in ['SYMBOL', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'TTL_TRD_QNTY']):
            if logger: logger.warning(f"Missing expected columns in bhav copy for {trade_date}: {available_cols}")
            return None
        spot_ohlcv = spot_data[available_cols].copy()

        numeric_columns = [col for col in ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'TTL_TRD_QNTY', 'DELIV_PER'] if col in spot_ohlcv.columns]
        for col in numeric_columns:
            spot_ohlcv[col] = pd.to_numeric(spot_ohlcv[col], errors='coerce')

        spot_ohlcv.rename(columns={
            'SYMBOL': 'symbols',
            'OPEN_PRICE': 'open',
            'HIGH_PRICE': 'high',
            'LOW_PRICE': 'low',
            'CLOSE_PRICE': 'close',
            'TTL_TRD_QNTY': 'volume',
            'DELIV_PER': 'delivery_perc'
        }, inplace=True)

        spot_ohlcv['symbols'] = spot_ohlcv['symbols'].str.upper().str.strip()
        spot_ohlcv['datetime'] = pd.to_datetime(trade_date, format='%d-%m-%Y')
        spot_ohlcv = spot_ohlcv.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        if symbol_filter is not None:
            spot_ohlcv = spot_ohlcv[spot_ohlcv['symbols'].isin(symbol_filter)]

        cols = spot_ohlcv.columns.tolist()
        if 'delivery_perc' in cols:
            cols.remove('delivery_perc')
            close_idx = cols.index('close')
            cols = cols[:close_idx+1] + ['delivery_perc'] + cols[close_idx+1:]
            spot_ohlcv = spot_ohlcv[cols]

        logger.info(f"Fetched {spot_ohlcv.shape[0]} records for {trade_date} after processing")
        return spot_ohlcv
    except Exception as e:
        if logger: logger.warning(f"Failed to fetch data for {trade_date}: {e}")
        return None

def fetch_data(symbols, from_date, to_date, logger):
    all_data = []
    date_list = []
    current_date = from_date

    nse_holidays = get_nse_holiday_dates(CONFIG['SYMBOLS_FILE'], logger)
    logger.info(f"Holidays: {sorted(nse_holidays)}")

    try:
        while current_date <= to_date:
            if current_date.weekday() < 5 and current_date.date() not in nse_holidays:
                date_list.append(current_date.strftime('%d-%m-%Y'))
            current_date += timedelta(days=1)

        logger.info(f"Fetching data for {len(date_list)} dates (weekdays & non-holidays): {date_list}")
        if not date_list:
            logger.warning("No valid trading dates to fetch")
            return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            futures = {executor.submit(fetch_and_format, d, symbols, logger): d for d in date_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
                trade_date = futures[future]
                result = future.result()
                if result is not None and not result.empty:
                    all_data.append(result)
                else:
                    logger.warning(f"No valid data fetched for {trade_date}")

        if not all_data:
            logger.error("No data fetched")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = standardize_data(combined_df, logger=logger)
        
        for symbol in symbols:
            symbol_data = combined_df[combined_df['symbols'] == symbol]
            if len(symbol_data) == 0:
                logger.warning(f"Symbol {symbol} has no data after standardization")
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        return pd.DataFrame()

def detect_splits(symbol_data, logger):
    try:
        if len(symbol_data) < 2:
            symbol = symbol_data['symbols'].iloc[0] if not symbol_data.empty else 'unknown'
            logger.warning(f"Skipping split detection for {symbol}: insufficient data ({len(symbol_data)} rows)")
            return pd.DataFrame()

        symbol_data = symbol_data.sort_values('datetime').reset_index(drop=True)
        price_diffs = symbol_data['close'].pct_change()
        splits = []

        for idx in price_diffs[price_diffs <= -0.3].index:
            if idx == 0:
                continue
            prev, curr = symbol_data['close'].iloc[idx - 1], symbol_data['close'].iloc[idx]
            if pd.isna(prev) or pd.isna(curr):
                continue
            ratio = prev / curr
            if 1.5 <= ratio <= 10:
                splits.append({
                    'symbols': symbol_data['symbols'].iloc[0],
                    'SPLIT_DATE': symbol_data['datetime'].iloc[idx],
                    'SPLIT_RATIO': round(ratio, 2),
                    'PREV_CLOSE': prev,
                    'CURR_CLOSE': curr
                })

        return pd.DataFrame(splits)
    
    except Exception as e:
        logger.error(f"Error detecting splits for {symbol_data['symbols'].iloc[0] if not symbol_data.empty else 'unknown'}: {e}")
        return pd.DataFrame()

def adjust_prices(df, logger):
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Skipping adjustment.")
        return df, pd.DataFrame()

    try:
        splits = pd.concat(
            [detect_splits(df[df['symbols'] == s], logger) for s in df['symbols'].unique()],
            ignore_index=True
        )
        adjusted = df.copy()
        if not splits.empty:
            for _, split in splits.iterrows():
                mask = (adjusted['symbols'] == split['symbols']) & (adjusted['datetime'] < split['SPLIT_DATE'])
                adjusted.loc[mask, ['open', 'high', 'low', 'close']] /= split['SPLIT_RATIO']
            logger.info('Adjusted splits')
            print(f"Adjusted {len(adjusted)} records with {len(splits)} splits detected")
        return adjusted, splits
        
        
    except Exception as e:
        logger.error(f"Error during split adjustment: {e}")
        return df, pd.DataFrame()

def analyze_symbol(symbol, logger, enable_candle_patterns=False):
    global df
    data = df[df['symbols'] == symbol].sort_values('datetime')
    if data.empty:
        logger.warning(f"No data for symbol {symbol}")
        return None
    if len(data) < 20:
        logger.warning(f"Skipping analysis for {symbol}: insufficient data ({len(data)} rows)")
        return None
    try:
        data['Change'] = (data['close'] - data['open']) / data['open'] * 100
        data['RSI'] = RSIIndicator(data['close'], window=14).rsi()
        for p in [20, 50, 100, 200]:
            data[f'SMA_{p}'] = SMAIndicator(data['close'], window=p).sma_indicator()
        data['ADX'] = ADXIndicator(data['high'], data['low'], data['close'], window=14).adx()
        data['OBV'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
        bb = BollingerBands(data['close'], window=20, window_dev=2)
        data['BB_UPPER'], data['BB_LOWER'], data['BB_MIDDLE'] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
        data['BB_BANDWIDTH'] = (data['BB_UPPER'] - data['BB_LOWER']) / data['BB_MIDDLE']
        data['BB_SQUEEZE'] = data['BB_BANDWIDTH'].rolling(window=301, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 <= 5, raw=True).astype(bool)
        data['BB_BREAKOUT_UP'] = (data['close'] > data['BB_UPPER']) & (data['close'].shift(1) <= data['BB_UPPER'].shift(1))
        data['BB_BREAKOUT_DOWN'] = (data['close'] < data['BB_LOWER']) & (data['close'].shift(1) >= data['BB_LOWER'].shift(1))
        data['52W_High'] = data['high'].rolling(window=252, min_periods=1).max()
        data['52W_Low'] = data['low'].rolling(window=252, min_periods=1).min()
        data['SWING_HIGH'] = 0
        data['SWING_LOW'] = 0
        high_idx = argrelextrema(data['close'].values, np.greater_equal, order=252)[0]
        low_idx = argrelextrema(data['close'].values, np.less_equal, order=252)[0]
        data.iloc[high_idx, data.columns.get_loc('SWING_HIGH')] = 1
        data.iloc[low_idx, data.columns.get_loc('SWING_LOW')] = 1

        data['AVG_VOLUME_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
        data['RELATIVE_VOLUME'] = data['volume'] / data['AVG_VOLUME_20']
        data['VOLUME_SPIKE'] = data['RELATIVE_VOLUME'] > 2
        data['ACTIVITY_SCORE'] = data['close'] * data['volume'] / 1e7

        latest = data.iloc[-1]
        close = latest['close']
        swing_high = data[data['SWING_HIGH'] == 1]['close'].iloc[-1] if not data[data['SWING_HIGH'] == 1].empty else np.nan
        swing_low = data[data['SWING_LOW'] == 1]['close'].iloc[-1] if not data[data['SWING_LOW'] == 1].empty else np.nan
        midpoint = (swing_high + swing_low) / 2 if pd.notna(swing_high) and pd.notna(swing_low) else np.nan
        zone = "NO DATA" if pd.isna(midpoint) or pd.isna(close) else (
            "Equilibrium" if abs(close - midpoint) / midpoint <= 0.05 else
            "Discount" if abs(close - swing_low) / swing_low <= 0.03 else
            "Premium" if abs(close - swing_high) / swing_high <= 0.03 else
            "Near Premium" if close > midpoint else "Near Discount"
        )
        
        include_candles = enable_candle_patterns and TA_LIB_AVAILABLE
        candle = None
        more_candles = None
        if include_candles:
            candle_functions = {name: func for name, func in talib.__dict__.items()
                                if name.startswith("CDL") and callable(func)}
            latest_pattern = None
            highest_score = 0
            patterns_detected = {}
            for name, func in candle_functions.items():
                try:
                    result = func(data['open'], data['high'], data['low'], data['close'])
                    last_val = result.iloc[-1]
                    if last_val != 0:
                        clean_name = name.replace("CDL", "").replace("_", " ").title()
                        sentiment = "Bullish" if last_val > 0 else "Bearish"
                        patterns_detected[name] = f"{clean_name} ({sentiment})"
                        score = abs(last_val)
                        if score > highest_score:
                            highest_score = score
                            latest_pattern = name
                except Exception as e:
                    logger.warning(f"Error processing candlestick {name} for {symbol}: {e}")
            candle = patterns_detected.get(latest_pattern, "No Pattern")
            more_candles = ', '.join([v for k, v in patterns_detected.items() if k != latest_pattern]) if latest_pattern else ', '.join(patterns_detected.values())
        else:
            if enable_candle_patterns and not TA_LIB_AVAILABLE:
                logger.info("TA-Lib not available; skipping candlestick pattern analysis.")
            # Leave candle columns absent when disabled
        
        above_sma_200 = close > latest['SMA_200'] if pd.notna(latest['SMA_200']) else False
        above_sma_50 = close > latest['SMA_50'] if pd.notna(latest['SMA_50']) else False
        above_sma_20 = close > latest['SMA_20'] if pd.notna(latest['SMA_20']) else False
        bb_position = ((close - latest['BB_LOWER']) / (latest['BB_UPPER'] - latest['BB_LOWER'])
                       if pd.notna(latest['BB_UPPER']) and pd.notna(latest['BB_LOWER']) else 0.5)
        per = round((latest['52W_High'] - close) / latest['52W_High'] * 100, 2) if pd.notna(latest['52W_High']) else np.nan
        vol_trend = "Increasing" if latest['volume'] > data['volume'].rolling(window=10).mean().iloc[-1] else "Decreasing"
        trend = ("Uptrend" if latest['ADX'] > 25 and (latest['SMA_50'] - data['SMA_50'].iloc[-5]) / 5 > 0 else
                 "Downtrend" if latest['ADX'] > 25 else "Sideways")
        trend_strength = "Strong" if latest['ADX'] > 25 else "Moderate" if latest['ADX'] > 15 else "Weak"
        volatility = data['close'].pct_change().rolling(window=21).std().iloc[-1] * np.sqrt(252) * 100 if len(data) >= 21 else np.nan

        row = {
            'datetime': latest['datetime'], 'symbols': symbol, 'close': round(close, 2), 'volume': latest['volume'],
            'Dl Per': round(latest['delivery_perc'], 2) if 'delivery_perc' in latest else np.nan,
            'CHANGE': round(latest['Change'], 2), 'SMA_200': round(latest['SMA_200'], 2),
            'ZONE': zone, 'RSI': round(latest['RSI'], 2), 'delta': per,
            '>200': above_sma_200, '>50': above_sma_50, '>20': above_sma_20,
            'OBV': round(latest['OBV'], 2), 'VOLUME_TREND': vol_trend, 'ADX': round(latest['ADX'], 2),
            'open': round(latest['open'], 2), 'TREND': trend, 'TREND_STRENGTH': trend_strength,
            'BB_BREAKOUT_UP': latest['BB_BREAKOUT_UP'], 'BB_BREAKOUT_DOWN': latest['BB_BREAKOUT_DOWN'],
            'BB_BANDWIDTH': round(latest['BB_BANDWIDTH'], 4), 'BB_SQUEEZE': latest['BB_SQUEEZE'],
            'VOLATILITY_%': round(volatility, 2), 'S_HIGH': round(swing_high, 2),
            'S_LOW': round(swing_low, 2), 'high': round(latest['high'], 2), 'low': round(latest['low'], 2),
            'EQB': round(midpoint, 2),
            'RELATIVE_VOLUME': round(latest['RELATIVE_VOLUME'], 2), 'VOLUME_SPIKE': latest['VOLUME_SPIKE'],
            'ACTIVITY_SCORE': round(latest['ACTIVITY_SCORE'], 2), 'SMA20': round(latest['SMA_20'], 2),
            'SMA50': round(latest['SMA_50'], 2), 'SMA100': round(latest['SMA_100'], 2),
            '52HIGH': round(latest['52W_High'], 2), '52LOW': round(latest['52W_Low'], 2),
        }
        if candle is not None:
            row['CANDLE'] = candle
        if more_candles is not None:
            row['MORECDL'] = more_candles
        return pd.DataFrame([row])
    
    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {e}")
        return None

def perform_technical_analysis(df_input, sector_df, logger, enable_candle_patterns=False):
    global df
    df = df_input
    if df.empty:
        logger.warning("No data for analysis")
        return pd.DataFrame()
    
    try:
        valid_symbols = [s for s in df['symbols'].unique() if len(df[df['symbols'] == s]) >= 20]
        logger.info(f"Found {len(valid_symbols)} symbols with sufficient data (>= 20 rows)")
        
        invalid_symbols = [s for s in df['symbols'].unique() if len(df[df['symbols'] == s]) < 20]
        for symbol in invalid_symbols:
            logger.warning(f"Symbol {symbol} has insufficient data: {len(df[df['symbols'] == symbol])} rows")
        
        results = []
        for symbol in tqdm(valid_symbols, desc="Analyzing data"):
            result = analyze_symbol(symbol, logger, enable_candle_patterns=enable_candle_patterns)
            if result is not None:
                results.append(result)
        
        if not results:
            logger.warning("No analysis results generated")
            return pd.DataFrame()
        
        analysis_df = pd.concat(results, ignore_index=True)
        
        analysis_df['VOLUME_RANK'] = analysis_df['volume'].rank(ascending=False, method='min')
        analysis_df['ACTIVITY_RANK'] = analysis_df['ACTIVITY_SCORE'].rank(ascending=False, method='min')
        analysis_df['RELATIVE_VOLUME_RANK'] = analysis_df['RELATIVE_VOLUME'].rank(ascending=False, method='min')
        analysis_df = analysis_df.sort_values(['ACTIVITY_SCORE', 'RELATIVE_VOLUME'], ascending=[False, False])
        
        if not sector_df.empty:
            try:
                analysis_df['symbols'] = analysis_df['symbols'].astype(str)
                sector_df['symbols'] = sector_df['symbols'].astype(str)
                analysis_df = analysis_df.merge(sector_df[['symbols', 'SECTOR']], on='symbols', how='left').fillna({'SECTOR': 'Unknown'})
                cols = [c for c in analysis_df.columns if c != 'SECTOR'] + ['SECTOR']
                analysis_df = analysis_df[cols]
            except Exception as e:
                logger.error(f"Error merging sector data: {e}")
        
        col_map = {
            'datetime': 'date', 'symbols': 'symb', 'close': 'clos', 'volume': 'volu',
            'Dl Per': 'DlPer', 'RELATIVE_VOLUME': 'rvol', 'VOLUME_SPIKE': 'vspk', 'ACTIVITY_SCORE': 'ascr',
            'ACTIVITY_RANK': 'arnk', 'CHANGE': 'chan', '>200': 'g200', 'ZONE': 'zone', 'RSI': 'rsi',
            'delta': 'delt', 'CANDLE': 'cand', 'OBV': 'obv', 'BB_BREAKOUT_UP': 'bbup', 'VOLUME_TREND': 'vtrd',
            'BB_BANDWIDTH': 'bbbw', 'MORECDL': 'mcdl', '>50': 'g050', '>20': 'g020', 'ADX': 'adx',
            'open': 'open', 'BB_BREAKOUT_DOWN': 'bbdn', 'BB_SQUEEZE': 'bbsq', 'S_HIGH': 'shgh', 'S_LOW': 'slw',
            'high': 'high', 'low': 'low', 'EQB': 'eqb', 'SMA20': 's020', 'SMA50': 's050', 'SMA100': 's100',
            'SMA_200': 's200', '52HIGH': 'h52h', '52LOW': 'l52l', 'VOLUME_RANK': 'vrnk', 'RELATIVE_VOLUME_RANK': 'rrnk',
            'TREND': 'tren', 'TREND_STRENGTH': 'tstr', 'VOLATILITY_%': 'vola', 'SECTOR': 'sect'
        }
        analysis_df.rename(columns={k: v for k, v in col_map.items() if k in analysis_df.columns}, inplace=True)
        
        desired_order = [
            'date','symb','clos','volu','DlPer','rvol','vspk','ascr','arnk','chan','g200','zone','rsi','delt','cand','bbup','vtrd','bbbw','bbsq',
            'mcdl','g050','g020','adx','open','bbdn','shgh','slw','high','low','eqb','s020','s050','s100','s200','h52h','l52l','vrnk','rrnk',
            'tren','tstr','vola','obv','sect'
        ]
        final_cols = [c for c in desired_order if c in analysis_df.columns]
        extra_cols = [c for c in analysis_df.columns if c not in final_cols]
        analysis_df = analysis_df[final_cols + extra_cols]
        
        logger.info(f"Generated analysis for {len(analysis_df)} symbols")
        return analysis_df
    
    except Exception as e:
        logger.error(f"Error in perform_technical_analysis: {e}")
        return pd.DataFrame()

def get_input(prompt):
    val = input(prompt)
    if val.strip().lower() == 'q':
        print("Operation cancelled by user.")
        exit(0)
    return val

def main():
    logger = setup_logging(verbose=True)
    try:
        symbols, sector_df, _ = load_symbols(CONFIG['SYMBOLS_FILE'], logger)
        if not symbols:
            logger.error("No symbols loaded")
            return

        print("Choose the operation mode:")
        print("1. Fetch")
        print("2. Update")
        print("3. Adjust")
        print("4. Analyze")

        choice = get_input("Enter your choice (1-4 or Q to quit): ").strip()

        raw_data = pd.DataFrame()
        if choice == '1':  # Fetch
            print("Choose date range mode:")
            print("1. Custom start and end dates")
            print("2. End date and years back")
            date_choice = input("Enter choice (1 or 2): ").strip()
            
            if date_choice == '1':
                start_date = input("Start date (DD-MM-YYYY): ").strip()
                end_date = input("End date (DD-MM-YYYY): ").strip()
                try:
                    start_date = datetime.strptime(start_date, '%d-%m-%Y')
                    end_date = datetime.strptime(end_date, '%d-%m-%Y')
                    if start_date > end_date:
                        logger.error("Start date cannot be after end date")
                        return
                    raw_data = fetch_data(symbols, start_date, end_date, logger)
                except ValueError as e:
                    logger.error(f"Invalid date format: {e}")
                    return
            
            elif date_choice == '2':
                end_date = input("End date (DD-MM-YYYY): ").strip()
                years_back = input("Years back (e.g., 1.5): ").strip()
                try:
                    end_date = datetime.strptime(end_date, '%d-%m-%Y')
                    years_back = float(years_back)
                    days_back = int(years_back * 365)
                    start_date = end_date - timedelta(days=days_back)
                    raw_data = fetch_data(symbols, start_date, end_date, logger)
                except ValueError as e:
                    logger.error(f"Invalid input for date or years: {e}")
                    return
            else:
                logger.error("Invalid date choice")
                return

        elif choice == '2':  # Update
            if not os.path.exists(CONFIG['RAW_DATA_FILE']):
                logger.error(f"No {CONFIG['RAW_DATA_FILE']} found")
                return
            try:
                existing = standardize_data(pd.read_csv(CONFIG['RAW_DATA_FILE']), CONFIG['RAW_DATA_FILE'], logger)
                if existing.empty:
                    logger.error(f"No valid data in {CONFIG['RAW_DATA_FILE']}")
                    return
                tasks = []
                to_date = datetime.now()
                logger.info(f"Checking for updates up to {to_date.strftime('%d-%m-%Y')}")
                for symbol in symbols:
                    last_date = existing[existing['symbols'] == symbol]['datetime'].max()
                    from_date = last_date + timedelta(days=1) if pd.notna(last_date) else to_date - timedelta(days=CONFIG['DEFAULT_LOOKBACK_DAYS'])
                    if from_date.date() <= to_date.date():
                        tasks.append(symbol)
                        logger.info(f"Symbol {symbol} needs update from {from_date.strftime('%d-%m-%Y')}")
                if tasks:
                    logger.info(f"Fetching new data for {len(tasks)} symbols")
                    new_data = fetch_data(tasks, from_date, to_date, logger)
                    if not new_data.empty:
                        new_data = new_data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                        if new_data.empty:
                            logger.warning("No new data with valid OHLCV values after filtering")
                            return
                        raw_data = pd.concat([existing, new_data]).drop_duplicates(['symbols', 'datetime'], keep='last')
                        logger.info(f"Combined {len(raw_data)} records after update")
                    else:
                        logger.warning("No new data fetched")
                        return
                else:
                    logger.info("No new data to fetch")
                    return
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                return

        elif choice == '3':  # Adjust
            if not os.path.exists(CONFIG['RAW_DATA_FILE']):
                logger.error(f"No {CONFIG['RAW_DATA_FILE']} found")
                return
            try:
                raw_data = standardize_data(pd.read_csv(CONFIG['RAW_DATA_FILE']), CONFIG['RAW_DATA_FILE'], logger)
                if raw_data.empty:
                    logger.error(f"No valid data in {CONFIG['RAW_DATA_FILE']}")
                    return
            except Exception as e:
                logger.error(f"Error reading raw data: {e}")
                return

        elif choice == '4':  # Analyze
            if not os.path.exists(CONFIG['ADJUSTED_DATA_FILE']):
                logger.error(f"No {CONFIG['ADJUSTED_DATA_FILE']} found")
                return
            try:
                adjusted_data = standardize_data(pd.read_csv(CONFIG['ADJUSTED_DATA_FILE']), CONFIG['ADJUSTED_DATA_FILE'], logger)
                if adjusted_data.empty:
                    logger.error(f"No valid data in {CONFIG['ADJUSTED_DATA_FILE']}")
                    return

                date_input = get_input("press Enter to analyze latest data or date/dates (DD-MM-YYYY to DD-MM-YYYY) ").strip()
                if date_input and 'to' in date_input:
                    try:
                        start_str, end_str = [d.strip() for d in date_input.split('to')]
                        start_date = datetime.strptime(start_str, '%d-%m-%Y')
                        end_date = datetime.strptime(end_str, '%d-%m-%Y')
                        # Ask once for candle pattern analysis
                        candle_choice = get_input("Enable candlestick pattern analysis? (y/N): ").strip().lower()
                        enable_candle_patterns = candle_choice in ['y', 'yes']
                        if enable_candle_patterns and not TA_LIB_AVAILABLE:
                            logger.info("TA-Lib is not installed. Candlestick analysis will be skipped.")
                        all_dates = pd.date_range(start_date, end_date, freq='D')
                        for d in all_dates:
                            mask = adjusted_data['datetime'] <= d
                            data_till_date = adjusted_data[mask]
                            # Skip if no data for this date
                            if data_till_date.empty or not (data_till_date['datetime'] == d).any():
                                logger.info(f"Skipping {d.strftime('%d-%m-%Y')}: no data for this date")
                                continue
                            analysis_df = perform_technical_analysis(data_till_date, sector_df, logger, enable_candle_patterns=enable_candle_patterns)
                            if not analysis_df.empty:
                                date_str = d.strftime('%d-%m-25')
                                outfile = f"{date_str}snapshot.csv"
                                analysis_df.to_csv(outfile, index=False)
                                print(f'Analysis for {d.strftime("%d-%m-%Y")} complete. Check the {outfile} file for results.')
                                logger.info(f"Saved analysis to {outfile}")
                    except Exception as e:
                        logger.error(f"Invalid date range format: {e}")
                        return
                else:
                    # Single date or blank: original logic
                    if date_input:
                        try:
                            specific_date = datetime.strptime(date_input, '%d-%m-%Y')
                            mask = adjusted_data['datetime'] <= specific_date
                            adjusted_data = adjusted_data[mask]
                        except Exception as e:
                            logger.error(f"Invalid date format: {e}")
                            return

                    if adjusted_data.empty:
                        logger.error("No data found for the specified date(s).")
                        return

                    candle_choice = get_input("Enable candlestick pattern analysis? (y/N): ").strip().lower()
                    enable_candle_patterns = candle_choice in ['y', 'yes']
                    if enable_candle_patterns and not TA_LIB_AVAILABLE:
                        logger.info("TA-Lib is not installed. Candlestick analysis will be skipped.")
                    analysis_df = perform_technical_analysis(adjusted_data, sector_df, logger, enable_candle_patterns=enable_candle_patterns)
                    if not analysis_df.empty:
                        latest_date = pd.to_datetime(analysis_df['date']).max()
                        date_str = latest_date.strftime('%d-%m-25')
                        outfile = f"{date_str}snapshot.csv"
                        analysis_df.to_csv(outfile, index=False)
                        print(f'Analysis complete. Check the {outfile} file for results.')
                        logger.info(f"Saved analysis to {outfile}")
                return
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                return

        else:
            logger.error("Invalid choice")
            return

        if choice in ['1', '2', '3'] and not raw_data.empty:
            raw_data.to_csv(CONFIG['RAW_DATA_FILE'], index=False)
            logger.info(f"Saved raw data to {CONFIG['RAW_DATA_FILE']}")
            adjusted_data, splits = adjust_prices(raw_data, logger)
            if not adjusted_data.empty:
                adjusted_data.to_csv(CONFIG['ADJUSTED_DATA_FILE'], index=False)
                splits.to_csv(CONFIG['SPLITS_LOG_FILE'], index=False)
                logger.info(f"Saved adjusted data to {CONFIG['ADJUSTED_DATA_FILE']} and splits to {CONFIG['SPLITS_LOG_FILE']}")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    logger = setup_logging(verbose=False)
    main()