import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import sys, os
from loguru import logger
import shap

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.feature_pipeline import build_features
from export_production_v2_intraday import build_intraday_features

def optimize_asset(symbol, ticker, timeframe):
    logger.info(f"Optimizing {symbol} ({timeframe})...")
    period = "730d" if timeframe == "1h" else "5y"
    df = yf.download(ticker, period=period, interval=timeframe, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
    elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns: df[col] = 0

    if timeframe == "1h":
        df_feat = build_intraday_features(df)
        buy_t, sell_t = 0.65, 0.35
    else:
        df_feat = build_features(df)
        buy_t, sell_t = 0.60, 0.40

    drop_cols = ["id", "asset_id", "timeframe", "timestamp", "target", "source", "symbol", "date", "datetime", "created_at"]
    features = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[features]
    y = df_feat["target"]

    split = int(len(df_feat) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    # Use robust params for final prod
    model = lgb.LGBMClassifier(num_leaves=15, max_depth=4, learning_rate=0.01, n_estimators=200, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # SHAP Feature Importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    vals = np.abs(shap_values).mean(0)
    feat_imp = pd.DataFrame(list(zip(features, vals)), columns=['Feature', 'Importance'])
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(5)
    top_features = feat_imp['Feature'].tolist()

    prob = model.predict_proba(X_test)[:, 1]
    test_df = df_feat.iloc[split:].copy()
    test_df['signal'] = 0
    test_df.loc[prob > buy_t, 'signal'] = 1
    test_df.loc[prob < sell_t, 'signal'] = -1

    shift_val = 12 if timeframe == "1h" else 5
    test_df['future_ret'] = (test_df['close'].shift(-shift_val) - test_df['close']) / test_df['close']
    test_df['strat_ret'] = test_df['signal'].shift(1) * test_df['future_ret']
    
    cost = 0.00015
    test_df['strat_ret_net'] = np.where(test_df['signal'].shift(1) != 0, test_df['strat_ret'] - cost, 0)

    time_col = 'datetime' if 'datetime' in test_df.columns else 'date' if 'date' in test_df.columns else test_df.columns[0]
    test_df['timestamp_col'] = pd.to_datetime(test_df[time_col], utc=True)
    portfolio_series = test_df[['timestamp_col', 'strat_ret_net']].set_index('timestamp_col')
    portfolio_series.columns = [symbol]

    return portfolio_series, top_features

if __name__ == "__main__":
    logger.info("=== FINAL PORTFOLIO OPTIMIZATION & SUMMARY ===")
    
    eur_series, eur_feats = optimize_asset("EURUSD", "EURUSD=X", "1h")
    chf_series, chf_feats = optimize_asset("USDCHF", "CHF=X", "1d")

    # Combine into Portfolio
    # Merge on date for daily approximation
    eur_daily = eur_series.resample('D').sum().dropna()
    chf_daily = chf_series.resample('D').sum().dropna()
    
    portfolio = pd.concat([eur_daily, chf_daily], axis=1).fillna(0)
    # Equal Weight Allocation (50% EURUSD, 50% USDCHF)
    portfolio['Port_Return'] = (portfolio['EURUSD'] * 0.5) + (portfolio['USDCHF'] * 0.5)
    
    # Portfolio Metrics
    port_mean = portfolio['Port_Return'].mean()
    port_std = portfolio['Port_Return'].std()
    port_sharpe = np.sqrt(252) * (port_mean / port_std) if port_std > 0 else 0
    
    port_cum = np.exp(portfolio['Port_Return'].cumsum())
    port_dd = ((port_cum - port_cum.cummax()) / port_cum.cummax()).min()
    port_net = port_cum.iloc[-1] - 1
    
    logger.info("\n=== 🏆 FINAL APPROVED PORTFOLIO (EURUSD 1H + USDCHF 1D) ===")
    logger.info(f"Combined Portfolio Sharpe Ratio: {port_sharpe:.2f}")
    logger.info(f"Combined Portfolio Max Drawdown: {port_dd*100:.2f}%")
    logger.info(f"Combined Portfolio Net Return  : {port_net*100:.2f}%")
    
    logger.info(f"\n[Explainability] Top 5 Driving Features for EURUSD 1H: {eur_feats}")
    logger.info(f"[Explainability] Top 5 Driving Features for USDCHF 1D: {chf_feats}")
