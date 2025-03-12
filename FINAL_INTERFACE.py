## cd "C:\Users\katai\Documents\University\Year 3\Mathematical Finance Group Projects\Project 3 - Portfolio Management"
## streamlit run FINAL INTERFACE.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import statsmodels.api as sm
import plotly.express as px
import datetime

# Configure Streamlit Page
st.set_page_config(page_title="dualCAPM", layout="wide")

# -----------------------------
# Global Settings and Variables
# -----------------------------
risk_free_rate = 0.02
bond_coupon = 0.0455
bond_duration = 5
bond_return = (1 + bond_coupon) ** (1 / 1) - 1  

# -----------------------------
# Utility Functions
# -----------------------------
@st.cache_data
def get_ftse100_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        tables = pd.read_html(url)
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].tolist()
                break
        else:
            raise ValueError("Could not find the 'Ticker' column in any table.")
        tickers = [ticker + ".L" for ticker in tickers]
        return tickers
    except Exception as e:
        st.error(f"Error fetching FTSE 100 stocks: {e}")
        return []

@st.cache_data
def get_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="10y")
            stock_data[ticker] = {
                "Name": info.get("longName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Market Cap": info.get("marketCap", 0),
                "Beta": info.get("beta", 0),
                "Volatility": np.std(history["Close"].pct_change()) * 100 if not history.empty else 0,
                "Dividend Yield": info.get("dividendYield", 0) * 100,
                "Current Price": info.get("currentPrice", 0),
            }
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {e}")
    df = pd.DataFrame.from_dict(stock_data, orient='index')
    df.fillna(0, inplace=True)
    return df

@st.cache_data
def get_ftse100_historical_data(period="10y", interval="1d"):
    tickers = get_ftse100_tickers()
    all_data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            all_data[ticker] = df
    return all_data

@st.cache_data
def get_tidy_historical_data(tickers, period="10y", interval="1d"):
    df = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df_tidy = df.stack(level=0).rename_axis(['Trading Date', 'Ticker']).reset_index()
    else:
        df_tidy = df.reset_index()
        df_tidy["Ticker"] = tickers[0]
    df_tidy.rename(columns={
        "Open": "Open Price",
        "High": "High Price",
        "Low": "Low Price",
        "Close": "Close Price",
        "Volume": "Trading Volume"
    }, inplace=True)
    df_tidy.sort_values(["Ticker", "Trading Date"], inplace=True)
    return df_tidy

@st.cache_data
def filter_ftse_stocks(df, risk_level):
    if risk_level == "Conservative":
        return df[(df["Market Cap"] >= 26.5e9) &
                  (df["Beta"] <= 1.0) &
                  (df["Volatility"] <= 1.6)]
    elif risk_level == "Moderate":
        return df[(df["Market Cap"] >= 8.7e9) &
                  (df["Beta"] <= 1.2) &
                  (df["Volatility"] <= 1.8)]
    elif risk_level == "Aggressive":
        return df[(df["Market Cap"] >= 5.2e9) &
                  (df["Beta"] > 1.0) & (df["Beta"] <= 2.0) &
                  (df["Volatility"].between(1.5, 3.0))]
    return df

@st.cache_data
def get_stock_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# -----------------------------
# CAPM Functions
# -----------------------------
def CAPM(matching_tickers, start_date, end_date, rolling_window=252):
    # Uses pre_downloaded_data (set in Tab 7) for faster access.
    market_index = '^FTSE'
    all_tickers = matching_tickers + [market_index]
    data = pre_downloaded_data.loc[start_date:end_date, all_tickers]
    data = data.ffill()
    returns = data.pct_change().replace(np.nan, 0)
    
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    market_excess_returns = returns[market_index] - daily_rf
    stock_excess_returns = returns[matching_tickers] - daily_rf
    
    betas = {}
    residual_vars = {}
    for ticker in matching_tickers:
        X = sm.add_constant(market_excess_returns)
        y = stock_excess_returns[ticker]
        if len(X) >= rolling_window:
            X_window = X.iloc[-rolling_window:]
            y_window = y.iloc[-rolling_window:]
            rlm_model = sm.RLM(y_window, X_window, M=sm.robust.norms.HuberT())
            model = rlm_model.fit()
            betas[ticker] = model.params[market_index]
            residual_vars[ticker] = np.var(model.resid, ddof=1)
        else:
            rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            model = rlm_model.fit()
            betas[ticker] = model.params[market_index]
            residual_vars[ticker] = np.var(model.resid, ddof=1)
    
    market_variance = np.var(market_excess_returns, ddof=1) * 252
    n = len(matching_tickers)
    cov_matrix = np.zeros((n, n))
    for i, ticker_i in enumerate(matching_tickers):
        for j, ticker_j in enumerate(matching_tickers):
            if i == j:
                cov_matrix[i, j] = (betas[ticker_i]**2 * market_variance) + (residual_vars[ticker_i] * 252)
            else:
                cov_matrix[i, j] = betas[ticker_i] * betas[ticker_j] * market_variance
    
    avg_daily_market_return = returns[market_index].mean()
    annual_market_return = (1 + avg_daily_market_return)**252 - 1
    expected_returns = {}
    for ticker in matching_tickers:
        expected_returns[ticker] = annual_rf + betas[ticker] * (annual_market_return - annual_rf)
    
    return pd.Series(expected_returns), cov_matrix

def CAPM_direct(matching_tickers, start_date, end_date, rolling_window=252):
    # Downloads data directly from Yahoo Finance.
    market_index = '^FTSE'
    all_tickers = matching_tickers + [market_index]
    data = yf.download(all_tickers, start=start_date, end=end_date, interval="1d")['Close']
    data = data.ffill()
    returns = data.pct_change().replace(np.nan, 0)
    
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    market_excess_returns = returns[market_index] - daily_rf
    stock_excess_returns = returns[matching_tickers] - daily_rf
    
    betas = {}
    residual_vars = {}
    for ticker in matching_tickers:
        X = sm.add_constant(market_excess_returns)
        y = stock_excess_returns[ticker]
        if len(X) >= rolling_window:
            X_window = X.iloc[-rolling_window:]
            y_window = y.iloc[-rolling_window:]
            rlm_model = sm.RLM(y_window, X_window, M=sm.robust.norms.HuberT())
            model = rlm_model.fit()
            betas[ticker] = model.params[market_index]
            residual_vars[ticker] = np.var(model.resid, ddof=1)
        else:
            rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            model = rlm_model.fit()
            betas[ticker] = model.params[market_index]
            residual_vars[ticker] = np.var(model.resid, ddof=1)
    
    market_variance = np.var(market_excess_returns, ddof=1) * 252
    n = len(matching_tickers)
    cov_matrix = np.zeros((n, n))
    for i, ticker_i in enumerate(matching_tickers):
        for j, ticker_j in enumerate(matching_tickers):
            if i == j:
                cov_matrix[i, j] = (betas[ticker_i]**2 * market_variance) + (residual_vars[ticker_i] * 252)
            else:
                cov_matrix[i, j] = betas[ticker_i] * betas[ticker_j] * market_variance
    
    avg_daily_market_return = returns[market_index].mean()
    annual_market_return = (1 + avg_daily_market_return)**252 - 1
    expected_returns = {}
    for ticker in matching_tickers:
        expected_returns[ticker] = annual_rf + betas[ticker] * (annual_market_return - annual_rf)
    
    return pd.Series(expected_returns), cov_matrix

# -----------------------------
# Optimization and CAPM Helper Functions
# -----------------------------
def negative_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return - (portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate):
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    result = sco.minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, covariance_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else None

def portfolio_volatility(weights, covariance_matrix):
    return np.dot(weights, np.dot(covariance_matrix, weights))

def mean_variance_optimization(expected_returns, covariance_matrix, target_return):
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
    )
    result = sco.minimize(
        portfolio_volatility,
        initial_weights,
        args=(covariance_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else None

# # -----------------------------
# # Simulation Function (with Weight History)
# # -----------------------------
# def simulate_rebalanced_portfolio(tickers, years, risk_profile, rebalance_interval=6, initial_investment=10_000_000):
#     # 'end' is the global end date entered by the user (format 'YYYY-MM-DD')
#     end_date = end
#     start_date = f"{int(end_date[:4]) - years}{end_date[4:]}"
#     data = get_stock_prices(tickers, start_date, end_date)
#     portfolio_value_sharpe = initial_investment
#     portfolio_value_mvo = initial_investment
#     portfolio_values_sharpe = []
#     portfolio_values_mvo = []
#     weights_sharpe = np.ones(len(tickers)) / len(tickers)
#     weights_mvo = np.ones(len(tickers)) / len(tickers)
#     weight_history_sharpe = []  # List of tuples: (rebalance_date, {ticker: weight})
#     weight_history_mvo = []
#     rebalance_dates = []
    
#     stock_allocation = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
#     bond_allocation = 1 - stock_allocation
#     bond_return_local = bond_return

#     for i, date in enumerate(data.index):
#         if i % (rebalance_interval * 21) == 0 or i == 0:
#             regression_start_date = date - pd.DateOffset(years=10)
#             # Use the CAPM function (pre-downloaded data)
#             expected_returns, covariance_matrix = CAPM(tickers, regression_start_date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"), rolling_window=252)
#             target_return = expected_returns.quantile(0.75)
#             new_weights_sharpe = optimize_sharpe_ratio(expected_returns, covariance_matrix, 0.02)
#             new_weights_mvo = mean_variance_optimization(expected_returns, covariance_matrix, target_return)
            
#             if new_weights_sharpe is not None:
#                 weights_sharpe = new_weights_sharpe
#                 print(f"Sharpe Optimal Portfolio Weights as of {date}:")
#                 for ticker, weight in zip(tickers, weights_sharpe):
#                     print(f"{ticker}: {weight:.4f}")
#             if new_weights_mvo is not None:
#                 weights_mvo = new_weights_mvo
#                 print(f"MVO Optimal Portfolio Weights as of {date}:")
#                 for ticker, weight in zip(tickers, weights_mvo):
#                     print(f"{ticker}: {weight:.4f}")
#             if new_weights_sharpe is not None or new_weights_mvo is not None:
#                 rebalance_dates.append(date)
#                 weight_history_sharpe.append((date, dict(zip(tickers, weights_sharpe))))
#                 weight_history_mvo.append((date, dict(zip(tickers, weights_mvo))))
        
#         data = data.ffill()
#         daily_returns = data.pct_change().loc[date].fillna(0)
#         stock_return_sharpe = np.dot(weights_sharpe, daily_returns)
#         stock_return_mvo = np.dot(weights_mvo, daily_returns)
#         portfolio_value_sharpe *= (1 + stock_allocation * stock_return_sharpe + bond_allocation * bond_return_local / 252)
#         portfolio_value_mvo *= (1 + stock_allocation * stock_return_mvo + bond_allocation * bond_return_local / 252)
#         portfolio_values_sharpe.append((date, portfolio_value_sharpe))
#         portfolio_values_mvo.append((date, portfolio_value_mvo))
    
#     return portfolio_values_sharpe, portfolio_values_mvo, weight_history_sharpe, weight_history_mvo

# -----------------------------
# Portfolio Breakdown Helper Function
# -----------------------------
def compute_stock_breakdown(last_rebalance_date, last_weights, portfolio_df, risk_profile):
    # Get the last portfolio value on the rebalance date
    V_last = portfolio_df.loc[last_rebalance_date].values[0]
    
    # Define stock allocation based on risk profile
    stock_alloc_percent = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
    
    # Calculate the monetary allocation for each stock based on weights at rebalance date
    allocated_money = {ticker: V_last * stock_alloc_percent * last_weights.get(ticker, 0) for ticker in last_weights}
    
    # Get the prices at the rebalance date and today for the drift calculation
    last_rebalance_str = last_rebalance_date.strftime("%Y-%m-%d")
    today_str = datetime.datetime.today().strftime("%Y-%m-%d")
    
    # Get stock price data for the tickers in last_weights
    price_data = get_stock_prices(list(last_weights.keys()), last_rebalance_str, today_str)
    price_data = price_data.ffill()  # Forward fill to handle missing data

    if price_data.empty:
        return None, None, None

    # Prices at rebalance and current prices
    prices_at_rebalance = price_data.iloc[0]
    prices_today = price_data.iloc[-1]
    
    # Prepare the breakdown data (stock drift and current values)
    breakdown = []
    for ticker in last_weights:
        alloc = allocated_money[ticker]
        drift = prices_today.get(ticker, np.nan) / prices_at_rebalance.get(ticker, np.nan)
        current_value = alloc * drift if pd.notna(drift) else np.nan
        breakdown.append({
            "Ticker": ticker, 
            "Allocation at Rebalance (£)": alloc, 
            "Drift Factor": drift, 
            "Current Stock Value (£)": current_value
        })
    
    # Create a DataFrame with the breakdown information
    breakdown_df = pd.DataFrame(breakdown)
    
    # Calculate the total stock value and bond value
    total_stock_value = breakdown_df["Current Stock Value (£)"].sum()
    current_portfolio_value = portfolio_df.iloc[-1].values[0]
    bond_value = current_portfolio_value - total_stock_value
    
    return breakdown_df, total_stock_value, bond_value


# -----------------------------
# Portfolio Simulation Function
# -----------------------------
def simulate_rebalanced_portfolio(tickers, years, risk_profile, rebalance_interval=6, initial_investment=10_000_000):
    # 'end' is the global end date entered by the user (format 'YYYY-MM-DD')
    end_date = end
    start_date = f"{int(end_date[:4]) - years}{end_date[4:]}"
    data = get_stock_prices(tickers, start_date, end_date)
    portfolio_value_sharpe = initial_investment
    portfolio_value_mvo = initial_investment
    portfolio_values_sharpe = []
    portfolio_values_mvo = []
    weights_sharpe = np.ones(len(tickers)) / len(tickers)
    weights_mvo = np.ones(len(tickers)) / len(tickers)
    weight_history_sharpe = []  # List of tuples: (rebalance_date, {ticker: weight})
    weight_history_mvo = []
    rebalance_dates = []
    
    stock_allocation = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
    bond_allocation = 1 - stock_allocation
    bond_return_local = bond_return

    for i, date in enumerate(data.index):
        if i % (rebalance_interval * 21) == 0 or i == 0:
            regression_start_date = date - pd.DateOffset(years=10)
            # Use the CAPM function (pre-downloaded data)
            expected_returns, covariance_matrix = CAPM(tickers, regression_start_date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"), rolling_window=252)
            target_return = expected_returns.quantile(0.75)
            new_weights_sharpe = optimize_sharpe_ratio(expected_returns, covariance_matrix, 0.02)
            new_weights_mvo = mean_variance_optimization(expected_returns, covariance_matrix, target_return)
            
            if new_weights_sharpe is not None:
                weights_sharpe = new_weights_sharpe
                print(f"Sharpe Optimal Portfolio Weights as of {date}:")
                for ticker, weight in zip(tickers, weights_sharpe):
                    print(f"{ticker}: {weight:.4f}")
            if new_weights_mvo is not None:
                weights_mvo = new_weights_mvo
                print(f"MVO Optimal Portfolio Weights as of {date}:")
                for ticker, weight in zip(tickers, weights_mvo):
                    print(f"{ticker}: {weight:.4f}")
            if new_weights_sharpe is not None or new_weights_mvo is not None:
                rebalance_dates.append(date)
                weight_history_sharpe.append((date, dict(zip(tickers, weights_sharpe))))
                weight_history_mvo.append((date, dict(zip(tickers, weights_mvo))))
        
        data = data.ffill()
        daily_returns = data.pct_change().loc[date].fillna(0)
        stock_return_sharpe = np.dot(weights_sharpe, daily_returns)
        stock_return_mvo = np.dot(weights_mvo, daily_returns)
        portfolio_value_sharpe *= (1 + stock_allocation * stock_return_sharpe + bond_allocation * bond_return_local / 252)
        portfolio_value_mvo *= (1 + stock_allocation * stock_return_mvo + bond_allocation * bond_return_local / 252)
        portfolio_values_sharpe.append((date, portfolio_value_sharpe))
        portfolio_values_mvo.append((date, portfolio_value_mvo))
    
    return portfolio_values_sharpe, portfolio_values_mvo, weight_history_sharpe, weight_history_mvo


# -----------------------------
# Tabbed Navigation
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Stock Screener", 
    "📊 Stock Analysis", 
    "⚖️ Compare Stocks", 
    "🛠️ Risk-Based Stock Selection", 
    "💹 CAPM Estimation",
    "📉 Portfolio Optimization",
    "🔄 Portfolio Rebalancing"
])

# -----------------------------
# Tab 1: Stock Screener
# -----------------------------
with tab1:
    st.header("Stock Screener")
    st.write("Filter stocks based on market cap, revenue, P/E ratio, P/B ratio, dividend yield, and beta.")
    tickers_input = st.text_area("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    if tickers:
        df = get_stock_data(tickers)
        st.subheader("Stock Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Enter tickers above to view stock data.")

# -----------------------------
# Tab 2: Stock Analysis
# -----------------------------
with tab2:
    st.header("Stock Analysis")
    stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):").upper()
    if stock_ticker:
        stock = yf.Ticker(stock_ticker)
        hist = stock.history(period="10y")
        if not hist.empty:
            st.line_chart(hist["Close"])
        else:
            st.warning(f"No data found for {stock_ticker}.")
    else:
        st.info("Please enter a stock ticker to see analysis.")

# -----------------------------
# Tab 3: Compare Stocks
# -----------------------------
with tab3:
    st.header("Stock Comparison")
    ticker1 = st.text_input("Enter first stock ticker (e.g., AAPL):").upper()
    ticker2 = st.text_input("Enter second stock ticker (e.g., MSFT):").upper()
    if ticker1 and ticker2:
        stock1 = yf.Ticker(ticker1).history(period="10y")["Close"]
        stock2 = yf.Ticker(ticker2).history(period="10y")["Close"]
        if not stock1.empty and not stock2.empty:
            st.subheader(f"📊 {ticker1} vs {ticker2} (1-Year Price Comparison)")
            stock_data = pd.DataFrame({ticker1: stock1, ticker2: stock2})
            st.line_chart(stock_data)
        else:
            st.warning("One or both tickers are invalid. Please enter valid stock symbols.")
    else:
        st.info("Enter two stock tickers above to compare their performance.")

# -----------------------------
# Tab 4: Risk-Based Stock Selection
# -----------------------------
with tab4:
    st.header("🛠️ FTSE 100 Risk-Based Stock Selection + Data Gathering")
    
    if st.button("Fetch 10-Year Data for FTSE 100"):
        ftse_data_dict = get_ftse100_historical_data(period="10y", interval="1d")
        if ftse_data_dict:
            st.success(f"✅ Fetched data for {len(ftse_data_dict)} tickers!")
            sample_ticker = list(ftse_data_dict.keys())[0]
            st.subheader(f"Sample data for {sample_ticker}")
            st.dataframe(ftse_data_dict[sample_ticker].head())
        else:
            st.warning("⚠️ No data fetched. Possibly an error occurred.")
    
    if st.button("Download Historical Data for All FTSE 100 Stocks (Last 10 Years)"):
        all_tickers = get_ftse100_tickers()
        tidy_data = get_tidy_historical_data(all_tickers)
        if not tidy_data.empty:
            csv_all = tidy_data.to_csv(index=False)
            st.download_button(
                "Download All FTSE 100 Historical Data CSV",
                data=csv_all,
                file_name="FTSE100_all_historical_data_tidy.csv",
                mime="text/csv"
            )
        else:
            st.warning("No historical data available for all FTSE 100 stocks.")
    
    st.subheader("📥 Download FTSE 100 Stock Metadata")
    if st.button("Download Metadata for All FTSE 100 Stocks"):
        ftse_tickers = get_ftse100_tickers()
        if ftse_tickers:
            df_metadata = get_stock_data(ftse_tickers)
            if not df_metadata.empty:
                df_metadata_filtered = df_metadata[["Market Cap", "Beta", "Dividend Yield", "Current Price", "Volatility"]]
                csv_metadata = df_metadata_filtered.to_csv(index=True)
                st.download_button(
                    "Download FTSE 100 Metadata CSV",
                    data=csv_metadata,
                    file_name="FTSE100_metadata.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No metadata available for FTSE 100 stocks.")
        else:
            st.warning("No FTSE 100 tickers found.")
    
    st.subheader("⚖️ Filter Stocks by Risk Level")
    risk_level = st.selectbox("Select Your Risk Appetite", ["Conservative", "Moderate", "Aggressive"])
    ftse_tickers = get_ftse100_tickers()
    if ftse_tickers:
        df_ftse = get_stock_data(ftse_tickers)
        if not df_ftse.empty:
            filtered_df_ftse = filter_ftse_stocks(df_ftse, risk_level)
            st.subheader(f"✅ Stocks Matching {risk_level} Risk Profile")
            st.dataframe(filtered_df_ftse, use_container_width=True)
            if not filtered_df_ftse.empty:
                st.download_button(
                    "Download Selected Stocks Metadata",
                    data=filtered_df_ftse.to_csv(index=False),
                    file_name=f"FTSE100_{risk_level}_stocks.csv",
                    mime="text/csv"
                )
                if st.button("Download Historical Data for Selected Stocks (Last 10 Years)"):
                    selected_tickers = filtered_df_ftse.index.tolist()
                    combined_hist_data = get_tidy_historical_data(selected_tickers)
                    if not combined_hist_data.empty:
                        st.download_button(
                            "Download Historical Data CSV",
                            data=combined_hist_data.to_csv(index=False),
                            file_name=f"{risk_level}_risk_historical_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No historical data available for selected stocks.")

# -----------------------------
# Tab 5: CAPM Model for Filtered Stocks
# -----------------------------
with tab5:
    st.header("💹 CAPM Model for Filtered Stocks")
    st.write("Run the predefined CAPM function on FTSE 100 stocks filtered by your selected risk profile.")

    risk_appetite = st.selectbox("Select Your Risk Appetite", ["Conservative", "Moderate", "Aggressive"], key="capm_risk_appetite")

    if st.button("Run CAPM Model", key="capm_run_model"):
        ftse_tickers = get_ftse100_tickers()
        if ftse_tickers:
            df_ftse = get_stock_data(ftse_tickers)
            filtered_df = filter_ftse_stocks(df_ftse, risk_appetite)
            if filtered_df.empty:
                st.warning("No stocks match the selected risk filters.")
            else:
                st.subheader(f"✅ Stocks Matching {risk_appetite} Risk Profile")
                st.dataframe(filtered_df[["Name", "Beta", "Market Cap", "Volatility", "Dividend Yield"]], use_container_width=True)
                matching_tickers = filtered_df.index.tolist()
                st.write("Matching Stock Tickers:", ", ".join(matching_tickers))
                
                capm_start_date = st.text_input("Enter CAPM Start Date (YYYY-MM-DD)", "2010-01-01", key="capm_start")
                capm_end_date = st.text_input("Enter CAPM End Date (YYYY-MM-DD)", "2020-01-01", key="capm_end")
                
                st.info("Running CAPM regression analysis on filtered stocks. Please wait...")
                expected_returns_series, cov_matrix = CAPM(matching_tickers, capm_start_date, capm_end_date, rolling_window=252)
                
                st.subheader("📊 CAPM Regression Results")
                st.write("Expected Annual Returns:")
                st.dataframe(expected_returns_series.to_frame(name="Expected Return"))
                st.write("Covariance Matrix (Annualized):")
                st.dataframe(pd.DataFrame(cov_matrix, index=expected_returns_series.index, columns=expected_returns_series.index))
                
                st.subheader("📥 Download CAPM Results")
                csv_expected_return = expected_returns_series.to_csv(index=False)
                st.download_button("📥 Download Expected Returns CSV", data=csv_expected_return, file_name="capm_expected_returns.csv", mime="text/csv")
                csv_cov = pd.DataFrame(cov_matrix, index=expected_returns_series.index, columns=expected_returns_series.index).to_csv(index=True)
                st.download_button("📥 Download Covariance Matrix CSV", data=csv_cov, file_name="capm_cov_matrix.csv", mime="text/csv")

# -----------------------------
# Tab 6: Portfolio Optimization (Optimal Allocation)
# (Uses CAPM_direct – data is downloaded directly from Yahoo Finance)
# -----------------------------
with tab6:
    st.header("📉 Portfolio Optimization (Optimal Allocation)")
    st.write("This tab uses a rolling CAPM approach, updated at yearly intervals (252 trading days), to compute the optimal portfolio allocations based on live data (the last 10 years up to today) using direct Yahoo Finance API.")

    risk_appetite_opt = st.selectbox("Select Your Risk Appetite for Optimization", ["Conservative", "Moderate", "Aggressive"], key="opt_risk")
    
    if risk_appetite_opt == "Conservative":
        stock_allocation = 0.3
    elif risk_appetite_opt == "Moderate":
        stock_allocation = 0.5
    elif risk_appetite_opt == "Aggressive":
        stock_allocation = 0.7
    bond_allocation = 1 - stock_allocation
    
    st.info(f"Based on your risk appetite '{risk_appetite_opt}', your portfolio will allocate £{10_000_000 * stock_allocation:,.0f} in stocks and £{10_000_000 * bond_allocation:,.0f} in bonds.")
    
    ftse_tickers = get_ftse100_tickers()
    df_ftse = get_stock_data(ftse_tickers)
    if df_ftse.empty:
        st.error("No stock data available.")
    else:
        filtered_df = filter_ftse_stocks(df_ftse, risk_appetite_opt)
        if filtered_df.empty:
            st.warning("No stocks match the selected risk appetite.")
        else:
            st.subheader("Stocks Matching the Selected Risk Profile")
            st.dataframe(filtered_df[["Name", "Market Cap", "Beta", "Dividend Yield", "Volatility"]], use_container_width=True)
            
            matching_tickers = filtered_df.index.tolist()
            
            today = datetime.date.today()
            capm_end_date = today.strftime("%Y-%m-%d")
            capm_start_date = (today - datetime.timedelta(days=10*365)).strftime("%Y-%m-%d")
            
            st.write(f"Running CAPM (direct download) over the period from {capm_start_date} to {capm_end_date}.")
            
            if st.button("Run Yearly Rolling CAPM & Portfolio Optimizations"):
                price_data = get_stock_prices(matching_tickers, capm_start_date, capm_end_date)
                price_data = price_data.ffill()
                if price_data.empty:
                    st.error("Unable to fetch price data for the given period.")
                else:
                    interval = 252  # roughly one trading year
                    weight_history_sharpe = []
                    weight_history_mvo = []
                    
                    for i in range(0, len(price_data), interval):
                        current_date = price_data.index[i]
                        regression_start_date = (pd.to_datetime(current_date) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
                        regression_end_date = current_date.strftime("%Y-%m-%d")
                        
                        # Use CAPM_direct here
                        expected_returns_series, cov_matrix = CAPM_direct(matching_tickers, regression_start_date, regression_end_date, rolling_window=252)
                        
                        if expected_returns_series.empty:
                            continue
                        
                        target_return = expected_returns_series.quantile(0.75)
                        
                        weights_sharpe = optimize_sharpe_ratio(expected_returns_series.values, cov_matrix, risk_free_rate=0.02)
                        weights_mvo = mean_variance_optimization(expected_returns_series.values, cov_matrix, target_return)
                        
                        if weights_sharpe is not None:
                            weight_history_sharpe.append((current_date, weights_sharpe))
                        if weights_mvo is not None:
                            weight_history_mvo.append((current_date, weights_mvo))
                    
                    if weight_history_sharpe:
                        latest_date_sharpe, latest_weights_sharpe = weight_history_sharpe[-1]
                    else:
                        st.warning("Sharpe ratio optimization failed for all rebalancing dates.")
                    
                    if weight_history_mvo:
                        latest_date_mvo, latest_weights_mvo = weight_history_mvo[-1]
                    else:
                        st.warning("Mean-variance optimization failed for all rebalancing dates.")
                    
                    st.subheader("Optimal Weights Visualization (Latest Yearly Rolling Window)")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if weight_history_sharpe:
                            sharpe_df = pd.DataFrame({
                                "Ticker": matching_tickers, 
                                "Weight": latest_weights_sharpe
                            })
                            fig_sharpe = px.pie(
                                sharpe_df, 
                                values="Weight", 
                                names="Ticker",
                                title=f"Optimal Weights (Sharpe Ratio Maximization) as of {latest_date_sharpe.strftime('%Y-%m-%d')}",
                                hole=0.15,
                                height=600
                            )
                            st.plotly_chart(fig_sharpe)
                            sharpe_df["Monetary Allocation (£)"] = sharpe_df["Weight"] * (10_000_000 * stock_allocation)
                            st.write("Monetary Allocation (Sharpe Ratio Maximization)")
                            st.dataframe(sharpe_df[["Ticker", "Weight", "Monetary Allocation (£)"]], use_container_width=True)
                        else:
                            st.warning("No Sharpe optimization results to display.")
                    
                    with col2:
                        if weight_history_mvo:
                            mvo_df = pd.DataFrame({
                                "Ticker": matching_tickers, 
                                "Weight": latest_weights_mvo
                            })
                            fig_mvo = px.pie(
                                mvo_df, 
                                values="Weight", 
                                names="Ticker",
                                title=f"Optimal Weights (Mean-Variance Optimization) as of {latest_date_mvo.strftime('%Y-%m-%d')}",
                                hole=0.15,
                                height=600
                            )
                            st.plotly_chart(fig_mvo)
                            mvo_df["Monetary Allocation (£)"] = mvo_df["Weight"] * (10_000_000 * stock_allocation)
                            st.write("Monetary Allocation (Mean-Variance Optimization)")
                            st.dataframe(mvo_df[["Ticker", "Weight", "Monetary Allocation (£)"]], use_container_width=True)
                        else:
                            st.warning("No MVO optimization results to display.")

# -----------------------------
# Tab 7: Portfolio Rebalancing Simulation (Live Data)
# (Uses pre-downloaded data via the original CAPM function)
# -----------------------------
# with tab7:
#     st.header("🔄 Portfolio Rebalancing Simulation (Live Data)")
#     st.write(
#         "Simulate how a rebalanced portfolio would have performed from a specified start date (within the last 5 years) to today. "
#         "This simulation assumes an initial investment of £10M on the start date, with portfolio rebalancing every 6 months."
#     )
    
#     risk_profile_sim = st.selectbox(
#         "Select Your Risk Profile for Simulation", 
#         ["Conservative", "Moderate", "Aggressive"], 
#         key="sim_risk"
#     )
    
#     default_start_date = (datetime.datetime.today() - datetime.timedelta(days=365*8)).strftime("%Y-%m-%d")
#     start_date_sim = st.text_input("Enter Simulation Start Date (YYYY-MM-DD)", default_start_date, key="sim_start")
    
#     rebalance_interval = st.slider("Select Rebalancing Interval (Months)", min_value=1, max_value=12, value=6)
    
#     try:
#         start_date_dt = pd.to_datetime(start_date_sim)
#         today_dt = pd.to_datetime(datetime.datetime.today().strftime("%Y-%m-%d"))
#         start_pre = '2007-01-01'
#         end_pre = datetime.datetime.today().strftime("%Y-%m-%d")
#         all_tickers = ftse_tickers + ['^FTSE']
#         pre_downloaded_data = yf.download(all_tickers, start=start_pre, end=end_pre, interval="1d")['Close'].ffill()
#         if start_date_dt > today_dt:
#             st.error("Start date cannot be in the future.")
#         elif (today_dt - start_date_dt).days > 8*365:
#             st.error("Please choose a start date within the last 8 years.")
#         else:
#             ftse_tickers = get_ftse100_tickers()
#             df_ftse = get_stock_data(ftse_tickers)
#             filtered_df = filter_ftse_stocks(df_ftse, risk_profile_sim)
#             tickers_sim = filtered_df.index.tolist()
            
#             if not tickers_sim:
#                 st.warning("No stocks available for simulation with the selected risk profile.")
#             else:
#                 st.subheader("Stocks Matching the Selected Risk Profile")
#                 st.dataframe(
#                     filtered_df[["Name", "Market Cap", "Beta", "Dividend Yield", "Volatility"]],
#                     use_container_width=True
#                 )
                
#                 @st.cache_data(show_spinner=False)
#                 def run_simulation(tickers, start_date, risk_profile, rebalance_interval, initial_investment=10_000_000):
#                     end_date = datetime.datetime.today().strftime("%Y-%m-%d")
#                     data = get_stock_prices(tickers, start_date, end_date)
#                     if data.empty:
#                         return None, None, None, None
#                     portfolio_value_sharpe = initial_investment
#                     portfolio_value_mvo = initial_investment
#                     portfolio_values_sharpe = []
#                     portfolio_values_mvo = []
#                     weights_sharpe = np.ones(len(tickers)) / len(tickers)
#                     weights_mvo = np.ones(len(tickers)) / len(tickers)
#                     weight_history_sharpe = []  
#                     weight_history_mvo = []
#                     rebalance_dates = []
                    
#                     stock_allocation = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
#                     bond_allocation = 1 - stock_allocation
#                     bond_return_local = bond_return  
                    
#                     interval_days = rebalance_interval * 21
#                     for i, date in enumerate(data.index):
#                         if i % interval_days == 0 or i == 0:
#                             regression_start_date = date - pd.DateOffset(years=10)
#                             expected_returns, covariance_matrix = CAPM(
#                                 tickers, 
#                                 regression_start_date.strftime("%Y-%m-%d"), 
#                                 date.strftime("%Y-%m-%d"), 
#                                 rolling_window=252
#                             )
#                             target_return = expected_returns.quantile(0.75)
#                             new_weights_sharpe = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate=0.02)
#                             new_weights_mvo = mean_variance_optimization(expected_returns, covariance_matrix, target_return)
                            
#                             if new_weights_sharpe is not None:
#                                 weights_sharpe = new_weights_sharpe
#                             if new_weights_mvo is not None:
#                                 weights_mvo = new_weights_mvo
                            
#                             rebalance_dates.append(date)
#                             weight_history_sharpe.append((date, dict(zip(tickers, weights_sharpe))))
#                             weight_history_mvo.append((date, dict(zip(tickers, weights_mvo))))
                        
#                         data = data.ffill()
#                         daily_returns = data.pct_change().loc[date].fillna(0)
#                         stock_return_sharpe = np.dot(weights_sharpe, daily_returns)
#                         stock_return_mvo = np.dot(weights_mvo, daily_returns)
#                         portfolio_value_sharpe *= (1 + stock_allocation * stock_return_sharpe + bond_allocation * bond_return_local / 252)
#                         portfolio_value_mvo *= (1 + stock_allocation * stock_return_mvo + bond_allocation * bond_return_local / 252)
#                         portfolio_values_sharpe.append((date, portfolio_value_sharpe))
#                         portfolio_values_mvo.append((date, portfolio_value_mvo))
                    
#                     return portfolio_values_sharpe, portfolio_values_mvo, weight_history_sharpe, weight_history_mvo
                
#                 pv_sharpe, pv_mvo, weight_hist_sharpe, weight_hist_mvo = run_simulation(
#                     tickers_sim, start_date_sim, risk_profile_sim, rebalance_interval, 10_000_000
#                 )
                
#                 if pv_sharpe is None or pv_mvo is None:
#                     st.error("Simulation failed due to insufficient data.")
#                 else:
#                     df_pv_sharpe = pd.DataFrame(pv_sharpe, columns=["Date", "Sharpe Portfolio Value"]).set_index("Date")
#                     df_pv_mvo = pd.DataFrame(pv_mvo, columns=["Date", "MVO Portfolio Value"]).set_index("Date")
                    
#                     initial_investment = 10_000_000
#                     current_sharpe = df_pv_sharpe.iloc[-1, 0]
#                     current_mvo = df_pv_mvo.iloc[-1, 0]
#                     gain_pct_sharpe = ((current_sharpe - initial_investment) / initial_investment) * 100
#                     gain_abs_sharpe = current_sharpe - initial_investment
#                     gain_pct_mvo = ((current_mvo - initial_investment) / initial_investment) * 100
#                     gain_abs_mvo = current_mvo - initial_investment
                    
#                     st.subheader("Current Portfolio Metrics")
#                     col_metrics1, col_metrics2 = st.columns(2)
#                     with col_metrics1:
#                         st.markdown("**Sharpe Optimized Portfolio**")
#                         st.write(f"Current Value: £{current_sharpe:,.0f}")
#                         st.write(f"Gain/Loss: {gain_pct_sharpe:+.2f}% (Absolute: £{gain_abs_sharpe:+,.0f})")
#                     with col_metrics2:
#                         st.markdown("**Minimum Variance Portfolio**")
#                         st.write(f"Current Value: £{current_mvo:,.0f}")
#                         st.write(f"Gain/Loss: {gain_pct_mvo:+.2f}% (Absolute: £{gain_abs_mvo:+,.0f})")
                    
#                     benchmark_data = get_stock_prices(['^FTSE'], start_date_sim, datetime.datetime.today().strftime("%Y-%m-%d"))
#                     benchmark_returns = benchmark_data.pct_change().dropna()
#                     benchmark_value = initial_investment * (1 + benchmark_returns).cumprod()
#                     benchmark_value.name = "FTSE 100"
                    
#                     combined_df = pd.concat([
#                         df_pv_sharpe.rename(columns={"Sharpe Portfolio Value": "Sharpe Portfolio"}),
#                         df_pv_mvo.rename(columns={"MVO Portfolio Value": "MVO Portfolio"}),
#                         benchmark_value
#                     ], axis=1)
                    
#                     st.subheader("Interactive Portfolio Performance")
#                     fig_line = px.line(
#                         combined_df,
#                         x=combined_df.index,
#                         y=combined_df.columns,
#                         title="Portfolio Performance Over Time"
#                     )
#                     fig_line.update_layout(
#                         xaxis_title="Date",
#                         yaxis_title="Portfolio Value (£)",
#                         hovermode="x unified",
#                         height=600
#                     )
#                     y_min = combined_df.min().min()
#                     y_max = combined_df.max().max()
#                     fig_line.update_yaxes(range=[y_min * 0.95, y_max * 1.05])
#                     st.plotly_chart(fig_line, use_container_width=True)
                    
#                     st.subheader("Historical Rebalance Weightings")
#                     col_hist_sharpe, col_hist_mvo = st.columns(2)
#                     with col_hist_sharpe:
#                         st.markdown("### Sharpe Optimization History")
#                         selected_date_sharpe = st.selectbox(
#                             "Select a Sharpe rebalance date",
#                             [date.strftime("%Y-%m-%d") for date, _ in weight_hist_sharpe],
#                             key="select_sharpe"
#                         )
#                         for date, weights in weight_hist_sharpe:
#                             if date.strftime("%Y-%m-%d") == selected_date_sharpe:
#                                 pv_at_date_value = df_pv_sharpe.loc[date, "Sharpe Portfolio Value"]
#                                 monetary_alloc = {
#                                     ticker: weight * (pv_at_date_value * {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile_sim])
#                                     for ticker, weight in weights.items()
#                                 }
#                                 df_weights = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
#                                 df_weights["Monetary Allocation (£)"] = df_weights["Ticker"].map(monetary_alloc)
#                                 fig_pie = px.pie(
#                                     df_weights,
#                                     values="Weight",
#                                     names="Ticker",
#                                     title=f"Sharpe Weights on {selected_date_sharpe}",
#                                     hole=0.3,
#                                     height=600
#                                 )
#                                 st.plotly_chart(fig_pie, use_container_width=True)
#                                 st.dataframe(df_weights, use_container_width=True)
#                                 break
#                     with col_hist_mvo:
#                         st.markdown("### MVO Optimization History")
#                         selected_date_mvo = st.selectbox(
#                             "Select an MVO rebalance date",
#                             [date.strftime("%Y-%m-%d") for date, _ in weight_hist_mvo],
#                             key="select_mvo"
#                         )
#                         for date, weights in weight_hist_mvo:
#                             if date.strftime("%Y-%m-%d") == selected_date_mvo:
#                                 pv_at_date_value = df_pv_mvo.loc[date, "MVO Portfolio Value"]
#                                 monetary_alloc_mvo = {
#                                     ticker: weight * (pv_at_date_value * {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile_sim])
#                                     for ticker, weight in weights.items()
#                                 }
#                                 df_weights_mvo = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
#                                 df_weights_mvo["Monetary Allocation (£)"] = df_weights_mvo["Ticker"].map(monetary_alloc_mvo)
#                                 fig_pie_mvo = px.pie(
#                                     df_weights_mvo,
#                                     values="Weight",
#                                     names="Ticker",
#                                     title=f"MVO Weights on {selected_date_mvo}",
#                                     hole=0.3,
#                                     height=600
#                                 )
#                                 st.plotly_chart(fig_pie_mvo, use_container_width=True)
#                                 st.dataframe(df_weights_mvo, use_container_width=True)
#                                 break
                    
#                     st.subheader("Detailed Portfolio Breakdown: Current Stock Allocations & Bonds")
                    
#                     def compute_stock_breakdown(last_rebalance_date, last_weights, portfolio_df, risk_profile):
#                         V_last = portfolio_df.loc[last_rebalance_date].values[0]
#                         stock_alloc_percent = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
#                         allocated_money = {ticker: V_last * stock_alloc_percent * last_weights.get(ticker, 0) for ticker in last_weights}
                        
#                         last_rebalance_str = last_rebalance_date.strftime("%Y-%m-%d")
#                         today_str = datetime.datetime.today().strftime("%Y-%m-%d")
#                         price_data = get_stock_prices(list(last_weights.keys()), last_rebalance_str, today_str)
#                         price_data = price_data.ffill()
#                         if price_data.empty:
#                             return None, None, None
#                         prices_at_rebalance = price_data.iloc[0]
#                         prices_today = price_data.iloc[-1]
#                         breakdown = []
#                         for ticker in last_weights:
#                             alloc = allocated_money[ticker]
#                             drift = prices_today.get(ticker, np.nan) / prices_at_rebalance.get(ticker, np.nan)
#                             current_value = alloc * drift if pd.notna(drift) else np.nan
#                             breakdown.append({
#                                 "Ticker": ticker, 
#                                 "Allocation at Rebalance (£)": alloc, 
#                                 "Drift Factor": drift, 
#                                 "Current Stock Value (£)": current_value
#                             })
#                         breakdown_df = pd.DataFrame(breakdown)
#                         total_stock_value = breakdown_df["Current Stock Value (£)"].sum()
#                         current_portfolio_value = portfolio_df.iloc[-1].values[0]
#                         bond_value = current_portfolio_value - total_stock_value
#                         return breakdown_df, total_stock_value, bond_value
                    
#                     if weight_hist_sharpe:
#                         last_rebalance_date_sharpe, last_weights_sharpe = weight_hist_sharpe[-1]
#                         breakdown_sharpe, total_stock_value_sharpe, bond_value_sharpe = compute_stock_breakdown(
#                             last_rebalance_date_sharpe, last_weights_sharpe, df_pv_sharpe, risk_profile_sim
#                         )
#                         st.markdown("### Sharpe Optimized Portfolio Breakdown")
#                         if breakdown_sharpe is not None:
#                             st.dataframe(breakdown_sharpe)
#                             st.write(f"**Total Stock Value:** £{total_stock_value_sharpe:,.0f}")
#                             st.write(f"**Bond Value (Remaining):** £{bond_value_sharpe:,.0f}")
#                         else:
#                             st.warning("Price data unavailable for detailed breakdown (Sharpe).")
                    
#                     if weight_hist_mvo:
#                         last_rebalance_date_mvo, last_weights_mvo = weight_hist_mvo[-1]
#                         breakdown_mvo, total_stock_value_mvo, bond_value_mvo = compute_stock_breakdown(
#                             last_rebalance_date_mvo, last_weights_mvo, df_pv_mvo, risk_profile_sim
#                         )
#                         st.markdown("### MVO Optimized Portfolio Breakdown")
#                         if breakdown_mvo is not None:
#                             st.dataframe(breakdown_mvo)
#                             st.write(f"**Total Stock Value:** £{total_stock_value_mvo:,.0f}")
#                             st.write(f"**Bond Value (Remaining):** £{bond_value_mvo:,.0f}")
#                         else:
#                             st.warning("Price data unavailable for detailed breakdown (MVO).")
#     except Exception as e:
#         st.error(f"Error: {e}")

with tab7:
    st.header("🔄 Portfolio Rebalancing Simulation (Live Data)")
    st.write(
        "Simulate how a rebalanced portfolio would have performed from a specified start date (within the last 5 years) to today. "
        "This simulation assumes an initial investment of £10M on the start date, with portfolio rebalancing every 6 months."
    )
    
    risk_profile_sim = st.selectbox(
        "Select Your Risk Profile for Simulation", 
        ["Conservative", "Moderate", "Aggressive"], 
        key="sim_risk"
    )
    
    default_start_date = (datetime.datetime.today() - datetime.timedelta(days=365*8)).strftime("%Y-%m-%d")
    start_date_sim = st.text_input("Enter Simulation Start Date (YYYY-MM-DD)", default_start_date, key="sim_start")
    
    rebalance_interval = st.slider("Select Rebalancing Interval (Months)", min_value=1, max_value=12, value=6)
    
    try:
        start_date_dt = pd.to_datetime(start_date_sim)
        today_dt = pd.to_datetime(datetime.datetime.today().strftime("%Y-%m-%d"))
        start_pre = '2007-01-01'
        end_pre = datetime.datetime.today().strftime("%Y-%m-%d")
        all_tickers = get_ftse100_tickers() + ['^FTSE']
        pre_downloaded_data = yf.download(all_tickers, start=start_pre, end=end_pre, interval="1d")['Close'].ffill()
        if start_date_dt > today_dt:
            st.error("Start date cannot be in the future.")
        elif (today_dt - start_date_dt).days > 8*365:
            st.error("Please choose a start date within the last 8 years.")
        else:
            ftse_tickers = get_ftse100_tickers()
            df_ftse = get_stock_data(ftse_tickers)
            filtered_df = filter_ftse_stocks(df_ftse, risk_profile_sim)
            tickers_sim = filtered_df.index.tolist()
            
            if not tickers_sim:
                st.warning("No stocks available for simulation with the selected risk profile.")
            else:
                st.subheader("Stocks Matching the Selected Risk Profile")
                st.dataframe(
                    filtered_df[["Name", "Market Cap", "Beta", "Dividend Yield", "Volatility"]],
                    use_container_width=True
                )
                
                @st.cache_data(show_spinner=False)
                def run_simulation(tickers, start_date, risk_profile, rebalance_interval, initial_investment=10_000_000):
                    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
                    data = get_stock_prices(tickers, start_date, end_date)
                    if data.empty:
                        return None, None, None, None
                    portfolio_value_sharpe = initial_investment
                    portfolio_value_mvo = initial_investment
                    portfolio_values_sharpe = []
                    portfolio_values_mvo = []
                    weights_sharpe = np.ones(len(tickers)) / len(tickers)
                    weights_mvo = np.ones(len(tickers)) / len(tickers)
                    weight_history_sharpe = []  
                    weight_history_mvo = []
                    rebalance_dates = []
                    
                    stock_allocation = {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile]
                    bond_allocation = 1 - stock_allocation
                    bond_return_local = bond_return  
                    
                    interval_days = rebalance_interval * 21
                    for i, date in enumerate(data.index):
                        if i % interval_days == 0 or i == 0:
                            regression_start_date = date - pd.DateOffset(years=10)
                            expected_returns, covariance_matrix = CAPM(
                                tickers, 
                                regression_start_date.strftime("%Y-%m-%d"), 
                                date.strftime("%Y-%m-%d"), 
                                rolling_window=252
                            )
                            target_return = expected_returns.quantile(0.75)
                            new_weights_sharpe = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate=0.02)
                            new_weights_mvo = mean_variance_optimization(expected_returns, covariance_matrix, target_return)
                            
                            if new_weights_sharpe is not None:
                                weights_sharpe = new_weights_sharpe
                            if new_weights_mvo is not None:
                                weights_mvo = new_weights_mvo
                            
                            rebalance_dates.append(date)
                            weight_history_sharpe.append((date, dict(zip(tickers, weights_sharpe))))
                            weight_history_mvo.append((date, dict(zip(tickers, weights_mvo))))
                        
                        data = data.ffill()
                        daily_returns = data.pct_change().loc[date].fillna(0)
                        stock_return_sharpe = np.dot(weights_sharpe, daily_returns)
                        stock_return_mvo = np.dot(weights_mvo, daily_returns)
                        portfolio_value_sharpe *= (1 + stock_allocation * stock_return_sharpe + bond_allocation * bond_return_local / 252)
                        portfolio_value_mvo *= (1 + stock_allocation * stock_return_mvo + bond_allocation * bond_return_local / 252)
                        portfolio_values_sharpe.append((date, portfolio_value_sharpe))
                        portfolio_values_mvo.append((date, portfolio_value_mvo))
                    
                    return portfolio_values_sharpe, portfolio_values_mvo, weight_history_sharpe, weight_history_mvo
                
                pv_sharpe, pv_mvo, weight_hist_sharpe, weight_hist_mvo = run_simulation(
                    tickers_sim, start_date_sim, risk_profile_sim, rebalance_interval, 10_000_000
                )
                
                if pv_sharpe is None or pv_mvo is None:
                    st.error("Simulation failed due to insufficient data.")
                else:
                    df_pv_sharpe = pd.DataFrame(pv_sharpe, columns=["Date", "Sharpe Portfolio Value"]).set_index("Date")
                    df_pv_mvo = pd.DataFrame(pv_mvo, columns=["Date", "MVO Portfolio Value"]).set_index("Date")
                    
                    initial_investment = 10_000_000
                    current_sharpe = df_pv_sharpe.iloc[-1, 0]
                    current_mvo = df_pv_mvo.iloc[-1, 0]
                    gain_pct_sharpe = ((current_sharpe - initial_investment) / initial_investment) * 100
                    gain_abs_sharpe = current_sharpe - initial_investment
                    gain_pct_mvo = ((current_mvo - initial_investment) / initial_investment) * 100
                    gain_abs_mvo = current_mvo - initial_investment
                    
                    st.subheader("Current Portfolio Metrics")
                    col_metrics1, col_metrics2 = st.columns(2)
                    with col_metrics1:
                        st.markdown("**Sharpe Optimized Portfolio**")
                        st.write(f"Current Value: £{current_sharpe:,.0f}")
                        st.write(f"Gain/Loss: {gain_pct_sharpe:+.2f}% (Absolute: £{gain_abs_sharpe:+,.0f})")
                    with col_metrics2:
                        st.markdown("**Minimum Variance Portfolio**")
                        st.write(f"Current Value: £{current_mvo:,.0f}")
                        st.write(f"Gain/Loss: {gain_pct_mvo:+.2f}% (Absolute: £{gain_abs_mvo:+,.0f})")
                    
                    benchmark_data = get_stock_prices(['^FTSE'], start_date_sim, datetime.datetime.today().strftime("%Y-%m-%d"))
                    benchmark_returns = benchmark_data.pct_change().dropna()
                    benchmark_value = initial_investment * (1 + benchmark_returns).cumprod()
                    benchmark_value.name = "FTSE 100"
                    
                    combined_df = pd.concat([
                        df_pv_sharpe.rename(columns={"Sharpe Portfolio Value": "Sharpe Portfolio"}),
                        df_pv_mvo.rename(columns={"MVO Portfolio Value": "MVO Portfolio"}),
                        benchmark_value
                    ], axis=1)
                    
                    st.subheader("Interactive Portfolio Performance")
                    fig_line = px.line(
                        combined_df,
                        x=combined_df.index,
                        y=combined_df.columns,
                        title="Portfolio Performance Over Time"
                    )
                    fig_line.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value (£)",
                        hovermode="x unified",
                        height=600
                    )
                    y_min = combined_df.min().min()
                    y_max = combined_df.max().max()
                    fig_line.update_yaxes(range=[y_min * 0.95, y_max * 1.05])
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    ############ NEW SECTIONS BELOW ############
                    # 1. Detailed Portfolio Breakdown: Current Stock Allocations & Bonds
                    st.subheader("Detailed Portfolio Breakdown: Current Stock Allocations & Bonds")
                    # Assume compute_stock_breakdown is defined elsewhere (it returns breakdown_df, total_stock_value, bond_value)
                    col_breakdown1, col_breakdown2 = st.columns(2)
                    with col_breakdown1:
                        st.markdown("#### Sharpe Optimized Breakdown")
                        if weight_hist_sharpe:
                            last_rebalance_date_sharpe, last_weights_sharpe = weight_hist_sharpe[-1]
                            breakdown_sharpe, total_stock_value_sharpe, bond_value_sharpe = compute_stock_breakdown(
                                last_rebalance_date_sharpe, last_weights_sharpe, df_pv_sharpe, risk_profile_sim
                            )
                            if breakdown_sharpe is not None:
                                st.dataframe(breakdown_sharpe, use_container_width=True)
                                # Preserve the original pie chart for stock breakdown:
                                fig_breakdown_sharpe = px.pie(
                                    breakdown_sharpe,
                                    values="Current Stock Value (£)",
                                    names="Ticker",
                                    title="Current Stock Allocation (Sharpe)",
                                    hole=0.3,
                                    height=600
                                )
                                st.plotly_chart(fig_breakdown_sharpe, use_container_width=True)
                                st.write(f"**Bond Value (Remaining):** £{bond_value_sharpe:,.0f}")
                            else:
                                st.warning("Price data unavailable for detailed breakdown (Sharpe).")
                    with col_breakdown2:
                        st.markdown("#### MVO Optimized Breakdown")
                        if weight_hist_mvo:
                            last_rebalance_date_mvo, last_weights_mvo = weight_hist_mvo[-1]
                            breakdown_mvo, total_stock_value_mvo, bond_value_mvo = compute_stock_breakdown(
                                last_rebalance_date_mvo, last_weights_mvo, df_pv_mvo, risk_profile_sim
                            )
                            if breakdown_mvo is not None:
                                st.dataframe(breakdown_mvo, use_container_width=True)
                                fig_breakdown_mvo = px.pie(
                                    breakdown_mvo,
                                    values="Current Stock Value (£)",
                                    names="Ticker",
                                    title="Current Stock Allocation (MVO)",
                                    hole=0.3,
                                    height=600
                                )
                                st.plotly_chart(fig_breakdown_mvo, use_container_width=True)
                                st.write(f"**Bond Value (Remaining):** £{bond_value_mvo:,.0f}")
                            else:
                                st.warning("Price data unavailable for detailed breakdown (MVO).")
                    
                    # 2. Current Overall Portfolio Allocation (Stock vs Bond) per Optimization Technique
                    st.subheader("Current Overall Portfolio Allocation (Stock vs Bond) per Optimization Technique")
                    col_alloc1, col_alloc2 = st.columns(2)
                    with col_alloc1:
                        st.markdown("#### Sharpe Portfolio Allocation")
                        if weight_hist_sharpe and breakdown_sharpe is not None:
                            overall_sharpe_df = pd.DataFrame({
                                "Asset": ["Stocks", "Bonds"],
                                "Value": [total_stock_value_sharpe, bond_value_sharpe]
                            })
                            fig_overall_sharpe = px.pie(
                                overall_sharpe_df,
                                values="Value",
                                names="Asset",
                                title=f"Overall Allocation (Sharpe) as of {today_dt.strftime('%Y-%m-%d')}",
                                hole=0.3,
                                height=600
                            )
                            st.plotly_chart(fig_overall_sharpe, use_container_width=True)
                            st.dataframe(overall_sharpe_df, use_container_width=True)
                        else:
                            st.warning("Detailed breakdown not available for Sharpe overall allocation.")
                    with col_alloc2:
                        st.markdown("#### MVO Portfolio Allocation")
                        if weight_hist_mvo and breakdown_mvo is not None:
                            overall_mvo_df = pd.DataFrame({
                                "Asset": ["Stocks", "Bonds"],
                                "Value": [total_stock_value_mvo, bond_value_mvo]
                            })
                            fig_overall_mvo = px.pie(
                                overall_mvo_df,
                                values="Value",
                                names="Asset",
                                title=f"Overall Allocation (MVO) as of {today_dt.strftime('%Y-%m-%d')}",
                                hole=0.3,
                                height=600
                            )
                            st.plotly_chart(fig_overall_mvo, use_container_width=True)
                            st.dataframe(overall_mvo_df, use_container_width=True)
                        else:
                            st.warning("Detailed breakdown not available for MVO overall allocation.")
                    
                    # 3. Historical Rebalance Weightings (displayed after the above sections)
                    st.subheader("Historical Rebalance Weightings")
                    col_hist_sharpe, col_hist_mvo = st.columns(2)
                    with col_hist_sharpe:
                        st.markdown("### Sharpe Optimization History")
                        selected_date_sharpe = st.selectbox(
                            "Select a Sharpe rebalance date",
                            [date.strftime("%Y-%m-%d") for date, _ in weight_hist_sharpe],
                            key="select_sharpe"
                        )
                        for date, weights in weight_hist_sharpe:
                            if date.strftime("%Y-%m-%d") == selected_date_sharpe:
                                pv_at_date_value = df_pv_sharpe.loc[date, "Sharpe Portfolio Value"]
                                monetary_alloc = {
                                    ticker: weight * (pv_at_date_value * {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile_sim])
                                    for ticker, weight in weights.items()
                                }
                                df_weights = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
                                df_weights["Monetary Allocation (£)"] = df_weights["Ticker"].map(monetary_alloc)
                                fig_pie = px.pie(
                                    df_weights,
                                    values="Weight",
                                    names="Ticker",
                                    title=f"Sharpe Weights on {selected_date_sharpe}",
                                    hole=0.3,
                                    height=600
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                                st.dataframe(df_weights, use_container_width=True)
                                break
                    with col_hist_mvo:
                        st.markdown("### MVO Optimization History")
                        selected_date_mvo = st.selectbox(
                            "Select an MVO rebalance date",
                            [date.strftime("%Y-%m-%d") for date, _ in weight_hist_mvo],
                            key="select_mvo"
                        )
                        for date, weights in weight_hist_mvo:
                            if date.strftime("%Y-%m-%d") == selected_date_mvo:
                                pv_at_date_value = df_pv_mvo.loc[date, "MVO Portfolio Value"]
                                monetary_alloc_mvo = {
                                    ticker: weight * (pv_at_date_value * {'Conservative': 0.3, 'Moderate': 0.5, 'Aggressive': 0.7}[risk_profile_sim])
                                    for ticker, weight in weights.items()
                                }
                                df_weights_mvo = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
                                df_weights_mvo["Monetary Allocation (£)"] = df_weights_mvo["Ticker"].map(monetary_alloc_mvo)
                                fig_pie_mvo = px.pie(
                                    df_weights_mvo,
                                    values="Weight",
                                    names="Ticker",
                                    title=f"MVO Weights on {selected_date_mvo}",
                                    hole=0.3,
                                    height=600
                                )
                                st.plotly_chart(fig_pie_mvo, use_container_width=True)
                                st.dataframe(df_weights_mvo, use_container_width=True)
                                break
    except Exception as e:
        st.error(f"Error: {e}")

