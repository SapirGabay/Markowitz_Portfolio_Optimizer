# =========================================================================
# Markowitz Portfolio Optimizer - Streamlit App (Stable & Optimized)
# Developed by Sapir Gabay | Industrial Engineering & Intelligent Systems
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from numpy.linalg import LinAlgError

# =========================================================================
# DISCLAIMER
# =========================================================================
# This application is for academic and demonstrative purposes only.
# It does NOT constitute financial advice or investment recommendations.
# =========================================================================

ANNUALIZATION_FACTOR = 12  # Monthly → Annual


# ======================= CORE MARKOWITZ FUNCTIONS ========================

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate annualized portfolio volatility (std_dev) and return (returns).
    mean_returns and cov_matrix are monthly; we annualize.
    """
    weights = np.asarray(weights, dtype=float)

    # Monthly portfolio mean return
    monthly_ret = np.dot(weights, mean_returns.values)
    # Annualized return
    returns = monthly_ret * ANNUALIZATION_FACTOR

    # Monthly variance
    monthly_var = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    # Annualized std dev
    std_dev = np.sqrt(monthly_var * ANNUALIZATION_FACTOR)

    return std_dev, returns


def minimize_volatility(mean_returns, cov_matrix, constraints, num_assets):
    """Find Global Minimum Variance Portfolio (GMVP) via SLSQP."""

    initial_weights = np.array(num_assets * [1.0 / num_assets], dtype=float)

    def objective_function(weights):
        std_dev, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        return std_dev

    result = minimize(
        objective_function,
        initial_weights,
        method="SLSQP",
        bounds=tuple([(0, 1)] * num_assets),  # long-only, 0–100%
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x


def generate_random_portfolios(mean_returns, cov_matrix, num_assets, num_portfolios=10000):
    """
    Vectorized random portfolio simulation.
    Returns array: [std_devs, annual_returns, sharpe_ratios]
    """
    rng = np.random.default_rng()
    weights = rng.random((num_portfolios, num_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    mean_vec = mean_returns.values  # monthly means
    cov_mat = cov_matrix.values     # monthly covariance

    # Monthly mean returns for each portfolio
    monthly_returns = weights @ mean_vec
    annual_returns = monthly_returns * ANNUALIZATION_FACTOR

    # Monthly variances via einsum
    monthly_vars = np.einsum("ij,jk,ik->i", weights, cov_mat, weights)
    std_devs = np.sqrt(monthly_vars * ANNUALIZATION_FACTOR)

    # Sharpe (rf = 0), handle std_dev = 0
    sharpe_ratios = np.divide(
        annual_returns,
        std_devs,
        out=np.zeros_like(annual_returns),
        where=std_devs > 0,
    )

    return np.vstack((std_devs, annual_returns, sharpe_ratios))


# =========================== DATA HANDLING ===============================

@st.cache_data
def get_data(tickers, start_date, end_date):
    """
    Download monthly price data from yfinance, resample to month-end,
    and clean NaNs.
    """
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )

    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # Multi-ticker: MultiIndex columns ('Adj Close', TICKER)
    if "Adj Close" in raw_data.columns and len(tickers) > 1:
        price_data = raw_data["Adj Close"]
    # Single ticker: make sure column is the ticker name
    elif "Adj Close" in raw_data.columns and len(tickers) == 1:
        price_data = raw_data[["Adj Close"]].copy()
        price_data.columns = tickers
    else:
        raise ValueError("Data retrieval error: 'Adj Close' not found in downloaded data.")

    # Forward-fill, then resample to month-end
    price_data = price_data.ffill()
    price_data_monthly = price_data.resample("ME").last()

    # Drop empty columns and rows with any NaNs
    price_data_monthly = price_data_monthly.dropna(axis=1, how="all").dropna(axis=0, how="any")

    return price_data_monthly


# ============================ STREAMLIT UI ===============================

st.set_page_config(layout="wide")

st.title("🛡️ Markowitz Portfolio Optimizer: Risk Minimization Model")
st.markdown("---")
st.markdown(
    """
    This app implements core **Markowitz Portfolio Theory** to compute the **Efficient Frontier**
    and the **Global Minimum Variance Portfolio (GMVP)** based on historical data.
    """
)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("1. Asset Selection")

ticker_input = st.sidebar.text_area(
    "Enter Stock Tickers, separated by commas (e.g., AAPL, MSFT, GOOG, JPM)",
    "GLD, MSFT",
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
num_assets = len(tickers)

if st.sidebar.button("Run Optimization"):
    # Basic validations
    if num_assets < 2:
        st.error("Please enter at least two stock tickers to perform portfolio optimization.")
    elif start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    else:
        try:
            with st.spinner("Pulling historical data and running simulations..."):
                # ----- DATA -----
                data = get_data(tickers, start_date, end_date)

                if data.empty or data.shape[1] < 2:
                    st.error(
                        "Error: Could not retrieve data for at least two common tickers. "
                        "Try different symbols or a different date range."
                    )
                    st.stop()

                actual_tickers = data.columns.tolist()

                returns = data.pct_change().dropna()
                if returns.empty or returns.shape[1] < 2:
                    st.error(
                        "Error: Insufficient common data points for correlation calculation. "
                        "Try a different date range or different tickers."
                    )
                    st.stop()

                # Correlation check (ignore diagonal)
                corr = returns.corr().abs()
                np.fill_diagonal(corr.values, 0.0)
                if corr.max().max() >= 0.999:
                    st.error(
                        "Error: Assets are (almost) perfectly correlated. "
                        "Choose less correlated assets or change the date range."
                    )
                    st.stop()

                mean_returns = returns.mean()   # monthly means
                cov_matrix = returns.cov()      # monthly covariance

                # Covariance matrix sanity check (positive-definite)
                try:
                    np.linalg.cholesky(cov_matrix.values)
                except LinAlgError:
                    st.error(
                        "Error: Covariance matrix is singular or not positive definite. "
                        "Choose assets with less correlation or a different date range."
                    )
                    st.stop()

                # Sum of weights must be 1
                constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

                # ----- OPTIMIZATION -----
                weights_min_risk = minimize_volatility(
                    mean_returns, cov_matrix, constraints, len(actual_tickers)
                )

                if not np.allclose(np.sum(weights_min_risk), 1.0, atol=1e-5):
                    raise ValueError("Optimization failed to find valid weights (sum != 1).")

                std_min, ret_min = calculate_portfolio_performance(
                    weights_min_risk, mean_returns, cov_matrix
                )

                # ----- SIMULATIONS -----
                results = generate_random_portfolios(
                    mean_returns, cov_matrix, len(actual_tickers), num_portfolios=10000
                )

                # ----- OUTPUT -----
                st.header("Results and Efficient Frontier")

                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal Annualized Return (μ)", f"{ret_min * 100:.2f}%")
                col2.metric("Minimum Annualized Volatility (σ)", f"{std_min * 100:.2f}%")

                sharpe_min = 0.0 if std_min == 0 else ret_min / std_min
                col3.metric("Sharpe Ratio (Min Risk)", f"{sharpe_min:.2f}")

                # Efficient Frontier plot
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=results[0, :],
                            y=results[1, :],
                            mode="markers",
                            marker=dict(
                                color=results[2, :],
                                colorbar=dict(title="Sharpe Ratio"),
                                colorscale="RdYlBu",
                                showscale=True,
                                size=5,
                            ),
                            name="Simulated Portfolios",
                        )
                    ]
                )

                fig.add_trace(
                    go.Scatter(
                        x=[std_min],
                        y=[ret_min],
                        mode="markers",
                        marker=dict(color="green", size=15, symbol="star"),
                        name="Global Minimum Variance Portfolio (GMVP)",
                    )
                )

                fig.update_layout(
                    title="Markowitz Efficient Frontier",
                    xaxis_title="Annualized Volatility (Risk)",
                    yaxis_title="Annualized Return (μ)",
                    hovermode="closest",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Weights table
                st.subheader("Optimal Asset Allocation (GMVP)")
                weights_df = pd.DataFrame(
                    {
                        "Asset": actual_tickers,
                        "Optimal Weight": [f"{w * 100:.2f}%" for w in weights_min_risk],
                    }
                )
                st.dataframe(
                    weights_df.sort_values(by="Optimal Weight", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(
                "Optimization Failed. Please change the date range or ticker symbols. "
                f"Error Details: {e}"
            )
