import pandas as pd
import yfinance as yf
import json
import streamlit as st
import requests

# Function to fetch interest rates from FRED
def fetch_interest_rate_from_fred(api_key, series_id="DGS10"):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    response = requests.get(url)
    data = response.json()
    if "observations" in data:
        return float(data["observations"][-1]["value"]) / 100  # Convert percentage to decimal
    else:
        raise ValueError("Unable to fetch interest rates from FRED")

# Function to fetch data from Yahoo Finance
def fetch_yahoo_data(ticker, api_key):
    ticker = ticker.strip().upper()  # Ensure ticker is in correct format
    ticker_obj = yf.Ticker(ticker)
    financials = ticker_obj.financials.T
    cashflow = ticker_obj.cashflow.T
    balance_sheet = ticker_obj.balance_sheet.T
    beta = ticker_obj.info.get('beta', 1.0)  # Default to 1.0 if not available

    # Fetch interest rate from FRED
    try:
        interest_rate = fetch_interest_rate_from_fred(api_key)
    except Exception as e:
        interest_rate = 0.04  # Default interest rate if FRED fetch fails
        st.write(f"Error fetching interest rate from FRED: {e}")

    # Free Cash Flow
    fcf = None
    if 'Free Cash Flow' in cashflow.index:
        fcf = cashflow.loc['Free Cash Flow'].dropna().infer_objects()

    # Convert indices to string to avoid issues with JSON serialization
    financials.index = financials.index.map(str)
    cashflow.index = cashflow.index.map(str)
    balance_sheet.index = balance_sheet.index.map(str)
    if fcf is not None:
        fcf.index = fcf.index.map(str)

    return {
        'financials': financials.to_dict(),
        'cashflow': cashflow.to_dict(),
        'balance_sheet': balance_sheet.to_dict(),
        'fcf': fcf.to_dict() if fcf is not None else {},
        'beta': beta,
        'interest_rate': interest_rate
    }

# Streamlit app to input tickers and fetch data
st.title('Financial Data Scraper from Yahoo Finance')

api_key =   # Your provided FRED API key
uploaded_file = st.file_uploader("Upload a CSV file with tickers", type=['csv'])
if uploaded_file is not None:
    tickers_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    us_exchanges = ['NMS', 'NYQ', 'NCM', 'NGM', 'PNK', 'ASE']

    st.write("Columns in the uploaded file:")
    st.json(list(tickers_df.columns))

    selected_column = st.selectbox("Select the column with tickers", tickers_df.columns)
    tickers = tickers_df[selected_column]

    # Filter tickers to include only those from US exchanges
    filtered_tickers_df = tickers_df[tickers_df['Exchange'].isin(us_exchanges)]

    # Display a multi-select box for users to select specific tickers
    selected_tickers = st.multiselect("Select tickers to fetch data for", filtered_tickers_df[selected_column].tolist())

    if st.button('Fetch Data'):
        all_data = {}
        for ticker in selected_tickers:
            ticker = ticker.strip()
            try:
                st.write(f'Fetching data for {ticker}...')
                yahoo_data = fetch_yahoo_data(ticker, api_key)
                all_data[ticker] = yahoo_data
            except Exception as e:
                st.write(f'Error fetching data for {ticker}: {e}')

        # Save the fetched data to a JSON file
        with open('financial_data.json', 'w') as json_file:
            json.dump(all_data, json_file, indent=4, default=str)

        st.write('Data has been successfully saved to financial_data.json')
