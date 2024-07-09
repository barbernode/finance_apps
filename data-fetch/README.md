# Financial Data Fetcher

## Import Libraries
```python
import os
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import yfinance as yf
from yahooquery import Ticker
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import requests
import json
```

## Load Environment Variables
```python
load_dotenv()
```

## Get API Keys from Environment Variables
```python
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')
CURRENCYLAYER_API_KEY = os.getenv('CURRENCYLAYER_API_KEY')
```

## Initialize Binance Client
```python
def initialize_binance_client(api_key, api_secret):
    try:
        return Client(api_key, api_secret)
    except BinanceAPIException as e:
        st.error(f"Binance API Error: {e.message}")
        return None

client = initialize_binance_client(BINANCE_API_KEY, BINANCE_API_SECRET)
```

## Fetch Historical Data from Binance
```python
def fetch_binance_data(symbol, interval, start_date, end_date):
    try:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except BinanceAPIException as e:
        st.error(f'Binance API Error: {e.message}')
        return None
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return None
```

## Fetch Max Range Stock Data from Yahoo Finance
```python
def get_stock_data_yahoo(ticker):
    try:
        stock = yf.Ticker(ticker)
        stock_df = stock.history(period="max")
        return stock_df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None
```

## Fetch Financial Statements from Yahoo Finance
```python
def get_financials_yahoo(ticker):
    try:
        ticker_obj = Ticker(ticker)
        income_df = ticker_obj.income_statement().loc[ticker].reset_index()
        balance_df = ticker_obj.balance_sheet().loc[ticker].reset_index()
        cash_df = ticker_obj.cash_flow().loc[ticker].reset_index()

        income_df['asOfDate'] = pd.to_datetime(income_df['asOfDate'])
        balance_df['asOfDate'] = pd.to_datetime(balance_df['asOfDate'])
        cash_df['asOfDate'] = pd.to_datetime(cash_df['asOfDate'])

        income_df['Year'] = income_df['asOfDate'].dt.year
        balance_df['Year'] = balance_df['asOfDate'].dt.year
        cash_df['Year'] = cash_df['asOfDate'].dt.year

        return income_df, balance_df, cash_df
    except Exception as e:
        st.error(f"Error fetching financials: {e}")
        return None, None, None
```

## Fetch Data from FRED API
```python
def fetch_fred_data(api_key, series_id):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['observations'])
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from FRED: {e}")
        return None
```

## Fetch Data from World Bank API
```python
def fetch_world_bank_data(qterm, fl):
    try:
        url = f"https://search.worldbank.org/api/v2/wds?format=json&qterm={qterm}&fl={fl}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from World Bank: {e}")
        return None
```

## Fetch Data from NewsAPI
```python
def fetch_newsapi_data(api_key, query, from_date, to_date):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from NewsAPI: {e}")
        return None
```

## Fetch Data from SDMX API
```python
def fetch_sdmx_data(context, agency_id, resource_id, version, key):
    try:
        url = f"https://localhost/data/{context}/{agency_id}/{resource_id}/{version}/{key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from SDMX: {e}")
        return None
```

## Fetch Data from CurrencyLayer API
```python
def fetch_currencylayer_data(api_key, currencies, source="USD"):
    try:
        url = f"http://api.currencylayer.com/live?access_key={api_key}&currencies={currencies}&source={source}&format=1"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from CurrencyLayer: {e}")
        return None
```

## Streamlit App Setup
```python
st.title('Financial Data Fetcher')
```

## Sidebar Options for Data Sources
```python
data_source = st.sidebar.selectbox(
    "Select data source",
    ["Yahoo Finance (Stock Market)", "Binance (Crypto)", "FRED (Federal Reserve)", 
     "World Bank (Economic Indicators)", "NEWSAPI (News)", "SDMX (Statistical Data and Metadata Exchange)",
     "CurrencyLayer API (Forex)"]
)
```

## Load and Read the CSV DataFrame
```python
yahoo_tickers = pd.read_csv("tickerlist.csv")
```

## Search Functionality
```python
search_query = st.text_input("Search for a stock ticker or name")
if search_query:
    filtered_tickers = yahoo_tickers[
        yahoo_tickers.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
    ]
else:
    filtered_tickers = yahoo_tickers
```

## Display the DataFrame in a Sortable and Filterable Table
```python
st.subheader("Available Stock Tickers")
gb = GridOptionsBuilder.from_dataframe(filtered_tickers)
gb.configure_pagination()
gb.configure_side_bar()
gb.configure_default_column(filterable=True, sortable=True)
gridOptions = gb.build()

AgGrid(
    filtered_tickers,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    height=400,
    width='100%',
    theme='streamlit',
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED
)
```

## Streamlit UI for Yahoo Finance
```python
def yahoo_finance_ui():
    ticker = st.text_input("Enter stock ticker (e.g., AAPL)")
    if st.button('Fetch Yahoo Finance Data'):
        stock_df = get_stock_data_yahoo(ticker)
        if stock_df is not None:
            income_df, balance_df, cash_df = get_financials_yahoo(ticker)
            if all(df is not None for df in [income_df, balance_df, cash_df]):
                available_years = stock_df.index.year.unique()
                income_filtered = income_df[income_df['Year'].isin(available_years)]
                balance_filtered = balance_df[balance_df['Year'].isin(available_years)]
                cash_filtered = cash_df[cash_df['Year'].isin(available_years)]

                merged_financials = pd.merge(income_filtered, balance_filtered, on='Year', suffixes=('_income', '_balance'))
                merged_financials = pd.merge(merged_financials, cash_filtered, on='Year', suffixes=('', '_cashflow'))

                ticker_df = pd.merge(merged_financials, stock_df, left_on='Year', right_on=stock_df.index.year)

                numeric_columns = ticker_df.select_dtypes(include=['number']).columns
                non_numeric_columns = ticker_df.select_dtypes(exclude=['number']).columns

                ticker_df[numeric_columns] = ticker_df[numeric_columns].fillna(ticker_df[numeric_columns].mean())

                st.write(ticker_df)
                ticker_df.to_csv(f'{ticker}_yahoo_data.csv')
                st.success(f'Data saved to {ticker}_yahoo_data.csv')
            else:
                st.error("Failed to fetch financial statements.")
        else:
            st.error("Failed to fetch stock data.")
```

## Streamlit UI for Binance Data
```python
def binance_ui():
    symbol = st.text_input("Enter crypto symbol (e.g., BTCUSDT

)")
    interval = st.selectbox("Select interval", ['1m', '1h', '4h', '1d', '1w', '1M'])
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date", value=datetime.now())

    if st.button('Fetch Binance Data'):
        if client:
            binance_data = fetch_binance_data(symbol, interval, str(start_date), str(end_date))
            if binance_data is not None:
                st.write(binance_data)
                binance_data.to_csv(f'{symbol}_binance_data.csv', index=False)
                st.success(f'Data saved to {symbol}_binance_data.csv')
```

## Streamlit UI for FRED Data
```python
def fred_ui():
    series_id = st.text_input("Enter FRED series ID (e.g., DGS10)")
    if st.button('Fetch FRED Data'):
        fred_data = fetch_fred_data(FRED_API_KEY, series_id)
        if fred_data is not None:
            st.write(fred_data)
            fred_data.to_csv(f'{series_id}_fred_data.csv', index=False)
            st.success(f'Data saved to {series_id}_fred_data.csv')
```

## Streamlit UI for World Bank Data
```python
def world_bank_ui():
    st.markdown("""
    **World Bank API Querying Instructions:**

    - **Context**: Enter the context of the query (e.g., `wds` for documents and reports).
    - **Query Term (`qterm`)**: Enter the search term to query records (e.g., `wind turbine`).
    - **Fields (`fl`)**: Enter the fields to be returned as a comma-separated list (e.g., `docdt,count`).

    Example Query:
    - Context: `wds`
    - Query Term: `wind turbine`
    - Fields: `docdt,count`
    """)

    context = st.text_input("Enter context (e.g., wds)")
    qterm = st.text_input("Enter search term (qterm) (e.g., wind turbine)")
    fl = st.text_input("Enter fields to return (fl) (e.g., docdt,count)")
    if st.button('Fetch World Bank Data'):
        world_bank_data = fetch_world_bank_data(qterm, fl)
        if world_bank_data is not None:
            st.write(world_bank_data)
            with open(f'{qterm}_world_bank_data.json', 'w') as f:
                json.dump(world_bank_data, f, indent=4)
            st.success(f'Data saved to {qterm}_world_bank_data.json')
```

## Streamlit UI for NewsAPI Data
```python
def newsapi_ui():
    query = st.text_input("Enter search query (e.g., Tesla)")
    from_date = st.date_input("From date", value=datetime.now() - pd.Timedelta(days=30))
    to_date = st.date_input("To date", value=datetime.now())
    if st.button('Fetch News Data'):
        news_data = fetch_newsapi_data(NEWSAPI_API_KEY, query, from_date, to_date)
        if news_data is not None:
            st.write(news_data)
            with open(f'{query}_news_data.json', 'w') as f:
                json.dump(news_data, f, indent=4)
            st.success(f'Data saved to {query}_news_data.json')
```

## Streamlit UI for SDMX Data
```python
def sdmx_ui():
    context = st.text_input("Enter context (e.g., datastructure)")
    agency_id = st.text_input("Enter agency ID (e.g., WB)")
    resource_id = st.text_input("Enter resource ID (e.g., DF_WITS)")
    version = st.text_input("Enter version (e.g., 1.0)")
    key = st.text_input("Enter key (e.g., *)")
    if st.button('Fetch SDMX Data'):
        sdmx_data = fetch_sdmx_data(context, agency_id, resource_id, version, key)
        if sdmx_data is not None:
            st.write(sdmx_data)
            with open(f'{resource_id}_sdmx_data.json', 'w') as f:
                json.dump(sdmx_data, f, indent=4)
            st.success(f'Data saved to {resource_id}_sdmx_data.json')
```

## Streamlit UI for CurrencyLayer API Data
```python
def currencylayer_ui():
    currencies = st.text_input("Enter comma-separated currencies (e.g., EUR,GBP,CAD,PLN)")
    source = st.text_input("Enter source currency (default is USD)", value="USD")
    if st.button('Fetch Forex Data'):
        currency_data = fetch_currencylayer_data(CURRENCYLAYER_API_KEY, currencies, source)
        if currency_data is not None:
            st.write(currency_data)
            with open(f'{source}_currency_data.json', 'w') as f:
                json.dump(currency_data, f, indent=4)
            st.success(f'Data saved to {source}_currency_data.json')
```

## Display Appropriate UI Based on Selected Data Source
```python
if data_source == "Yahoo Finance (Stock Market)":
    yahoo_finance_ui()
elif data_source == "Binance (Crypto)":
    binance_ui()
elif data_source == "FRED (Federal Reserve)":
    fred_ui()
elif data_source == "World Bank (Economic Indicators)":
    world_bank_ui()
elif data_source == "NEWSAPI (News)":
    newsapi_ui()
elif data_source == "SDMX (Statistical Data and Metadata Exchange)":
    sdmx_ui()
elif data_source == "CurrencyLayer API (Forex)":
    currencylayer_ui()
```
