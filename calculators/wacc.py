import streamlit as st
import json

st.title('Valuation App: Financial Metrics Analysis')

# Function to calculate WACC with actual cost of equity and debt values
def calculate_wacc(equity, debt, interest_expense, tax_rate):
    total_value = equity + debt
    cost_of_equity = 0.1  # Replace with actual value if available
    cost_of_debt = interest_expense / debt if debt != 0 else 0
    weight_equity = equity / total_value
    weight_debt = debt / total_value
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return wacc

# Function to calculate financial metrics
def calculate_metrics(ticker_data, target_date):
    try:
        # Extract necessary data
        financials = ticker_data['financials']
        balance_sheet = ticker_data['balance_sheet']
        cashflow = ticker_data['cashflow']

        if 'Total Revenue' not in financials or target_date not in financials['Total Revenue']:
            # Use the most recent available date if target date is not found
            target_date = max(financials['Total Revenue'].keys(), default=target_date)
            st.write(f"Using most recent available date: {target_date}")

        revenue = financials['Total Revenue'].get(target_date, 0)
        net_income = financials['Net Income'].get(target_date, 0)
        total_assets = balance_sheet['Total Assets'].get(target_date, 0)
        total_equity = balance_sheet['Common Stock Equity'].get(target_date, 0)
        ebitda = financials['EBITDA'].get(target_date, 0)
        fcf = cashflow.get('Free Cash Flow', {}).get(target_date, 0)
        interest_expense = financials['Interest Expense'].get(target_date, 0)

        # Calculating NOPLAT (assuming tax rate of 21%)
        noplat = ebitda * (1 - 0.21)

        # Assuming total debt as the sum of current and long-term debt, with checks for existence
        current_debt = balance_sheet.get('Current Debt', {}).get(target_date, 0)
        long_term_debt = balance_sheet.get('Long Term Debt', {}).get(target_date, 0)
        total_debt = current_debt + long_term_debt

        # Calculating WACC
        tax_rate = 0.21  # Example tax rate, replace with actual if available
        wacc = calculate_wacc(total_equity, total_debt, interest_expense, tax_rate)

        # Calculating ROIC
        invested_capital = total_assets - balance_sheet.get('Current Liabilities', {}).get(target_date, 0)
        roic = noplat / invested_capital if invested_capital != 0 else 0

        # Calculating EBITDA Margin
        ebitda_margin = ebitda / revenue if revenue != 0 else 0

        # Placeholder VAN calculation (Net Present Value)
        # Simplified, normally it involves discounting future cash flows
        van = fcf / wacc if wacc != 0 else 0  # Here we simplify by dividing FCF by WACC

        return {
            "Revenue": revenue,
            "Net Income": net_income,
            "Total Assets": total_assets,
            "Total Equity": total_equity,
            "EBITDA": ebitda,
            "ROA": net_income / total_assets if total_assets != 0 else 0,
            "ROE": net_income / total_equity if total_equity != 0 else 0,
            "ROIC": roic,
            "EBITDA Margin": ebitda_margin,
            "FCF": fcf,
            "NOPLAT": noplat,
            "WACC": wacc,
            "VAN": van
        }
    except Exception as e:
        return f"Error calculating financial metrics: {e}"

# Load JSON data
uploaded_file = st.file_uploader("Choose a JSON file")
if uploaded_file is not None:
    data = json.load(uploaded_file)
    all_metrics = {}

    # Display metrics for each ticker
    for ticker, ticker_data in data.items():
        st.write(f"Processing data for {ticker}...")
        metrics = calculate_metrics(ticker_data, '2023-09-30 00:00:00')
        if isinstance(metrics, dict):
            st.write(f"Metrics for {ticker}:")
            st.json(metrics)
            all_metrics[ticker] = metrics
        else:
            st.write(metrics)

    # Save the calculated metrics to a JSON file
    with open('financial_analytics.json', 'w') as json_file:
        json.dump(all_metrics, json_file, indent=4, default=str)

    st.write('Financial metrics have been successfully saved to financial_analytics.json')
    st.success("Data processing completed.")
else:
    st.write("Please upload a JSON file to proceed.")
