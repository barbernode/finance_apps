# P&L Crafter

## Import Libraries
```python
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items
```

## Function to Split the 'Descrição' Column
```python
def split_descricao(descricao):
    parts = descricao.split(' - ')
    if len(parts) >= 4:
        transaction = parts[0]
        remittant = parts[1]
        bank = parts[3]
        return pd.Series([transaction, remittant, bank])
    else:
        return pd.Series([None, None, None])
```

## Function to Retrieve Total Income and Expenses for a Given Remittant
```python
def get_totals(df, name):
    total_income = df[(df['Remittant'] == name) & (df['Type'] == 'Income')]['Amount'].sum()
    total_expenses = df[(df['Remittant'] == name) & (df['Type'] == 'Expense')]['Amount'].sum()
    return total_income, total_expenses
```

## Function to Update the Balance Sheet Based on User Assignments
```python
def update_balance_sheet(balance_sheet, category, income, expense, month_key):
    net_result = income - expense
    if category in balance_sheet:
        if month_key in balance_sheet[category]:
            balance_sheet[category][month_key] += net_result
        else:
            balance_sheet[category][month_key] = net_result
    else:
        balance_sheet[category] = {month_key: net_result}
    return balance_sheet
```

## Initialize the Streamlit App
```python
st.title('P&L Crafter')
```

## Initialize Session State
```python
if 'balance_sheet' not in st.session_state:
    st.session_state['balance_sheet'] = {}
if 'assignments' not in st.session_state:
    st.session_state['assignments'] = {}
if 'undo_stack' not in st.session_state:
    st.session_state['undo_stack'] = []
if 'redo_stack' not in st.session_state:
    st.session_state['redo_stack'] = []
```

## Upload CSV File
```python
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df_flow = pd.read_csv(uploaded_file)
    
    # Rename columns
    df_flow.rename(columns={'Valor': 'Amount', 'Data': 'Date'}, inplace=True)
    
    # Display the columns and head of the dataframe for debugging
    st.write("DataFrame Columns:", df_flow.columns.tolist())
    st.write("DataFrame Head:", df_flow.head())
    
    df_flow_copy = df_flow.copy()

    # Apply the function to the 'Descrição' column and create new columns
    df_flow_copy[['Transaction', 'Remittant', 'Bank']] = df_flow_copy['Descrição'].apply(split_descricao)

    # Drop the original 'Descrição' and 'Identificador' columns
    if 'Descrição' in df_flow_copy.columns and 'Identificador' in df_flow_copy.columns:
        df_flow_copy.drop(columns=['Descrição', 'Identificador'], inplace=True)
    
    # Check if the 'Amount' column exists
    if 'Amount' in df_flow_copy.columns:
        # Create the 'Type' column
        df_flow_copy['Type'] = df_flow_copy['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
    else:
        st.error("The 'Amount' column is not found in the uploaded CSV file.")

    # Convert 'Date' column to datetime if it exists
    if 'Date' in df_flow_copy.columns:
        df_flow_copy['Date'] = pd.to_datetime(df_flow_copy['Date'], dayfirst=True)
        df_flow_copy['Month'] = df_flow_copy['Date'].dt.month
        df_flow_copy['Year'] = df_flow_copy['Date'].dt.year

        # Select the time frame
        selected_years = st.multiselect('Select Year(s)', sorted(df_flow_copy['Year'].unique()))
        if selected_years:
            selected_months = st.multiselect('Select Month(s)', sorted(df_flow_copy[df_flow_copy['Year'].isin(selected_years)]['Month'].unique()))

            # Filter the dataframe based on the selected time frame
            df_filtered = df_flow_copy[(df_flow_copy['Year'].isin(selected_years)) & (df_flow_copy['Month'].isin(selected_months))]

            # List all unique remittants in the filtered dataframe
            unique_remittants = df_filtered['Remittant'].unique()
            st.write(f"Unique remittants in the selected time frame: {len(unique_remittants)}")

            # Display the unique remittants
            st.write(unique_remittants)

            # Drag-and-Drop Interface
            for year in selected_years:
                st.subheader(f"{year}")
                for month in selected_months:
                    st.subheader(f"{year}-{month:02d}")

                    # Placeholder for draggable items
                    remittant_items = []
                    for remittant in unique_remittants:
                        total_income, total_expenses = get_totals(df_filtered, remittant)
                        remittant_info = f"{remittant} - Income: {total_income}, Expenses: {total_expenses}"
                        remittant_items.append(remittant_info)
                    
                    sorted_items = sort_items(
                        remittant_items, 
                        direction="horizontal",
                        key=f"sortable_{year}_{month}"
                    )

                    # Placeholder for drop zones
                    categories = ['Cash and Banks', 'Accounts Receivable', 'Inventory of Consumable Material',
                                  'Buildings (net accumulated depreciation)', 'Vehicles and Machinery (net accumulated depreciation)',
                                  'Financial Investments', 'Current Bank Credit', 'Interest Payable',
                                  'Accounts Payable', 'Salaries Payable', 'Taxes Payable', 'Customer Advances',
                                  'Non-current Bank Credit']
                    
                    for category in categories:
                        drop_zone = sort_items(
                            [],
                            key=f"{category}_{year}_{month}",
                            direction="horizontal"
                        )

                        # Check for dropped items
                        for item in drop_zone:
                            remittant_name = item.split(' - ')[0]
                            total_income, total_expenses = get_totals(df_filtered, remittant_name)
                            month_key = f"{year}-{month:02d}"
                            st.session_state['balance_sheet'] = update_balance_sheet(st.session_state['balance_sheet'], category, total_income, total_expenses, month_key)
                            st.session_state['assignments'][remittant_name] = (category, month_key)

            # Display Balance Sheet
            st.write(st.session_state['balance_sheet'])

            # Download the Balance Sheet as an Excel File
            if st.button('Download Balance Sheet'):
                writer = pd.ExcelWriter('Balance_Sheet.xlsx', engine='xlsxwriter')

                for year in selected_years:
                    year_data = {cat: {k: v for k, v in vals.items() if k.startswith(f"{year}-")} for cat, vals in st.session_state['balance_sheet'].items()}
                    if year_data:
                        df = pd.DataFrame(year_data).fillna(0).T
                        df = df.reindex(sorted(df.columns), axis=1)  # Sort columns (months) in ascending order
                        df.to_excel(writer, sheet_name=str(year))

                writer.save()
                st.write("Balance sheet downloaded as 'Balance_Sheet.xlsx'")
```

