## Cálculos de Tipo de Cambio en Streamlit

### Tipo de Cambio Directo e Indirecto
```python
rate_direct = st.number_input("Ingrese el tipo de cambio directo (A/B = c)", min_value=0.0, value=1.0, step=0.01)
st.write(f"Tipo de cambio indirecto (B/A) = 1/c: {1/rate_direct:.4f}")
```

### Tipos Cruzados
```python
rate_ab = st.number_input("Ingrese el tipo de cambio A/B", min_value=0.0, value=1.0, step=0.01)
rate_bc = st.number_input("Ingrese el tipo de cambio B/C", min_value=0.0, value=1.0, step=0.01)
st.write(f"Tipo de cambio cruzado A/C = A/B * B/C: {rate_ab * rate_bc:.4f}")
```

### Cálculo de Arbitraje
```python
buy_price = st.number_input("Precio de compra de la moneda", min_value=0.0, value=1.0, step=0.01)
sell_price = st.number_input("Precio de venta de la moneda", min_value=0.0, value=1.0, step=0.01)
st.write(f"Ganancia de arbitraje = Precio de venta - Precio de compra: {sell_price - buy_price:.4f}")
```

### Swaps de Divisas
```python
spot_rate = st.number_input("Ingrese la tasa spot", min_value=0.0, value=1.0, step=0.01)
forward_rate = st.number_input("Ingrese la tasa forward", min_value=0.0, value=1.0, step=0.01)
st.write(f"Tasa de swap = Tasa spot + Tasa forward: {spot_rate + forward_rate:.4f}")
```

## Gráficos de Opciones

### Simulación de Comprador Call
```python
K_call = st.number_input("Precio de ejercicio (Call)", min_value=1000, max_value=5000, value=3000, step=100)
P_call = st.number_input("Prima pagada (Call)", min_value=0, max_value=1000, value=200, step=50)
S_min_call = st.number_input("Precio mínimo del subyacente (Call)", min_value=1000, max_value=4000, value=2500, step=100)
S_max_call = st.number_input("Precio máximo del subyacente (Call)", min_value=2000, max_value=5000, value=3500, step=100)

def plot_call_buyer(K, P, S_min, S_max):
    fig, ax = plt.subplots()
    S = np.linspace(S_min, S_max, 100)
    payoff = np.maximum(S - K, 0) - P
    ax.plot(S, payoff, label="Call Buyer")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(K, color='black', lw=0.5, linestyle='--')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Profit / Loss')
    ax.legend()
    st.pyplot(fig)

plot_call_buyer(K_call, P_call, S_min_call, S_max_call)
```

### Simulación de Comprador Put
```python
K_put = st.number_input("Precio de ejercicio (Put)", min_value=1000, max_value=5000, value=3000, step=100)
P_put = st.number_input("Prima pagada (Put)", min_value=0, max_value=1000, value=200, step=50)
S_min_put = st.number_input("Precio mínimo del subyacente (Put)", min_value=1000, max_value=4000, value=2500, step=100)
S_max_put = st.number_input("Precio máximo del subyacente (Put)", min_value=2000, max_value=5000, value=3500, step=100)

def plot_put_buyer(K, P, S_min, S_max):
    fig, ax = plt.subplots()
    S = np.linspace(S_min, S_max, 100)
    payoff = np.maximum(K - S, 0) - P
    ax.plot(S, payoff, label="Put Buyer")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(K, color='black', lw=0.5, linestyle='--')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Profit / Loss')
    ax.legend()
    st.pyplot(fig)

plot_put_buyer(K_put, P_put, S_min_put, S_max_put)
```
