# 📈 Análisis Técnica con Python

## 🛠️ Bibliotecas Usadas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Subo mi archivo de ejemplo

```python
df = pd.read_csv('scraped_historical_data.csv')
print(df.head())
```

## 📊 Obtención de Indicadores

### 📈 Cálculo de EMA

```python
df['EMA'] = df['close'].ewm(span=N, adjust=False).mean()

short_window = 20
long_window = 50

df['SMA_short'] = df['close'].rolling(window=short_window).mean()
df['SMA_long'] = df['close'].rolling(window=long_window).mean()

# Señales de cruce
df['Signal'] = 0.0
df['Signal'][short_window:] = np.where(df['SMA_short'][short_window:] > df['SMA_long'][short_window:], 1.0, 0.0)
df['Position'] = df['Signal'].diff()

# Cálculo de SMA y desviación estándar
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()

# Bandas de Bollinger
df['Upper Band'] = df['SMA_20'] + (df['stddev'] * 2)
df['Lower Band'] = df['SMA_20'] - (df['stddev'] * 2)

# Definición de ventanas cortas y largas
short_window = 40
long_window = 100

# Cálculo de medias móviles
df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()

# Señales de compra/venta
df['signal'] = 0
df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0)
df['positions'] = df['signal'].diff()
```

### 📉 Cálculo del Oscilador Estocástico

```python
def stochastic_oscillator(df, window=14):
    df['L14'] = df['low'].rolling(window=window).min()
    df['H14'] = df['high'].rolling(window=window).max()
    df['%K'] = 100 * ((df['close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    return df
```

### 📈 Cálculo del MACD

```python
def macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    return df
```

### 📉 Cálculo del RSI

```python
def rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df
```

## 💡 Estrategias con Indicadores

### 📈 Stoch RSI

```python
def calculate_stoch_rsi(df, window=14):
    df = calculate_rsi(df, window)
    min_rsi = df['rsi'].rolling(window=window).min()
    max_rsi = df['rsi'].rolling(window=window).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)
    
    return df

def stoch_rsi_reversion_strategy(df, oversold=0.2, overbought=0.8):
    df['signal'] = 0
    df['signal'][(df['stoch_rsi'] < oversold) & (df['stoch_rsi'].shift(1) >= oversold)] = 1
    df['signal'][(df['stoch_rsi'] > overbought) & (df['stoch_rsi'].shift(1) <= overbought)] = -1
    df['position'] = df['signal'].diff()
    
    return df

def stoch_rsi_trend_following_strategy(df):
    df['signal'] = 0
    df['signal'][(df['stoch_rsi'] > 0.5) & (df['stoch_rsi'].shift(1) <= 0.5)] = 1
    df['signal'][(df['stoch_rsi'] < 0.5) & (df['stoch_rsi'].shift(1) >= 0.5)] = -1
    df['position'] = df['signal'].diff()
    
    return df

# Gráficos
def plot_stoch_rsi(df):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    ax1.set_ylabel('Precio')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['stoch_rsi'], label='RSI Estocástico', color='orange')
    ax2.axhline(y=0.2, color='red', linestyle='--')
    ax2.axhline(y=0.8, color='green', linestyle='--')
    ax2.set_ylabel('RSI Estocástico')
    
    fig.suptitle('Precio de Cierre y RSI Estocástico')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
```

### 📉 Divergencias

```python
def find_divergences(df):
    df['Price_Trend'] = np.where(df['close'] > df['close'].shift(1), 'up', 'down')
    df['Stoch_Trend'] = np.where(df['%D'] > df['%D'].shift(1), 'up', 'down')
    df['MACD_Trend'] = np.where(df['MACD'] > df['MACD'].shift(1), 'up', 'down')
    df['RSI_Trend'] = np.where(df['RSI'] > df['RSI'].shift(1), 'up', 'down')

    df['Regular_Divergence'] = np.where(
        ((df['Price_Trend'] == 'up') & (df['Stoch_Trend'] == 'down')) |
        ((df['Price_Trend'] == 'down') & (df['Stoch_Trend'] == 'up')),
        'divergence', 'none'
        )
    
    df['Hidden_Divergence'] = np.where(
        ((df['Price_Trend'] == 'up') & (df['Stoch_Trend'] == 'up')) |
        ((df['Price_Trend'] == 'down') & (df['Stoch_Trend'] == 'down')),
        'hidden_divergence', 'none'
        )
    
    return df
```

## 📈 Análisis Gráfico

### 📊 Fibonacci

```python
def find_fibonacci_retracement_signals(df, high_col, low_col, close_col, window=30):
    df['swing_high'] = df[high_col].rolling(window=window).max()
    df['swing_low'] = df[low_col].rolling(window=window).min()
    
    signals = []

    for i in range(window, len(df)):
        high = df['swing_high'].iloc[i]
        low = df['swing_low'].iloc[i]
        fib_levels = fibonacci_retracement_levels(high, low)
        
        if df[close_col].iloc[i] <= fib_levels['level_0.618']:
            signals.append(('buy', df.index[i], df[close_col].iloc[i]))
        
        elif df[close_col].iloc[i] >= fib_levels['level_0.618']:
            signals.append(('sell', df.index[i], df[close_col].iloc[i]))
    
    return signals

signals = find_fibonacci_retracement_signals(df, 'high', 'low', 'close')

# Visualizar señales
plt.figure(figsize=(10, 6))
plt.plot(df['close'], label='Precio de Cierre')

for signal in signals:
    if signal[0] == 'buy':
        plt.plot(signal[1], signal[2], 'g^', markersize=10)
    elif signal[0] == 'sell':
        plt.plot(signal[1], signal[2], 'rv', markersize=10)

plt.title('Señales de Entrada y Salida basadas en Fibonacci')
plt.legend()
plt.show()
```

### 📈 Líneas de Tendencia

```python
def identify_trend(df

, window=20):
    df['trend'] = np.where(df['close'] > df['close'].rolling(window).mean(), 'up', 'down')
    
    return df

def identify_pullback(df, window=20):
    df['pullback'] = np.where((df['close'] < df['close'].rolling(window).mean()) & (df['trend'] == 'up'), 'pullback_up', 
    np.where((df['close'] > df['close'].rolling(window).mean()) & (df['trend'] == 'down'), 'pullback_down', 'no_pullback'))
    
    return df

def identify_reversal_patterns(df, pattern):
    # Placeholder de una función que identifica patrones de reversión específicos
    df['reversal_pattern'] = np.nan  # Ejemplo de marcador de posición
    
df = identify_reversal_patterns(df, 'double_top')

def identify_continuation_patterns(df, trend_col, pattern_col):
    df['continuation_pattern'] = np.where((df[trend_col] == 'up') & (df[pattern_col] == 'pullback_up'), 'continuation_up', 
    np.where((df[trend_col] == 'down') & (df[pattern_col] == 'pullback_down'), 'continuation_down', 'no_pattern'))
    
    return df

# Ejemplo de uso para identificar patrones de continuación
df = identify_continuation_patterns(df, 'trend', 'pullback')

def identify_double_top(df):
    df['double_top'] = np.where((df['high'] == df['high'].shift(1)) & (df['high'] > df['high'].shift(2)), 'double_top', 'no_pattern')
    
    return df

def plot_pattern(df, pattern_col, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['close'], label='Precio de Cierre')

    for i in range(len(df)):
        if df[pattern_col].iloc[i] != 'no_pattern':
            plt.scatter(df.index[i], df['close'].iloc[i], color='red', marker='x', s=100, label=df[pattern_col].iloc[i])

    plt.title(title)
    plt.legend()
    plt.show()
```

### 📈 Ichimoku

```python
def ichimoku(df):
    # Tenkan-sen (Línea de Conversión)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    # Kijun-sen (Línea Base)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    # Senkou Span A (Span Adelantado A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Span Adelantado B)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    # Chikou Span (Span Rezagado)
    df['chikou_span'] = df['close'].shift(-26)

    return df

# Supongamos que 'df' es un DataFrame con columnas 'high', 'low', 'close'
df = ichimoku(df)

def ichimoku_cross_strategy(df):
    df['signal'] = 0
    df['signal'][df['tenkan_sen'] > df['kijun_sen']] = 1
    df['signal'][df['tenkan_sen'] < df['kijun_sen']] = -1
    df['position'] = df['signal'].diff()

    return df

def ichimoku_cloud_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b']) & (df['tenkan_sen'] > df['kijun_sen'])] = 1
    df['signal'][(df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b']) & (df['tenkan_sen'] < df['kijun_sen'])] = -1
    df['position'] = df['signal'].diff()

    return df

def plot_ichimoku(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Precio de Cierre')
    plt.plot(df.index, df['tenkan_sen'], label='Tenkan-sen', color='red')
    plt.plot(df.index, df['kijun_sen'], label='Kijun-sen', color='blue')
    plt.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] >= df['senkou_span_b'], facecolor='lightgreen', interpolate=True, alpha=0.5)  
    plt.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] < df['senkou_span_b'], facecolor='lightcoral', interpolate=True, alpha=0.5)
    plt.plot(df.index, df['chikou_span'], label='Chikou Span', color='green')
    plt.legend()
    plt.title('Ichimoku Kinko Hyo')
    plt.show()
```

### 📉 Perfil Volumétrico

```python
def plot_volume_profile(df, vol_profile):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Gráfico de precios
    ax1.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    ax1.set_ylabel('Precio')
    
    # Gráfico del perfil de volumen
    ax2 = ax1.twinx()
    ax2.barh(vol_profile['price'], vol_profile['volume'], alpha=0.3, color='gray')
    ax2.set_ylabel('Volumen')
    
    ax1.set_title('Perfil de Volumen')
    ax1.legend()
    plt.show()

def poc_trading_strategy(df, vol_profile):
    poc_price = vol_profile.loc[vol_profile['volume'].idxmax(), 'price']
    df['signal'] = 0
    df['signal'][(df['close'] > poc_price) & (df['volume'] > df['volume'].rolling(window=5).mean())] = 1
    df['signal'][(df['close'] < poc_price) & (df['volume'] > df['volume'].rolling(window=5).mean())] = -1
    df['position'] = df['signal'].diff()
    
    return df, poc_price

def volume_node_trading_strategy(df, vol_profile, threshold=0.1):
    vol_profile['volume_pct'] = vol_profile['volume'] / vol_profile['volume'].sum()
    hvn = vol_profile[vol_profile['volume_pct'] > threshold]['price']
    lvn = vol_profile[vol_profile['volume_pct'] < threshold]['price']
    df['signal'] = 0
    df['signal'][(df['close'].isin(lvn)) & (df['volume'] > df['volume'].rolling(window=5).mean())] = 1
    df['signal'][(df['close'].isin(hvn)) & (df['volume'] > df['volume'].rolling(window=5).mean())] = -1
    df['position'] = df['signal'].diff()
    
    return df
```

### 📊 VWAP

```python
def calculate_vwap(df):
    df['cum_price_vol'] = (df['close'] * df['volume']).cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_price_vol'] / df['cum_volume']
    
    return df

def plot_vwap(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    plt.plot(df.index, df['vwap'], label='VWAP', color='orange', linestyle='--')
    plt.title('Precio de Cierre y VWAP')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()

def vwap_reversion_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['vwap']) & (df['close'].shift(1) < df['vwap'].shift(1))] = 1
    df['signal'][(df['close'] < df['vwap']) & (df['close'].shift(1) > df['vwap'].shift(1))] = -1
    df['position'] = df['signal'].diff()

    return df

def vwap_trend_following_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['vwap'])] = 1
    df['signal'][(

df['close'] < df['vwap'])] = -1
    df['position'] = df['signal'].diff()

    return df
    
```

## 📚 Integración de Estrategias de Trading

### 🕒 Paso 1: Análisis del Marco Temporal

**Empezar desde marcos temporales más amplios (mensual y semanal) y luego pasar a marcos más cortos (diario e intradía):** Identificar niveles clave de soporte y resistencia en estos marcos temporales.

**Calcular y visualizar el perfil de volumen para identificar POC, HVN y LVN:** Usar estos niveles como referencia para tomar decisiones de entrada y salida.

### ✅ Paso 2: Confirmación Técnica

**Calcular VWAP** y utilizarlo como referencia para identificar posibles puntos de reversión o seguimiento de tendencias.

**Calcular y visualizar el RSI Estocástico:** Buscar señales de reversión cuando el RSI Estocástico esté en niveles de sobrecompra o sobreventa.

**Calcular y visualizar los componentes de Ichimoku:** Confirmar señales de trading con cruces de Tenkan-sen y Kijun-sen y la posición del precio respecto a la nube Ichimoku.

### 🧠 Paso 3: Psicología y Gestión del Riesgo

**Checklist y Guía de Entrada:**

1. Seguir una lista de verificación antes de entrar en una operación.
2. Evaluar los pros y contras de cada operación potencial basándose en niveles clave y confirmaciones técnicas.

**Control Emocional:**

1. Mantener la disciplina y seguir las reglas establecidas.
2. Aceptar la responsabilidad de las decisiones de trading y evitar culpar al mercado.
3. Adoptar una mentalidad de probabilidad y estar preparado para cualquier resultado.

## 🧩 Estrategias Avanzadas para Trading

### 📈 Trade Largo con Confluencia de Marcos Temporales

**Marco temporal principal donde se encontró la configuración de entrada:** Acción del precio alcista = sesgo alcista.

**Marco temporal inferior:** Acción del precio alcista nuevamente = confirmando el sesgo alcista del marco temporal principal.

**Marco temporal de entrada:** Acción del precio alcista nuevamente = confirma los otros dos marcos temporales = estrategia de entrada.

### 📉 Trade Corto con Confluencia de Marcos Temporales

**Marco temporal principal donde se encontró la configuración de entrada:** Acción del precio bajista = sesgo bajista.

**Marco temporal inferior:** Acción del precio bajista nuevamente = confirmando el sesgo bajista del marco temporal principal.

**Marco temporal de entrada:** Acción del precio bajista nuevamente = confirma los otros dos marcos temporales = estrategia de entrada.

```python
def identify_time_frame_confluence(df, higher_time_frame, lower_time_frame, entry_time_frame):
    # Identificar la tendencia en diferentes marcos temporales
    df['higher_tf_trend'] = np.where(df[higher_time_frame] > df[higher_time_frame].shift(1), 'up', 'down')
    df['lower_tf_trend'] = np.where(df[lower_time_frame] > df[lower_time_frame].shift(1), 'up', 'down')
    df['entry_tf_trend'] = np.where(df[entry_time_frame] > df[entry_time_frame].shift(1), 'up', 'down')
    
    # Confirmar la confluencia de los marcos temporales
    df['confluence'] = np.where(
        (df['higher_tf_trend'] == df['lower_tf_trend']) & (df['lower_tf_trend'] == df['entry_tf_trend']), 
        df['higher_tf_trend'], 'no_confluence')
    
    return df
```

### 📈 Estrategia de Combo de Tendencia Dinámica

El combo de tendencia dinámica se enfoca en operar en la dirección de la tendencia dominante, esperando una ruptura seguida de un retroceso.

1. Seguimiento de la tendencia principal.
2. Confirmar la ruptura y el retroceso.
3. Buscar confluencia de marcos temporales.

```python
def dynamic_trend_combo_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['resistance']) & (df['close'].shift(1) <= df['resistance'])] = 1
    df['signal'][(df['close'] < df['support']) & (df['close'].shift(1) >= df['support'])] = -1
    df['position'] = df['signal'].diff()
    
    return df
```

### 📉 Gestión de Riesgo y Tamaño de Posición

**Cuánto arriesgar por operación:** Como principiante, arriesga un máximo del 1% de tu cuenta por operación.

**Relación de Riesgo-Recompensa:** Asegúrate de que la relación riesgo-recompensa sea favorable, por ejemplo, 1:2.

**Stop-Loss y Objetivo de Beneficio:** Establece un stop-loss para limitar las pérdidas y un objetivo de beneficio para cerrar la posición en ganancias.

**Tamaño de la Posición:** Calcula el tamaño de la posición basado en el riesgo y el tamaño de la cuenta.

#### Ejemplo:

```python
def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss_price):
    risk_amount = account_balance * risk_per_trade
    pip_risk = abs(entry_price - stop_loss_price)
    position_size = risk_amount / pip_risk
    
    return position_size

account_balance = 5000
risk_per_trade = 0.01
entry_price = 76.75
stop_loss_price = 75.90

position_size = calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss_price)
print(f"Tamaño de la posición: {position_size} unidades")
```

### 📈 Uso del Apalancamiento y el Margen

**Apalancamiento:** Permite controlar una mayor cantidad de dinero con una menor cantidad de capital.

**Margen:** Es la cantidad de dinero que necesitas en tu cuenta para mantener una posición apalancada.

**Selección de Broker:** Elige un broker confiable y adecuado para tus necesidades.

**Practica en una cuenta demo antes de operar con dinero real.**

### 📊 Estrategia de Correlación Usando Bitcoin como Indicador Principal

**Utilizar la configuración de Bitcoin como indicador principal para entradas correlacionadas.**

**Confirmación de ruptura en Bitcoin:** Esperar una confirmación de ruptura en Bitcoin antes de buscar entradas en activos correlacionados.

**Entradas basadas en correlación:** Buscar configuraciones de entrada en activos que estén fuertemente correlacionados con Bitcoin.

```python
def correlation_trading_strategy(df, lead_asset, correlated_asset):
    df['lead_signal'] = np.where((df[lead_asset] > df[lead_asset].shift(1)), 1, -1)
    df['correlated_signal'] = np.where((df[correlated_asset] > df[correlated_asset].shift(1)), 1, -1)
    df['trade_signal'] = np.where((df['lead_signal'] == df['correlated_signal']), df['lead_signal'], 0)
    
    return df
```

### 📈 Ejemplo de Confluencia en USD/CAD

**Marco temporal semanal:** Identificar una tendencia bajista.

**Marco temporal diario:** Confirmar la continuación de la tendencia bajista.

**Marco temporal intradía:** Buscar una entrada en una ruptura a la baja.

```python
def identify_trend(df, window=20):
    df['trend'] = np.where(df['close'] > df['close'].rolling(window).mean(), 'up', 'down')
    
    return df

def identify_pullback(df, window=20):
    df['pullback'] = np.where((df['close'] < df['close'].rolling(window).mean()) & (df['trend'] == 'up'), 'pullback_up', 
                              np.where((df['close'] > df['close'].rolling(window).mean()) & (df['trend'] == 'down'), 'pullback_down', 'no_pullback'))
    
    return df
```

### 📉 Estrategia de Salida Múltiple

Esta estrategia implica identificar múltiples niveles de soporte y resistencia para establecer varios objetivos de salida. La clave es observar la acción del precio en estos niveles para decidir en cuál salir.

**Procedimiento:**

1. Identificar niveles de soporte y resistencia cercanos.
2. Observar la acción del precio en estos niveles.
3. Salir en el nivel donde se observe una reacción significativa de la acción del precio.

```python
def identify_exit_targets(df, levels):
    df['exit_target'] = np.nan
    for level in levels:
        df.loc[df['close'] == level, 'exit_target'] = 'exit'
    
    return df
```

**Ejemplos:**

1. Si hay múltiples velas de reacción en un nivel de resistencia, salir en ese nivel.
2. Si no hay reacción significativa en el primer nivel, esperar a los siguientes niveles.

## 🧩 Integración de Estrategias

Finalmente, integramos estas estrategias avanzadas con la gestión del riesgo, relación riesgo-recompensa y uso de indicadores líderes para entradas correlacionadas.

**Gestión del Riesgo:**

1. Riesgo por operación: No más del 1% del capital total.
2. Relación Riesgo-Recompensa: Idealmente 1:2 o mejor.

**Posicionamiento:** Calcular el tamaño de la posición basado en el riesgo máximo permitido y la distancia del stop-loss.

### Uso de Indicadores Líderes

Utilizar la acción del precio de

 activos altamente correlacionados (como Bitcoin) para confirmar entradas en otros activos correlacionados (como acciones de empresas mineras de criptomonedas).

```python
# Gestión del riesgo
account_balance = 5000
risk_per_trade = account_balance * 0.01

def calculate_position_size(entry_price, stop_loss_price, risk_per_trade):
    pips = abs(entry_price - stop_loss_price)
    pip_value = risk_per_trade / pips
    
    return pip_value

entry_price = 76.75
stop_loss_price = 75.90
pip_value = calculate_position_size(entry_price, stop_loss_price, risk_per_trade)

print(f"Valor por pip: {pip_value}")
```

### 📊 Visualización de la Estrategia Integrada

```python
def plot_combined_strategy(df, vol_profile):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    ax1.plot(df.index, df['vwap'], label='VWAP', color='orange', linestyle='--')
    ax1.plot(df.index, df['tenkan_sen'], label='Tenkan-sen', color='red')
    ax1.plot(df.index, df['kijun_sen'], label='Kijun-sen', color='blue')
    ax1.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] >= df['senkou_span_b'], facecolor='lightgreen', interpolate=True, alpha=0.5)
    ax1.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] < df['senkou_span_b'], facecolor='lightcoral', interpolate=True, alpha=0.5)
    ax1.plot(df.index, df['chikou_span'], label='Chikou Span', color='green')
    ax2 = ax1.twinx()
    ax2.barh(vol_profile['price'], vol_profile['volume'], alpha=0.3, color='gray')
    ax2.set_ylabel('Volumen')
    ax1.set_title('Estrategia Combinada')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
```
