# Análisis Técnica con Python

```
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
````

### Supongamos que 'df' es un DataFrame con una columna 'close' que contiene los precios de cierre
```
N = 50  # Número de períodos para el SMA
df['SMA'] = df['close'].rolling(window=N).mean()
````

### Cálculo de EMA
````
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
````
### Implementación de Algoritmos de Trading con Medias Móviles

````
class MovingAverageCrossStrategy(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageCrossStrategy)
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.run()
cerebro.plot()

# Cálculo del Oscilador Estocástico
def stochastic_oscillator(df, window=14):
    df['L14'] = df['low'].rolling(window=window).min()
    df['H14'] = df['high'].rolling(window=window).max()
    df['%K'] = 100 * ((df['close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

# Cálculo del MACD
def macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

# Cálculo del RSI
def rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Aplicamos las funciones a nuestro DataFrame
df = stochastic_oscillator(df)
df = macd(df)
df = rsi(df)

````

# Función para identificar divergencias
````
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

# Aplicamos la función a nuestro DataFrame
df = find_divergences(df)

def fibonacci_retracement_levels(high, low):
    diff = high - low
    levels = {
        'level_0': high,
        'level_0.382': high - diff * 0.382,
        'level_0.5': high - diff * 0.5,
        'level_0.618': high - diff * 0.618,
        'level_1': low
    }
    return levels

# Ejemplo de uso con datos de precios
high = df['high'].max()
low = df['low'].min()
fib_levels = fibonacci_retracement_levels(high, low)
````
### Fibonacci
````
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
````
### Lineas de Tendencia
````
def identify_trend(df, window=20):
    df['trend'] = np.where(df['close'] > df['close'].rolling(window).mean(), 'up', 'down')
    return df

def identify_pullback(df, window=20):
    df['pullback'] = np.where((df['close'] < df['close'].rolling(window).mean()) & (df['trend'] == 'up'), 'pullback_up', 
                              np.where((df['close'] > df['close'].rolling(window).mean()) & (df['trend'] == 'down'), 'pullback_down', 'no_pullback'))
    return df

df = identify_trend(df)
df = identify_pullback(df)

def identify_reversal_patterns(df, pattern):
    # Placeholder for una función que identifica patrones de reversión específicos
    df['reversal_pattern'] = np.nan  # Ejemplo de marcador de posición
    return df

# Ejemplo de uso para identificar patrones de reversión
df = identify_reversal_patterns(df, 'double_top')

def identify_reversal_patterns(df, pattern):
    # Placeholder for una función que identifica patrones de reversión específicos
    df['reversal_pattern'] = np.nan  # Ejemplo de marcador de posición
    return df

# Ejemplo de uso para identificar patrones de reversión
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

df = identify_double_top(df)

def plot_pattern(df, pattern_col, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['close'], label='Precio de Cierre')

    for i in range(len(df)):
        if df[pattern_col].iloc[i] != 'no_pattern':
            plt.scatter(df.index[i], df['close'].iloc[i], color='red', marker='x', s=100, label=df[pattern_col].iloc[i])

    plt.title(title)
    plt.legend()
    plt.show()

# Ejemplo de visualización para patrones de doble techo
plot_pattern(df, 'double_top', 'Identificación de Doble Techo')
````
### Ichimoku
````
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

df = ichimoku_cross_strategy(df)

def ichimoku_cloud_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b']) & (df['tenkan_sen'] > df['kijun_sen'])] = 1
    df['signal'][(df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b']) & (df['tenkan_sen'] < df['kijun_sen'])] = -1
    df['position'] = df['signal'].diff()

    return df

df = ichimoku_cloud_strategy(df)

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

plot_ichimoku(df)

def volume_profile(df, price_col='close', volume_col='volume', bins=50):
    min_price = df[price_col].min()
    max_price = df[price_col].max()
    price_bins = np.linspace(min_price, max_price, bins)
    
    df['price_bin'] = np.digitize(df[price_col], price_bins)
    volume_profile = df.groupby('price_bin')[volume_col].sum()
    
    volume_profile = pd.DataFrame(volume_profile).reset_index()
    volume_profile['price'] = price_bins[volume_profile['price_bin'] - 1]
    
    return volume_profile
````
### Volume Profile

````
# Supongamos que 'df' es un DataFrame con columnas 'close' y 'volume'
vol_profile = volume_profile(df)

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

plot_volume_profile(df, vol_profile)

def poc_trading_strategy(df, vol_profile):
    poc_price = vol_profile.loc[vol_profile['volume'].idxmax(), 'price']
    
    df['signal'] = 0
    df['signal'][(df['close'] > poc_price) & (df['volume'] > df['volume'].rolling(window=5).mean())] = 1
    df['signal'][(df['close'] < poc_price) & (df['volume'] > df['volume'].rolling(window=5).mean())] = -1
    df['position'] = df['signal'].diff()
    
    return df, poc_price

df, poc_price = poc_trading_strategy(df, vol_profile)

def volume_node_trading_strategy(df, vol_profile, threshold=0.1):
    vol_profile['volume_pct'] = vol_profile['volume'] / vol_profile['volume'].sum()
    
    hvn = vol_profile[vol_profile['volume_pct'] > threshold]['price']
    lvn = vol_profile[vol_profile['volume_pct'] < threshold]['price']
    
    df['signal'] = 0
    df['signal'][(df['close'].isin(lvn)) & (df['volume'] > df['volume'].rolling(window=5).mean())] = 1
    df['signal'][(df['close'].isin(hvn)) & (df['volume'] > df['volume'].rolling(window=5).mean())] = -1
    df['position'] = df['signal'].diff()
    
    return df

df = volume_node_trading_strategy(df, vol_profile)
````
### VWAP

````

def calculate_vwap(df):
    df['cum_price_vol'] = (df['close'] * df['volume']).cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_price_vol'] / df['cum_volume']
    return df

# Supongamos que 'df' es un DataFrame con columnas 'close' y 'volume'
df = calculate_vwap(df)

def plot_vwap(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    plt.plot(df.index, df['vwap'], label='VWAP', color='orange', linestyle='--')
    plt.title('Precio de Cierre y VWAP')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()

plot_vwap(df)

def vwap_reversion_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['vwap']) & (df['close'].shift(1) < df['vwap'].shift(1))] = 1
    df['signal'][(df['close'] < df['vwap']) & (df['close'].shift(1) > df['vwap'].shift(1))] = -1
    df['position'] = df['signal'].diff()

    return df

df = vwap_reversion_strategy(df)

def vwap_trend_following_strategy(df):
    df['signal'] = 0
    df['signal'][(df['close'] > df['vwap'])] = 1
    df['signal'][(df['close'] < df['vwap'])] = -1
    df['position'] = df['signal'].diff()

    return df

df = vwap_trend_following_strategy(df)

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df
````
### Stoch RSI
````

# Supongamos que 'df' es un DataFrame con una columna 'close'
df = calculate_rsi(df)

def calculate_stoch_rsi(df, window=14):
    df = calculate_rsi(df, window)
    min_rsi = df['rsi'].rolling(window=window).min()
    max_rsi = df['rsi'].rolling(window=window).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)
    return df

# Calcular el RSI Estocástico
df = calculate_stoch_rsi(df)

def stoch_rsi_reversion_strategy(df, oversold=0.2, overbought=0.8):
    df['signal'] = 0
    df['signal'][(df['stoch_rsi'] < oversold) & (df['stoch_rsi'].shift(1) >= oversold)] = 1
    df['signal'][(df['stoch_rsi'] > overbought) & (df['stoch_rsi'].shift(1) <= overbought)] = -1
    df['position'] = df['signal'].diff()
    return df

df = stoch_rsi_reversion_strategy(df)

def stoch_rsi_trend_following_strategy(df):
    df['signal'] = 0
    df['signal'][(df['stoch_rsi'] > 0.5) & (df['stoch_rsi'].shift(1) <= 0.5)] = 1
    df['signal'][(df['stoch_rsi'] < 0.5) & (df['stoch_rsi'].shift(1) >= 0.5)] = -1
    df['position'] = df['signal'].diff()
    return df

df = stoch_rsi_trend_following_strategy(df)

def plot_stoch_rsi(df):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Gráfico del precio de cierre
    ax1.plot(df.index, df['close'], label='Precio de Cierre', color='blue')
    ax1.set_ylabel('Precio')
    
    # Gráfico del RSI Estocástico
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['stoch_rsi'], label='RSI Estocástico', color='orange')
    ax2.axhline(y=0.2, color='red', linestyle='--')
    ax2.axhline(y=0.8, color='green', linestyle='--')
    ax2.set_ylabel('RSI Estocástico')
    
    fig.suptitle('Precio de Cierre y RSI Estocástico')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

plot_stoch_rsi(df)

````

# Integración de Estrategias de Trading
## Análisis Técnico

### Perfil de Volumen (Volume Profile):

Identificación de áreas de soporte y resistencia basadas en el volumen.
Estrategias de trading en POC (Point of Control), HVN (High Volume Node) y LVN (Low Volume Node).
VWAP (Volume Weighted Average Price):

Estrategias de reversión a la media y seguimiento de tendencias utilizando VWAP.
RSI Estocástico (Stochastic RSI):

Estrategias de reversión y seguimiento de tendencias utilizando el RSI Estocástico.
Ichimoku Kinko Hyo:

Cruce de Tenkan-sen y Kijun-sen.
Trading con la nube Ichimoku.
Psicología del Trading

Trading in the Zone - Mark Douglas:

Comprender que el mercado es neutral y dominar la psicología del trading.
Identificar y superar los problemas comunes de los traders, como la falta de reglas, no asumir responsabilidades, adicción a recompensas aleatorias, y la diferencia entre control externo e interno.
Adoptar una mentalidad de probabilidad y aceptar la incertidumbre del mercado.
The Intelligent Investor - Benjamin Graham:

Principios de inversión inteligente: minimizar pérdidas irreversibles, maximizar ganancias sostenibles y controlar las emociones.
Focalización en el valor intrínseco de las acciones en lugar de seguir las tendencias del mercado.
Estrategia de Trading Combinada

### Paso 1: Análisis del Marco Temporal

Perspectiva General:

Empezar desde marcos temporales más amplios (mensual y semanal) y luego pasar a marcos más cortos (diario e intradía).
Identificar niveles clave de soporte y resistencia en estos marcos temporales.

Perfil de Volumen:

Calcular y visualizar el perfil de volumen para identificar POC, HVN y LVN.
Usar estos niveles como referencia para tomar decisiones de entrada y salida.

### Paso 2: Confirmación Técnica

VWAP:

Calcular VWAP y utilizarlo como referencia para identificar posibles puntos de reversión o seguimiento de tendencias.
RSI Estocástico:

Calcular y visualizar el RSI Estocástico.
Buscar señales de reversión cuando el RSI Estocástico esté en niveles de sobrecompra o sobreventa.
Ichimoku:

Calcular y visualizar los componentes de Ichimoku.
Confirmar señales de trading con cruces de Tenkan-sen y Kijun-sen y la posición del precio respecto a la nube Ichimoku.

### Paso 3: Psicología y Gestión del Riesgo

**Checklist y Guía de Entrada:**

Seguir una lista de verificación antes de entrar en una operación.
Evaluar los pros y contras de cada operación potencial basándose en niveles clave y confirmaciones técnicas.

**Control Emocional:**

Mantener la disciplina y seguir las reglas establecidas.
Aceptar la responsabilidad de las decisiones de trading y evitar culpar al mercado.
Adoptar una mentalidad de probabilidad y estar preparado para cualquier resultado.

## Implementación en Python

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cálculo de RSI
def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# Cálculo de RSI Estocástico
def calculate_stoch_rsi(df, window=14):
    df = calculate_rsi(df, window)
    min_rsi = df['rsi'].rolling(window=window).min()
    max_rsi = df['rsi'].rolling(window=window).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)
    return df

# Cálculo de VWAP
def calculate_vwap(df):
    df['cum_price_vol'] = (df['close'] * df['volume']).cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_price_vol'] / df['cum_volume']
    return df

# Cálculo del Perfil de Volumen
def volume_profile(df, price_col='close', volume_col='volume', bins=50):
    min_price = df[price_col].min()
    max_price = df[price_col].max()
    price_bins = np.linspace(min_price, max_price, bins)
    df['price_bin'] = np.digitize(df[price_col], price_bins)
    volume_profile = df.groupby('price_bin')[volume_col].sum()
    volume_profile = pd.DataFrame(volume_profile).reset_index()
    volume_profile['price'] = price_bins[volume_profile['price_bin'] - 1]
    return volume_profile

# Cálculo de Ichimoku
def ichimoku(df):
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['chikou_span'] = df['close'].shift(-26)
    return df

# Aplicamos las funciones a nuestro DataFrame
df = calculate_stoch_rsi(df)
df = calculate_vwap(df)
df = ichimoku(df)
vol_profile = volume_profile(df)

# Visualización de los componentes y estrategias
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

plot_combined_strategy(df, vol_profile)

```













