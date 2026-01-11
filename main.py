import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ROBUST DATA INGESTION (With Fallback)
# ==========================================
print("Initializing Physics Engine...")

def get_data():
    ticker = "^NSEI"
    print(f"Attempting to download {ticker}...")
    try:
        # Try fetching max history first
        df = yf.download(ticker, period="10y", progress=False, auto_adjust=False)

        # Flatten MultiIndex (Fix for new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Standardize Columns
        col_map = {c: c for c in df.columns}
        if "Adj Close" in df.columns: col_map["Adj Close"] = "Price"
        elif "Close" in df.columns:   col_map["Close"] = "Price"
        df = df.rename(columns=col_map)

        if 'Price' not in df.columns:
            raise ValueError("Price column missing from download.")

        # Handle Volume
        if 'Volume' not in df.columns: df['Volume'] = 0
        df['Volume'] = df['Volume'].replace(0, np.nan).fillna(method='ffill').fillna(100000)

        # Validation
        if len(df) < 50:
            raise ValueError("Downloaded data is too short (<50 rows).")

        print(f" -> Success! Downloaded {len(df)} rows.")
        return df[['Price', 'Volume']]

    except Exception as e:
        print(f" -> Download Failed/Empty ({e}). Generating SYNTHETIC data for demonstration.")
        # GENERATE DUMMY DATA SO THE CODE DOES NOT CRASH
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
        prices = [10000]
        for _ in range(499):
            change = np.random.normal(0, 100)
            prices.append(max(100, prices[-1] + change))

        df_fake = pd.DataFrame({'Price': prices, 'Volume': 100000}, index=dates)
        return df_fake

df = get_data()

# ==========================================
# 2. RESPONSIVE PHYSICS (SNAP & JERK)
# ==========================================
# EMA for fast reaction
df['EMA'] = df['Price'].ewm(span=10, adjust=False).mean()
df['E_STD'] = df['Price'].ewm(span=10, adjust=False).std()

# Stiffness (k)
df['Upper'] = df['EMA'] + (2 * df['E_STD'])
df['Lower'] = df['EMA'] - (2 * df['E_STD'])
df['Bandwidth'] = (df['Upper'] - df['Lower']) / (df['EMA'] + 1e-9)
df['k'] = 1.0 / (df['Bandwidth'] + 0.001)
df['k'] = df['k'] / df['k'].mean()

# Damping (c)
vol_ma = df['Volume'].rolling(20).mean()
df['c'] = (df['Volume'] / (vol_ma + 1e-9)).fillna(1.0) * 0.5

# Kinematics
df['Log_Ret'] = np.log(df['Price'] / df['Price'].shift(1))
forces = df['Log_Ret'].fillna(0).values * 100

n = len(df)
v, x, a, jerk, snap = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
k_vals, c_vals = df['k'].values, df['c'].values
m, dt = 1.0, 1.0

for i in range(1, n):
    F_net = forces[i] - c_vals[i]*v[i-1] - k_vals[i]*x[i-1]
    a[i] = F_net / m
    jerk[i] = (a[i] - a[i-1]) / dt
    snap[i] = (jerk[i] - jerk[i-1]) / dt # The "Snap" Signal
    v[i] = v[i-1] + a[i]*dt
    x[i] = x[i-1] + v[i]*dt

df['Phys_Power'] = np.abs(forces * v)
df['Signal_Jerk'] = pd.Series(jerk, index=df.index).rolling(3).mean()
df['Signal_Snap'] = pd.Series(snap, index=df.index).rolling(3).mean()

# ==========================================
# 3. FEATURES & SPLIT (The Fix)
# ==========================================
TRADING_DAYS = 21

# Target: Future Volatility
df['Target_Vol'] = df['Log_Ret'].rolling(TRADING_DAYS).std().shift(-TRADING_DAYS) * np.sqrt(252)

# Features: Lagged Volatility
df['Current_Vol'] = df['Log_Ret'].rolling(TRADING_DAYS).std() * np.sqrt(252)
df['Vol_Lag_1'] = df['Current_Vol'].shift(1)
df['Ret_Lag_1'] = df['Log_Ret'].shift(1)

# Drop NaNs
df_model = df.dropna()

# CRITICAL FIX: Safe Split Logic
if len(df_model) < 50:
    print("Warning: Data extremely short after processing.")
    split_ratio = 0.5
else:
    split_ratio = 0.85 # Standard 85/15 split

split_idx = int(len(df_model) * split_ratio)
train_data = df_model.iloc[:split_idx]
test_data = df_model.iloc[split_idx:]

print(f"Training Samples: {len(train_data)} | Testing Samples: {len(test_data)}")

# Stop if split failed
if len(train_data) == 0:
    raise ValueError("Training data is empty. The dataset is too small for the requested windows.")

features = [
    'k', 'Phys_Power',
    'Signal_Jerk', 'Signal_Snap',
    'Current_Vol', 'Vol_Lag_1', 'Ret_Lag_1'
]

# ==========================================
# 4. TRAINING
# ==========================================
model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=500, max_depth=6,
    loss='absolute_error', random_state=42
)

model.fit(train_data[features], train_data['Target_Vol'])
test_data['Predicted_Vol'] = model.predict(test_data[features])

# ==========================================
# 5. VISUALIZATION
# ==========================================
def bs_pricer(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

T, r = 30/365, 0.07

# Calculate Prices
test_data['Price_Ideal'] = bs_pricer(test_data['Price'], test_data['Price'], T, r, test_data['Target_Vol'])
test_data['Price_AI']    = bs_pricer(test_data['Price'], test_data['Price'], T, r, test_data['Predicted_Vol'])
test_data['Price_Std']   = bs_pricer(test_data['Price'], test_data['Price'], T, r, test_data['Current_Vol'])

# Plot
plt.figure(figsize=(12, 10))
plt.style.use('dark_background')

# Subplot 1: Prices
plt.subplot(3, 1, 1)
plt.plot(test_data.index, test_data['Price_Ideal'], color='#00FF00', lw=1.5, label='Ideal Price')
plt.plot(test_data.index, test_data['Price_AI'], color='cyan', lw=1.5, ls='--', label='AI Responsive Model')
plt.plot(test_data.index, test_data['Price_Std'], color='white', lw=0.5, alpha=0.5, label='Standard BS')
plt.title('Option Pricing Model (Responsive Physics)')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.2)

# Subplot 2: Snap Signal
plt.subplot(3, 1, 2)
plt.plot(test_data.index, test_data['Signal_Snap'], color='orange', lw=1, label='Physics "Snap" (4th Deriv)')
plt.title('The Snap Signal (Market Breakpoint)')
plt.legend()
plt.grid(alpha=0.2)

# Subplot 3: Volatility
plt.subplot(3, 1, 3)
plt.plot(test_data.index, test_data['Target_Vol'], color='green', alpha=0.6, label='Actual Vol')
plt.plot(test_data.index, test_data['Predicted_Vol'], color='cyan', linestyle='--', label='AI Vol')
plt.title('Volatility Forecast')
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

err_std = (test_data['Price_Std'] - test_data['Price_Ideal']).abs().mean()
err_ai = (test_data['Price_AI'] - test_data['Price_Ideal']).abs().mean()
print(f"--- RESULTS ---")
print(f"Std Error: {err_std:.2f}")
print(f"AI Error:  {err_ai:.2f}")
if err_std > 0:
    print(f"Gain:      {((err_std - err_ai)/err_std)*100:.2f}%")