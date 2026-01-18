from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ibovespa.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# =========================
# Leitura do CSV  👇
# =========================
df = pd.read_csv(DATA_PATH)

# 👉 LOGO APÓS O read_csv, entra ISTO:
df = df.rename(columns={
    "Data": "Date",
    "Último": "Close",
    "Abertura": "Open",
    "Máxima": "High",
    "Mínima": "Low",
    "Vol.": "Volume"
})

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# =========================
# Feature Engineering
# =========================
df["return"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_10"] = df["Close"].rolling(10).mean()
df["volatility"] = df["Close"].rolling(10).std()
df["close_open"] = df["Close"] - df["Open"]
df["high_low"] = df["High"] - df["Low"]

df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

features = [
    "return", "ma_5", "ma_10",
    "volatility", "close_open", "high_low"
]

X = df[features]
y = df["target"]

X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

# =========================
# Treino + PKL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at {MODEL_PATH}")
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	precision_score, 
	recall_score, 
	confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ibovespa.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ibovespa.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

# Ensure date order
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Feature engineering
df["return"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_10"] = df["Close"].rolling(10).mean()
df["volatility"] = df["Close"].rolling(10).std()
df["close_open"] = df["Close"] - df["Open"]
df["high_low"] = df["High"] - df["Low"]

# Target
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop NaNs
df = df.dropna()

# Features / target
features = [
    "return", "ma_5", "ma_10",
    "volatility", "close_open", "high_low"
]

X = df[features]
y = df["target"]

# Temporal split
X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at {MODEL_PATH}")
