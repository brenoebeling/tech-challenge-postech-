import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import datetime


# ======================
# Config
# ======================
st.set_page_config(page_title="IBOVESPA Predictor", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# ======================
# Load model
# ======================
model = joblib.load(MODEL_PATH)

st.title("📈 Previsão de Tendência do IBOVESPA")
st.write("Modelo preditivo treinado para indicar **alta ou baixa** do índice no próximo dia.")
st.subheader("📊 Exemplo de visualização (mock)")
import numpy as np

mock_series = np.cumsum(np.random.randn(30))
st.line_chart(mock_series)

# ======================
# Inputs
# ======================
st.header("🔢 Insira os dados do dia atual")

return_ = st.number_input("Retorno diário", value=0.0)
ma_5 = st.number_input("Média móvel 5 dias", value=0.0)
ma_10 = st.number_input("Média móvel 10 dias", value=0.0)
volatility = st.number_input("Volatilidade (10 dias)", value=0.0)
close_open = st.number_input("Close - Open", value=0.0)
high_low = st.number_input("High - Low", value=0.0)

LOG_PATH = BASE_DIR / "log_inputs.csv"

if st.button("🔮 Prever"):
    X = pd.DataFrame([[
        return_, ma_5, ma_10,
        volatility, close_open, high_low
    ]], columns=[
        "return", "ma_5", "ma_10",
        "volatility", "close_open", "high_low"
    ])

    # =========================
    # LOG DE USO (AQUI 👇)
    # =========================

    log_row = X.copy()
    log_row["timestamp"] = datetime.datetime.now().isoformat()

    if LOG_PATH.exists():
        log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_row.to_csv(LOG_PATH, index=False)

    # =========================
    # PREVISÃO
    # =========================

    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][prediction]

    if prediction == 1:
        st.success(f"📈 Tendência de ALTA (probabilidade: {prob:.2%})")
    else:
        st.error(f"📉 Tendência de BAIXA (probabilidade: {prob:.2%})")

