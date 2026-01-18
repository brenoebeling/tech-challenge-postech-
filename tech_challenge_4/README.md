# Tech Challenge – Fase 4  
## Deploy do Modelo Preditivo do IBOVESPA com Streamlit

Este projeto faz parte do Tech Challenge da Pós Tech (FIAP) e tem como 
objetivo
disponibilizar de forma produtiva o modelo preditivo desenvolvido na Fase 
2.

---

## 🎯 Objetivo
Criar uma aplicação interativa utilizando **Streamlit** que permita ao 
usuário
inserir dados do IBOVESPA e visualizar a previsão de tendência (alta ou 
baixa)
para o próximo dia.

---

## 🧠 Modelo
- Tipo: Classificação binária
- Algoritmo: Regressão Logística
- Target:  
  - 1 → fechamento do dia seguinte maior que o atual  
  - 0 → fechamento do dia seguinte menor ou igual
- Modelo salvo via `joblib` (`model.pkl`)

---

## 📁 Estrutura do Projeto

```text
tech_challenge_4/
│
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── model.pkl

