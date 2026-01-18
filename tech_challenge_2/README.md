# Tech Challenge – Fase 2  
## Previsão de Tendência do IBOVESPA

### 🎯 Objetivo
Desenvolver um modelo preditivo capaz de indicar se o índice IBOVESPA irá fechar
em alta ou baixa no dia seguinte, utilizando dados históricos diários do próprio índice.

O modelo serve como insumo para dashboards internos de apoio à decisão
em um fundo de investimentos.

---

### 📊 Dados
- Fonte: Investing.com
- Frequência: Diária
- Período: Mínimo de 2 anos
- Variáveis principais:
  - Open, High, Low, Close, Volume

---

### 🧠 Estratégia de Modelagem
- Criação de variáveis derivadas:
  - Retorno diário
  - Médias móveis (5, 10, 20 dias)
  - Volatilidade de curto prazo
  - Range diário (High - Low)
- Target:
  - 1 → fechamento do dia seguinte maior que o atual
  - 0 → caso contrário
- Split temporal:
  - Treino: histórico
  - Teste: últimos 30 dias

---

### 🤖 Modelo Utilizado
- Regressão Logística
- Motivos da escolha:
  - Simplicidade
  - Interpretabilidade
  - Baixo risco de overfitting
  - Boa performance em classificação binária

---

### 📈 Resultados
- Acuracidade superior a 75% no conjunto de teste
- Métricas analisadas:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

### 🚀 Próximos Passos
O modelo treinado foi salvo em formato `.pkl` e será utilizado na Fase 4,
onde será realizado o deploy em ambiente produtivo com Streamlit,
incluindo visualizações interativas e monitoramento de performance.
