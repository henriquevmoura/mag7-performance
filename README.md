# Mag7 3y Performance and 3m Correlation (+ Rolling)

Análise “Mag Seven” (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA):
- Coleta `Adj Close` via `yfinance` (3 anos, data-base último du).
- Performance **normalizada (base 100)**.
- **Matriz de correlação (3 meses)** com base em retornos diários.
- **Extra:** correlação média móvel 3m ao longo do tempo.

## Como rodar
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py

