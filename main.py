from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Busca dados do yfinance
def fetch_adj_close(tickers: List[str], years: int = 3) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Download de dados falhou")

    adj = raw["Adj Close"].copy()

    adj = adj.reindex(columns=[t for t in tickers if t in adj.columns])

    # dropna remove dias em que nenhum ativo foi negociado (ex: feriados)
    # ffill() preenche dados ausentes com o valor do dia anterior (para normalização)
    adj = adj.dropna(how="all").ffill()

    return adj, start, end


# Normaliza df para Base 100
def normalize_base100(df: pd.DataFrame) -> pd.DataFrame:
   
    # A normalização começa no primeiro dia em que todos os ativos têm dados
    df_common = df.dropna(how="any")
    if df_common.empty:
        raise ValueError("Sem interseção de datas com dados para todos os tickers.")
    base = df_common.iloc[0]

    # Pega a primeira linha de dados comuns
    norm = (df_common / base) * 100.0
    return norm


# Gráfico de performance normalizada
def plot_performance(norm: pd.DataFrame, start, end, out_path: Path) -> Path:
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    ax = norm.plot(linewidth=2, ax=plt.gca())
    ax.set_title(f'Performance Normalizada (Base 100) das Magnificent Seven ({start.date()} a {end.date()})', fontsize=16, weight='bold')
    ax.set_ylabel('Performance (Base 100)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.legend(title='Tickers', loc='upper left', frameon=True)
    
    ax.axhline(100, linestyle='--', color='grey', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


# Correlação retornos diários dos últimos 3 meses
def compute_corr_last3m(adj: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
   
    start_3m = end - pd.DateOffset(months=3)

    # Só dias com dados de todos os tickers
    recent = adj.loc[adj.index >= start_3m].dropna(how="any")

    # Correlação calculada sobre retornos (evitar correlação espúria)
    returns = recent.pct_change().dropna(how="all")
    corr = returns.corr()
    return corr


# Heatmap da matriz de correlação
def plot_corr_heatmap(corr: pd.DataFrame, end: pd.Timestamp, out_path: Path) -> Path:

    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, 
                mask=mask, 
                annot=True,     
                fmt=".2f",      
                cmap='coolwarm',
                vmin=-1, 
                vmax=1,
                linewidths=0.5,
                cbar_kws={"shrink":.8})
                
    plt.title(f'Matriz de Correlação de Retornos Diários (Últimos 3 Meses)', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()

    return out_path


# Função principal
def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Baixando dados…")
    adj, start, end = fetch_adj_close(TICKERS, years=3)
    print(f"Janela: {start.date()} → {end.date()} | Linhas: {len(adj)}")

    print("Normalizando em base=100…")
    norm = normalize_base100(adj)

    perf_path = out_dir / "performance_normalized.png"
    print(f"Plotando performance → {perf_path}")
    plot_performance(norm, start, end, perf_path)

    print("Calculando correlação (últimos 3 meses)…")
    corr = compute_corr_last3m(adj, end)

    corr_path = out_dir / "corr_heatmap_last3m.png"
    print(f"Plotando heatmap → {corr_path}")
    plot_corr_heatmap(corr, end, corr_path)

    adj.to_csv(out_dir / "adj_close.csv", index=True)
    norm.to_csv(out_dir / "normalized_base100.csv", index=True)
    corr.to_csv(out_dir / "corr_last3m.csv", index=True)
    

# Só roda quando o script é executado diretamente
if __name__ == "__main__":
    main()
