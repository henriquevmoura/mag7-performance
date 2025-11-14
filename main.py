#!/usr/bin/env python3
"""
Mag7 Performance Analyzer
- Baixa Adj Close (3 anos)
- Normaliza em base 100
- Plota performance acumulada
- Calcula correlação (retornos diários) dos últimos 3 meses
- Gera gráficos + CSVs em outputs/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Tickers da "Magnificent Seven"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def fetch_adj_close(tickers: List[str], years: int = 3) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Baixa preços ajustados (Adj Close) dos últimos `years` anos.
    Retorna: (DataFrame adj_close, start_date, end_date)
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    # yfinance: end é exclusivo; somamos 1 dia pra garantir captura de hoje
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("Download de dados falhou: DataFrame vazio retornado.")

    # Seleciona a coluna 'Adj Close' de forma robusta para 1 ou N tickers
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        if "Adj Close" in first_level:
            adj = raw["Adj Close"].copy()
        elif "Close" in first_level:
            # Fallback: usa Close se Adj Close não estiver presente
            adj = raw["Close"].copy()
        else:
            raise KeyError("Não encontrei 'Adj Close' ou 'Close' no retorno do yfinance.")
    else:
        # Caso raro: apenas 1 ticker vira coluna simples
        cols = list(raw.columns)
        if "Adj Close" in cols:
            adj = raw[["Adj Close"]].copy()
            adj.columns = [tickers[0]]
        elif "Close" in cols:
            adj = raw[["Close"]].copy()
            adj.columns = [tickers[0]]
        else:
            raise KeyError("Colunas esperadas não encontradas (Adj Close/Close).")

    # Ordena colunas na ordem do input e trata NaNs
    adj = adj.reindex(columns=[t for t in tickers if t in adj.columns])
    adj = adj.dropna(how="all").ffill()

    return adj, start, end


def normalize_base100(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza todas as séries para iniciar em 100 na primeira data comum.
    """
    df_common = df.dropna(how="any")
    if df_common.empty:
        raise ValueError("Sem interseção de datas com dados para todos os tickers.")
    base = df_common.iloc[0]
    norm = (df_common / base) * 100.0
    return norm


def plot_performance(norm: pd.DataFrame, out_path: Path) -> Path:
    """
    Plota gráfico de linhas da performance normalizada.
    """
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(13, 7))
    for col in norm.columns:
        plt.plot(norm.index, norm[col], label=col, linewidth=2)
    plt.title("Mag7 — Performance Normalizada (base=100)")
    plt.xlabel("Data")
    plt.ylabel("Índice (base=100)")
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()
    return out_path


def compute_corr_last3m(adj: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula correlação dos retornos diários dos últimos 3 meses.
    """
    start_3m = end - pd.DateOffset(months=3)
    recent = adj.loc[adj.index >= start_3m].dropna(how="any")
    returns = recent.pct_change().dropna(how="all")
    corr = returns.corr()
    return corr


def plot_corr_heatmap(corr: pd.DataFrame, end: pd.Timestamp, out_path: Path) -> Path:
    """
    Plota heatmap da matriz de correlação (com anotações).
    """
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "label": "Correlação"},
    )
    plt.title(f"Mag7 — Correlação de Retornos Diários (últimos 3 meses até {end.date()})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()
    return out_path


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
    plot_performance(norm, perf_path)

    print("Calculando correlação (últimos 3 meses)…")
    corr = compute_corr_last3m(adj, end)

    corr_path = out_dir / "corr_heatmap_last3m.png"
    print(f"Plotando heatmap → {corr_path}")
    plot_corr_heatmap(corr, end, corr_path)

    # (Útil p/ auditoria/reuso)
    adj.to_csv(out_dir / "adj_close.csv", index=True)
    norm.to_csv(out_dir / "normalized_base100.csv", index=True)
    corr.to_csv(out_dir / "corr_last3m.csv", index=True)

    print("✅ Pronto! Arquivos salvos em:", out_dir.resolve())


if __name__ == "__main__":
    main()
