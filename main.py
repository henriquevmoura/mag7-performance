from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Busca dados do yfinance
def fetch_adj_close(tickers: List[str], years: int = 3) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:

    end = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[0]
    start = end - pd.DateOffset(years=years)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end+pd.Timedelta(days=1),
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
        raise ValueError("Sem interseção de datas com dados para todos os tickers")
    base = df_common.iloc[0]

    # Pega a primeira linha de dados comuns
    norm = (df_common / base) * 100.0
    return norm


# Gráfico de performance normalizada
def plot_performance(norm: pd.DataFrame, end: pd.Timestamp, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))

    for col in norm.columns:
        ax.plot(norm.index, norm[col], linewidth=2, label=col)

    ax.set_title(
        'Performance Normalizada Mag Seven - Base 100',
        fontsize=16, fontweight='bold', pad=12
    )

    ax.set_ylabel('Performance (Base 100)')
    ax.set_xlabel('Data')

    ax.axhline(100, linestyle='--', color='grey', linewidth=1)

    ax.legend(title='Tickers', loc='upper left', frameon=True)

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(norm.index.min(), norm.index.max())

    fig.text(
        0.01, -0.02,
        f"Fonte: Yahoo Finance. Data base {end.date()}",
        fontsize=9
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(12, 9))

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(
        'Matriz de Correlação Mag Seven – Retornos Diários (Últimos 3 Meses)',
        fontsize=16,
        fontweight='bold',
        pad=12
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(
        0.01, -0.02,
        f"Fonte: Yahoo Finance. Data base {end.date()}",
        fontsize=9
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path



# Adicional: Média dos elementos fora da diagonal da matriz de correlação
def _avg_offdiag(corr: pd.DataFrame) -> float:
    m = corr.to_numpy()
    n = m.shape[0]
    tri = np.triu_indices(n, k=1)
    return float(np.nanmean(m[tri]))


# Adicional: Rolling Average
def compute_rolling_avg_corr_3m(
    adj: pd.DataFrame,
    min_obs: int
) -> pd.Series:
    prices = adj.dropna(how="any")
    rets = prices.pct_change().dropna(how="any")

    vals = []
    idx = []

    for dt in rets.index:
        start = dt - pd.DateOffset(months=3)
        win = rets.loc[start:dt]
        if len(win) < min_obs:
            vals.append(np.nan)
        else:
            vals.append(_avg_offdiag(win.corr()))
        idx.append(dt)

    idx = rets.index

    s = pd.Series(vals, index=idx, name="avg_corr_3m")
    return s


# Adicional: Rolling Average Chart
def plot_rolling_avg_corr(series: pd.Series, end: pd.Timestamp, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(series.index, series.values, linewidth=2)
    ax.set_title("Correlação Média Mag Seven - Retornos Diários (Janela Móvel de 3 Meses)", fontsize=16, fontweight="bold", pad=12)
    ax.set_ylim(0.10, 0.90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax.set_xlim(series.index.min()+ pd.DateOffset(months=2), series.index.max())

    fig.text(
        0.01, -0.02,
        f"Série: média das correlações (retornos diários) com janela móvel de 3 meses. "
        f"Fonte: Yahoo Finance. Data base {end.date()}",
        fontsize=9
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Função principal
def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Baixando dados")
    adj, start, end = fetch_adj_close(TICKERS, years=3)
    print(f"Janela: {start.date()} a {end.date()} | Linhas: {len(adj)}")

    print("Normalizando em base=100")
    norm = normalize_base100(adj)

    perf_path = out_dir / "performance_normalized.png"
    print(f"Plotando performance {perf_path}")
    plot_performance(norm, end, perf_path)

    print("Calculando correlação (últimos 3 meses)")
    corr = compute_corr_last3m(adj, end)

    corr_path = out_dir / "corr_heatmap_last3m.png"
    print(f"Plotando heatmap {corr_path}")
    plot_corr_heatmap(corr, end, corr_path)

    print("Calculando rolling 3-month avg pairwise correlation")
    roll = compute_rolling_avg_corr_3m(adj, min_obs=40)
    roll_path = out_dir / "magnificent7_rolling_3m_avg_corr.png"
    print(f"Plotando série {roll_path}")
    plot_rolling_avg_corr(roll, end, roll_path)

    adj.to_csv(out_dir / "adj_close.csv", index=True)
    norm.to_csv(out_dir / "normalized_base100.csv", index=True)
    corr.to_csv(out_dir / "corr_last3m.csv", index=True)
    roll.to_csv(out_dir / "rolling_avg_corr_3m.csv", index=True)
    

if __name__ == "__main__":
    main()
