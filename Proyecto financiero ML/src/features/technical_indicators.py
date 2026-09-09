"""
src/features/technical_indicators.py
=====================================
Ingeniería de variables para portafolio de acciones del S&P Composite 900.

Módulo autónomo: recibe un DataFrame de precios (fechas × tickers) y devuelve
un DataFrame multi-nivel con todas las features calculadas.

Features implementadas
----------------------
Grupo 1 – Retornos y momentum
    log_ret_1d, log_ret_5d, log_ret_21d, log_ret_63d
    mom_1m, mom_3m, mom_6m, mom_12m     (retorno 21/63/126/252 días excluyendo
                                          el último mes – convención estándar)

Grupo 2 – Volatilidad y riesgo
    vol_21d, vol_63d                     (std anualizada de log-retornos)
    beta_63d                             (β rolling respecto a SPY)
    sharpe_63d                           (Sharpe ratio rolling, rf = 0)

Grupo 3 – Indicadores técnicos
    rsi_14, rsi_21                       (RSI de Wilder)
    macd, macd_signal, macd_hist         (12/26/9 EMA)
    bb_pct                               (% posición en Bandas de Bollinger 20d)
    ema_ratio_20_50, ema_ratio_50_200    (ratios de EMAs para tendencia)

Grupo 4 – Normalización cross-sectional
    zscore_ret_1d                        (z-score diario entre activos)
    zscore_mom_3m

Autor: Equipo de Finanzas Cuantitativas – Tecnológico de Monterrey, AD2026
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

TRADING_DAYS_YEAR: int = 252

# Universo representativo extraído del EDA (processed.ipynb)
# 10 representantes por sector GICS × 4 sectores + SPY
REPRES: dict[str, List[str]] = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CSCO",
        "CRM",  "AMD",  "QCOM", "TXN",  "NOW",
    ],
    "Consumer Staples": [
        "PG",  "KO",   "PEP",  "WMT",  "COST",
        "MO",  "PM",   "CL",   "GIS",  "HRL",
    ],
    "Health Care": [
        "JNJ",  "MRK",  "LLY",  "ABBV", "PFE",
        "ABT",  "AMGN", "ISRG", "UNH",  "MDT",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD",   "MCD",  "NKE",
        "SBUX", "BKNG", "RCL",  "LOW",  "TJX",
    ],
}

BENCHMARK: str = "SPY"

# Todos los tickers representativos (40 acciones + SPY)
ALL_REPRES: List[str] = [t for tickers in REPRES.values() for t in tickers] + [BENCHMARK]


# ---------------------------------------------------------------------------
# Funciones auxiliares de bajo nivel
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """EMA ajustada (adjust=False, min_periods=span//2)."""
    return series.ewm(span=span, adjust=False, min_periods=span // 2).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI de Wilder.

    Parameters
    ----------
    series : pd.Series
        Serie de precios de cierre.
    period : int
        Ventana en días.

    Returns
    -------
    pd.Series
        RSI en [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score cross-sectional: normaliza por fecha (fila) en lugar de por ticker."""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    return df.subtract(mu, axis=0).divide(sigma, axis=0)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Calcula un conjunto completo de features técnicas y de momentum
    a partir de un DataFrame de precios ajustados.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame de precios de cierre ajustados.
        Índice: DatetimeIndex (días hábiles).
        Columnas: tickers (str).
    benchmark : str
        Ticker del benchmark para cálculo de beta. Debe estar en `prices`.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        benchmark: str = BENCHMARK,
    ) -> None:
        if benchmark not in prices.columns:
            raise ValueError(
                f"El benchmark '{benchmark}' no está en las columnas de prices. "
                f"Columnas disponibles: {list(prices.columns[:10])}..."
            )
        self.prices = prices.copy().sort_index()
        self.benchmark = benchmark
        self._log_ret: Optional[pd.DataFrame] = None  # caché

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def log_returns(self) -> pd.DataFrame:
        """Log-retornos diarios (caché)."""
        if self._log_ret is None:
            self._log_ret = np.log(self.prices / self.prices.shift(1))
        return self._log_ret

    @property
    def tickers(self) -> List[str]:
        """Lista de tickers en el dataset (sin el benchmark si aplica)."""
        return [c for c in self.prices.columns if c != self.benchmark]

    # ------------------------------------------------------------------
    # Grupo 1 – Retornos y Momentum
    # ------------------------------------------------------------------

    def _feature_returns(self) -> pd.DataFrame:
        """Retornos log para ventanas: 1d, 5d, 21d, 63d."""
        lr = self.log_returns
        frames: dict[str, pd.DataFrame] = {
            "log_ret_1d":  lr,
            "log_ret_5d":  lr.rolling(5).sum(),
            "log_ret_21d": lr.rolling(21).sum(),
            "log_ret_63d": lr.rolling(63).sum(),
        }
        return pd.concat(frames, axis=1)

    def _feature_momentum(self) -> pd.DataFrame:
        """
        Momentum de precio (convención 12-1): retorno total del período
        excluyendo el último mes para evitar mean-reversion de corto plazo.

        Ventanas: 1M(21d), 3M(63d), 6M(126d), 12M(252d)
        """
        p = self.prices
        # Retorno = precio actual / precio hace N días − 1 (usando log para consistencia)
        frames: dict[str, pd.DataFrame] = {
            "mom_1m":  np.log(p / p.shift(21)),
            "mom_3m":  np.log(p / p.shift(63)),
            "mom_6m":  np.log(p / p.shift(126)),
            "mom_12m": np.log(p / p.shift(252)),
        }
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Grupo 2 – Volatilidad y Riesgo
    # ------------------------------------------------------------------

    def _feature_volatility(self) -> pd.DataFrame:
        """Volatilidad rolling anualizada (21d y 63d)."""
        lr = self.log_returns
        ann = np.sqrt(TRADING_DAYS_YEAR)
        frames: dict[str, pd.DataFrame] = {
            "vol_21d": lr.rolling(21).std() * ann,
            "vol_63d": lr.rolling(63).std() * ann,
        }
        return pd.concat(frames, axis=1)

    def _feature_beta(self, window: int = 63) -> pd.DataFrame:
        """
        Beta rolling de cada activo respecto al benchmark.

        β_i = Cov(r_i, r_bm) / Var(r_bm)
        """
        lr = self.log_returns
        bm = lr[self.benchmark]
        tickers = [c for c in lr.columns if c != self.benchmark]

        betas: dict[str, pd.Series] = {}
        var_bm = bm.rolling(window).var()

        for t in tickers:
            cov = lr[t].rolling(window).cov(bm)
            betas[t] = (cov / var_bm).rename(t)

        beta_df = pd.DataFrame(betas, index=lr.index)
        # Incluir beta del benchmark = 1 por definición
        beta_df[self.benchmark] = 1.0

        return pd.concat({"beta_63d": beta_df}, axis=1)

    def _feature_rolling_sharpe(self, window: int = 63) -> pd.DataFrame:
        """Sharpe ratio rolling (rf = 0, anualizado)."""
        lr = self.log_returns
        ann = np.sqrt(TRADING_DAYS_YEAR)
        mu_r = lr.rolling(window).mean() * TRADING_DAYS_YEAR
        sig_r = lr.rolling(window).std() * ann
        sharpe = (mu_r / sig_r.replace(0, np.nan)).rename(
            columns={c: c for c in lr.columns}
        )
        return pd.concat({"sharpe_63d": sharpe}, axis=1)

    # ------------------------------------------------------------------
    # Grupo 3 – Indicadores Técnicos
    # ------------------------------------------------------------------

    def _feature_rsi(self) -> pd.DataFrame:
        """RSI de Wilder en ventanas 14 y 21 días."""
        frames: dict[str, pd.DataFrame] = {}
        for period in (14, 21):
            rsi_dict: dict[str, pd.Series] = {}
            for col in self.prices.columns:
                rsi_dict[col] = _rsi(self.prices[col], period)
            frames[f"rsi_{period}"] = pd.DataFrame(rsi_dict, index=self.prices.index)
        return pd.concat(frames, axis=1)

    def _feature_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD = EMA(fast) − EMA(slow), señal EMA(signal), histograma."""
        macd_dict: dict[str, pd.Series] = {}
        signal_dict: dict[str, pd.Series] = {}
        hist_dict: dict[str, pd.Series] = {}

        for col in self.prices.columns:
            p = self.prices[col]
            macd_line = _ema(p, fast) - _ema(p, slow)
            sig_line  = _ema(macd_line, signal)
            macd_dict[col]   = macd_line
            signal_dict[col] = sig_line
            hist_dict[col]   = macd_line - sig_line

        idx = self.prices.index
        frames: dict[str, pd.DataFrame] = {
            "macd":        pd.DataFrame(macd_dict, index=idx),
            "macd_signal": pd.DataFrame(signal_dict, index=idx),
            "macd_hist":   pd.DataFrame(hist_dict, index=idx),
        }
        return pd.concat(frames, axis=1)

    def _feature_bollinger(self, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
        """
        %B de Bandas de Bollinger.

        %B = (precio − BB_inf) / (BB_sup − BB_inf)
        0 → en la banda inferior, 0.5 → en la media, 1 → en la banda superior.
        """
        rolling_mean = self.prices.rolling(window).mean()
        rolling_std  = self.prices.rolling(window).std()
        bb_upper = rolling_mean + n_std * rolling_std
        bb_lower = rolling_mean - n_std * rolling_std
        band_width = (bb_upper - bb_lower).replace(0, np.nan)
        bb_pct = (self.prices - bb_lower) / band_width

        return pd.concat({"bb_pct": bb_pct}, axis=1)

    def _feature_ema_ratios(self) -> pd.DataFrame:
        """
        Ratios de EMAs para detectar tendencia.

        ema_ratio_20_50:  EMA(20) / EMA(50) − 1   (tendencia corto vs medio)
        ema_ratio_50_200: EMA(50) / EMA(200) − 1  (tendencia medio vs largo)
        """
        frames: dict[str, pd.DataFrame] = {}
        ema20  = self.prices.apply(lambda s: _ema(s, 20))
        ema50  = self.prices.apply(lambda s: _ema(s, 50))
        ema200 = self.prices.apply(lambda s: _ema(s, 200))

        frames["ema_ratio_20_50"]  = ema20 / ema50.replace(0, np.nan) - 1
        frames["ema_ratio_50_200"] = ema50 / ema200.replace(0, np.nan) - 1

        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Grupo 4 – Normalización Cross-Sectional
    # ------------------------------------------------------------------

    def _feature_zscore_cross(self) -> pd.DataFrame:
        """Z-score cross-sectional del retorno diario y momentum 3M."""
        lr = self.log_returns
        mom3m = np.log(self.prices / self.prices.shift(63))

        frames: dict[str, pd.DataFrame] = {
            "zscore_ret_1d": _zscore_cross_section(lr),
            "zscore_mom_3m": _zscore_cross_section(mom3m),
        }
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def compute_all(
        self,
        tickers_subset: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Calcula todas las features y devuelve un DataFrame con MultiIndex
        de columnas (feature_group, ticker).

        Parameters
        ----------
        tickers_subset : list of str, optional
            Si se proporciona, filtra las columnas del resultado.
            El benchmark siempre se incluye en los cálculos pero puede
            excluirse del resultado final.

        Returns
        -------
        pd.DataFrame
            Índice: DatetimeIndex.
            Columnas: MultiIndex (feature_name, ticker).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            parts = [
                self._feature_returns(),
                self._feature_momentum(),
                self._feature_volatility(),
                self._feature_beta(),
                self._feature_rolling_sharpe(),
                self._feature_rsi(),
                self._feature_macd(),
                self._feature_bollinger(),
                self._feature_ema_ratios(),
                self._feature_zscore_cross(),
            ]

        features = pd.concat(parts, axis=1)

        if tickers_subset is not None:
            # Mantener sólo los tickers solicitados en el segundo nivel del MultiIndex
            cols_keep = [
                col for col in features.columns
                if col[1] in tickers_subset
            ]
            features = features[cols_keep]

        return features

    def compute_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Devuelve un DataFrame con todas las features para un único ticker.
        Columnas: nombre de la feature.
        """
        all_feats = self.compute_all()
        # Seleccionar el segundo nivel del MultiIndex = ticker
        ticker_feats = all_feats.xs(ticker, axis=1, level=1)
        return ticker_feats

    def summary_stats(self, ticker: str) -> pd.DataFrame:
        """
        Estadísticas descriptivas de las features para un ticker dado.
        Útil para validación rápida.
        """
        return self.compute_for_ticker(ticker).describe().T


# ---------------------------------------------------------------------------
# Función de conveniencia para uso en notebooks
# ---------------------------------------------------------------------------

def load_prices(
    csv_path: str,
    fecha_inicio: str = "2015-01-02",
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Carga el dataset de precios desde el CSV del proyecto.

    Parameters
    ----------
    csv_path : str
        Ruta al archivo `datos.csv`.
    fecha_inicio : str
        Fecha de inicio para construir el índice de días hábiles.
    tickers : list of str, optional
        Si se proporciona, filtra las columnas al subconjunto solicitado.
        El benchmark (SPY) se agrega automáticamente si no está en la lista.

    Returns
    -------
    pd.DataFrame
        Precios ajustados con DatetimeIndex.
    """
    df_raw = pd.read_csv(csv_path)

    # Reconstruir índice temporal (días hábiles desde fecha_inicio)
    business_days = pd.bdate_range(
        start=pd.Timestamp(fecha_inicio),
        periods=len(df_raw),
    )
    df_raw.index = business_days
    df_raw.index.name = "Date"

    # Eliminar columna 'promedio' si existe
    df_raw = df_raw.drop(columns=["promedio"], errors="ignore")

    if tickers is not None:
        # Asegurar que SPY siempre esté
        cols = list(tickers)
        if BENCHMARK not in cols:
            cols.append(BENCHMARK)
        # Filtrar sólo los tickers disponibles
        cols_available = [c for c in cols if c in df_raw.columns]
        df_raw = df_raw[cols_available]

    return df_raw


def build_features(
    csv_path: str,
    tickers: Optional[Sequence[str]] = None,
    fecha_inicio: str = "2015-01-02",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo: carga precios → calcula features.

    Parameters
    ----------
    csv_path : str
        Ruta a `datos.csv`.
    tickers : list of str, optional
        Subconjunto de tickers (usa ALL_REPRES por defecto).
    fecha_inicio : str
        Fecha inicio del índice temporal.

    Returns
    -------
    prices : pd.DataFrame
        Precios ajustados.
    features : pd.DataFrame
        DataFrame de features con MultiIndex de columnas.
    """
    if tickers is None:
        tickers = ALL_REPRES

    prices = load_prices(csv_path, fecha_inicio=fecha_inicio, tickers=tickers)
    fe = FeatureEngineer(prices, benchmark=BENCHMARK)
    features = fe.compute_all(tickers_subset=tickers)
    return prices, features


# ---------------------------------------------------------------------------
# Ejecución directa (test rápido)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # Ruta relativa desde la raíz del proyecto
    CSV = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "datos.csv"
    )
    CSV = os.path.normpath(CSV)

    print("Cargando precios y calculando features...")
    prices, features = build_features(CSV, tickers=ALL_REPRES)

    print(f"\n✓ Precios cargados   : {prices.shape}")
    print(f"✓ Features calculadas: {features.shape}")
    print(f"  Período: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Features disponibles: {features.columns.get_level_values(0).unique().tolist()}")
    print(f"\nMuestra de features para NVDA:")
    fe = FeatureEngineer(prices)
    print(fe.compute_for_ticker("NVDA").tail(3).to_string())
