"""
src/models/markowitz_benchmark.py
===================================
Modelo base de optimización media-varianza de Markowitz.

Implementa la frontera eficiente clásica como baseline cuantitativo
para ser superado por modelos de ML con optimización CVaR.

Portafolios calculados
----------------------
1. **Máximo Sharpe Ratio** (MSR)  — portafolio tangente al CML.
2. **Mínima Varianza Global** (GMV) — punto más a la izquierda de la frontera.
3. **Igual Ponderación** (EW)     — naive baseline 1/N.

Estimación de covarianza
------------------------
Se usa **Ledoit-Wolf shrinkage** (implementado en scikit-learn) para
estabilizar la matriz de covarianza cuando N_activos es grande
relativo al período de estimación.

Métricas de evaluación
----------------------
Sharpe, Sortino, Max Drawdown, CVaR 95%, Beta, Información Ratio vs SPY.
Evaluación in-sample (2015-2020) y out-of-sample (2021-2025).

Autor: Equipo de Finanzas Cuantitativas – Tecnológico de Monterrey, AD2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TRADING_DAYS: int = 252
RF_ANNUAL: float = 0.0525        # Tasa libre de riesgo anual (aprox. T-bill 2024)
RF_DAILY: float = RF_ANNUAL / TRADING_DAYS
BENCHMARK: str = "SPY"

# Universo representativo (mismo que en processed.ipynb / technical_indicators.py)
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

SECTOR_LOOKUP: dict[str, str] = {
    t: s for s, tickers in REPRES.items() for t in tickers
}

ALL_TICKERS: List[str] = [t for tickers in REPRES.values() for t in tickers]


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Resultado de un portafolio optimizado."""
    name: str
    weights: pd.Series                      # ticker → peso
    ret_annual: float                       # retorno esperado anualizado
    vol_annual: float                       # volatilidad anualizada
    sharpe: float                           # Sharpe (con rf)
    # Métricas calculadas fuera de muestra si se pasa una serie de retornos OOS
    metrics_oos: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PortfolioResult('{self.name}' | "
            f"ret={self.ret_annual:.2%}, vol={self.vol_annual:.2%}, "
            f"sharpe={self.sharpe:.3f})"
        )


# ---------------------------------------------------------------------------
# Funciones de métricas de evaluación
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    weights: pd.Series,
    log_returns: pd.DataFrame,
) -> pd.Series:
    """
    Retornos diarios de un portafolio dados sus pesos estáticos.

    r_p = Σ w_i · r_i   (suma ponderada de log-retornos)
    """
    tickers = [t for t in weights.index if t in log_returns.columns]
    w = weights[tickers]
    w = w / w.sum()  # re-normalizar por si hay activos faltantes
    return log_returns[tickers].dot(w)


def annualized_return(ret_series: pd.Series) -> float:
    """Retorno anualizado desde retornos diarios."""
    return float(ret_series.mean() * TRADING_DAYS)


def annualized_vol(ret_series: pd.Series) -> float:
    """Volatilidad anualizada desde retornos diarios."""
    return float(ret_series.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(ret_series: pd.Series, rf: float = RF_ANNUAL) -> float:
    """Sharpe ratio anualizado."""
    mu = annualized_return(ret_series)
    sigma = annualized_vol(ret_series)
    return (mu - rf) / sigma if sigma > 0 else 0.0


def sortino_ratio(ret_series: pd.Series, rf: float = RF_ANNUAL) -> float:
    """Sortino ratio: usa sólo la desviación negativa (downside deviation)."""
    mu = annualized_return(ret_series)
    downside = ret_series[ret_series < RF_DAILY]
    dd = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 1e-8
    return (mu - rf) / dd if dd > 0 else 0.0


def max_drawdown(ret_series: pd.Series) -> float:
    """Maximum Drawdown (valor negativo, e.g. -0.35 = -35%)."""
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    return float(drawdowns.min())


def cvar_historical(
    ret_series: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    CVaR histórico (Expected Shortfall) al nivel alpha.

    Retorna el valor esperado de las pérdidas en el peor alpha% de días.
    """
    var = ret_series.quantile(alpha)
    return float(ret_series[ret_series <= var].mean())


def information_ratio(
    portfolio_ret: pd.Series,
    benchmark_ret: pd.Series,
) -> float:
    """
    Information Ratio = α / TE donde:
    - α = retorno activo promedio (portfolio − benchmark)
    - TE = tracking error (std del retorno activo)
    """
    active = portfolio_ret - benchmark_ret
    te = active.std() * np.sqrt(TRADING_DAYS)
    ir = (active.mean() * TRADING_DAYS) / te if te > 0 else 0.0
    return float(ir)


def beta_vs_benchmark(
    portfolio_ret: pd.Series,
    benchmark_ret: pd.Series,
) -> float:
    """Beta del portafolio respecto al benchmark."""
    cov = np.cov(portfolio_ret.dropna(), benchmark_ret.dropna())
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0


def full_metrics(
    portfolio_ret: pd.Series,
    benchmark_ret: pd.Series,
    label: str = "Portfolio",
    rf: float = RF_ANNUAL,
) -> Dict[str, float]:
    """
    Calcula el conjunto completo de métricas de evaluación.

    Returns
    -------
    dict con claves:
        ret_annual, vol_annual, sharpe, sortino, max_drawdown,
        cvar_95, beta, information_ratio, calmar
    """
    mdd = max_drawdown(portfolio_ret)
    ret_ann = annualized_return(portfolio_ret)
    vol_ann = annualized_vol(portfolio_ret)
    calmar = ret_ann / abs(mdd) if mdd != 0 else np.nan

    return {
        "ret_annual":        round(ret_ann, 4),
        "vol_annual":        round(vol_ann, 4),
        "sharpe":            round(sharpe_ratio(portfolio_ret, rf), 4),
        "sortino":           round(sortino_ratio(portfolio_ret, rf), 4),
        "max_drawdown":      round(mdd, 4),
        "cvar_95":           round(cvar_historical(portfolio_ret, 0.05), 6),
        "beta":              round(beta_vs_benchmark(portfolio_ret, benchmark_ret), 4),
        "information_ratio": round(information_ratio(portfolio_ret, benchmark_ret), 4),
        "calmar":            round(calmar, 4),
    }


# ---------------------------------------------------------------------------
# Estimación de parámetros estadísticos
# ---------------------------------------------------------------------------

def estimate_params(
    log_returns: pd.DataFrame,
    shrinkage: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Estima vector de retornos esperados y matriz de covarianza.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Log-retornos diarios.
    shrinkage : bool
        Si True usa Ledoit-Wolf shrinkage; si False usa covarianza muestral.

    Returns
    -------
    mu : np.ndarray, shape (N,)
        Retornos esperados anualizados.
    cov : np.ndarray, shape (N, N)
        Matriz de covarianza anualizada.
    tickers : list of str
        Tickers en el mismo orden que mu y cov.
    """
    tickers = list(log_returns.columns)
    ret_daily = log_returns.dropna(how="all").fillna(0.0)

    # Vector de medias anualizadas
    mu = ret_daily.mean().values * TRADING_DAYS

    # Matriz de covarianza anualizada
    if shrinkage:
        lw = LedoitWolf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lw.fit(ret_daily.values)
        cov = lw.covariance_ * TRADING_DAYS
    else:
        cov = ret_daily.cov().values * TRADING_DAYS

    return mu, cov, tickers


# ---------------------------------------------------------------------------
# Clase principal de optimización
# ---------------------------------------------------------------------------

class MarkowitzOptimizer:
    """
    Optimizador de portafolio Media-Varianza de Markowitz.

    Parameters
    ----------
    prices : pd.DataFrame
        Precios ajustados de cierre. Debe incluir el benchmark (SPY).
    tickers : list of str, optional
        Subconjunto de tickers a incluir en el portafolio.
        Por defecto usa ALL_TICKERS (sin el benchmark).
    train_end : str, optional
        Fecha de corte para división in-sample / OOS.
        Por defecto '2022-12-31'.
    weight_bounds : tuple, optional
        (min_w, max_w) por activo. Defecto (0.0, 0.25) — sin short selling,
        máx 25% en cualquier activo.
    shrinkage : bool
        Si True aplica Ledoit-Wolf shrinkage en la estimación de covarianza.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        tickers: Optional[Sequence[str]] = None,
        train_end: str = "2022-12-31",
        weight_bounds: Tuple[float, float] = (0.0, 0.25),
        shrinkage: bool = True,
        rf: float = RF_ANNUAL,
    ) -> None:
        self.prices = prices.copy().sort_index()
        self.tickers = list(tickers) if tickers else ALL_TICKERS
        # Filtrar tickers disponibles en el dataset
        self.tickers = [t for t in self.tickers if t in self.prices.columns]
        self.train_end = pd.Timestamp(train_end)
        self.weight_bounds = weight_bounds
        self.shrinkage = shrinkage
        self.rf = rf

        # Log-retornos completos
        self._log_ret = np.log(self.prices / self.prices.shift(1)).dropna(how="all")

        # Splits
        self._ret_train = self._log_ret[self._log_ret.index <= self.train_end]
        self._ret_oos   = self._log_ret[self._log_ret.index >  self.train_end]

        # Estimación de parámetros (in-sample)
        self._mu, self._cov, _ = estimate_params(
            self._ret_train[self.tickers], shrinkage=self.shrinkage
        )
        self._n = len(self.tickers)

        # Resultados guardados
        self._results: Dict[str, PortfolioResult] = {}
        self._frontier: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Funciones objetivo y restricciones
    # ------------------------------------------------------------------

    def _portfolio_vol(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self._cov @ w))

    def _portfolio_ret(self, w: np.ndarray) -> float:
        return float(self._mu @ w)

    def _neg_sharpe(self, w: np.ndarray) -> float:
        """Negativo del Sharpe para minimización."""
        r = self._portfolio_ret(w)
        v = self._portfolio_vol(w)
        return -(r - self.rf) / v if v > 1e-10 else 1e10

    def _constraints(self) -> List[dict]:
        """Restricción: pesos suman 1."""
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _bounds(self) -> List[Tuple[float, float]]:
        return [self.weight_bounds] * self._n

    def _optimize(
        self,
        objective,
        w0: Optional[np.ndarray] = None,
        n_restarts: int = 5,
    ) -> np.ndarray:
        """
        Optimización con múltiples puntos de inicio para robustez.
        """
        if w0 is None:
            w0 = np.ones(self._n) / self._n

        best_result = None
        best_val = np.inf

        for seed in range(n_restarts):
            rng = np.random.default_rng(seed)
            if seed == 0:
                w_init = w0
            else:
                # Punto aleatorio en el simplex
                w_init = rng.dirichlet(np.ones(self._n))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    objective,
                    w_init,
                    method="SLSQP",
                    bounds=self._bounds(),
                    constraints=self._constraints(),
                    options={"maxiter": 1000, "ftol": 1e-12},
                )

            if res.success and res.fun < best_val:
                best_val = res.fun
                best_result = res

        if best_result is None:
            # Fallback a igual ponderación
            return np.ones(self._n) / self._n

        return best_result.x

    # ------------------------------------------------------------------
    # Portafolios base
    # ------------------------------------------------------------------

    def max_sharpe(self) -> PortfolioResult:
        """Portafolio de Máximo Sharpe Ratio."""
        w = self._optimize(self._neg_sharpe)
        result = self._build_result("Max Sharpe (MSR)", w)
        self._results["MSR"] = result
        return result

    def min_variance(self) -> PortfolioResult:
        """Portafolio de Mínima Varianza Global."""
        w = self._optimize(self._portfolio_vol)
        result = self._build_result("Min Varianza (GMV)", w)
        self._results["GMV"] = result
        return result

    def equal_weight(self) -> PortfolioResult:
        """Portafolio de Igual Ponderación 1/N."""
        w = np.ones(self._n) / self._n
        result = self._build_result("Igual Ponderación (EW)", w)
        self._results["EW"] = result
        return result

    def _build_result(self, name: str, w: np.ndarray) -> PortfolioResult:
        """Construye un PortfolioResult con métricas IS y OOS."""
        weights = pd.Series(w, index=self.tickers, name=name)

        ret_ann = self._portfolio_ret(w)
        vol_ann = self._portfolio_vol(w)
        sharpe = (ret_ann - self.rf) / vol_ann if vol_ann > 0 else 0.0

        result = PortfolioResult(
            name=name,
            weights=weights,
            ret_annual=ret_ann,
            vol_annual=vol_ann,
            sharpe=sharpe,
        )

        # Métricas OOS si hay datos fuera de muestra
        if len(self._ret_oos) > 10:
            p_ret_oos = compute_portfolio_returns(weights, self._ret_oos)
            bm_ret_oos = self._ret_oos.get(BENCHMARK, pd.Series(dtype=float))
            if len(bm_ret_oos) > 0:
                result.metrics_oos = full_metrics(
                    p_ret_oos, bm_ret_oos, label=name
                )

        return result

    # ------------------------------------------------------------------
    # Frontera eficiente
    # ------------------------------------------------------------------

    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """
        Calcula la frontera eficiente con n_points portafolios.

        Minimiza la varianza para cada nivel de retorno objetivo
        entre el retorno del GMV y el máximo retorno del universo.

        Returns
        -------
        pd.DataFrame con columnas: ret, vol, sharpe, weights (N columnas)
        """
        gmv_w = self._optimize(self._portfolio_vol)
        ret_min = self._portfolio_ret(gmv_w)
        ret_max = float(self._mu.max())

        target_rets = np.linspace(ret_min, ret_max, n_points)
        records = []

        for target_ret in target_rets:
            constraints = self._constraints() + [
                {
                    "type": "eq",
                    "fun": lambda w, r=target_ret: self._portfolio_ret(w) - r,
                }
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    self._portfolio_vol,
                    np.ones(self._n) / self._n,
                    method="SLSQP",
                    bounds=self._bounds(),
                    constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-10},
                )

            if res.success:
                w = res.x
                vol = self._portfolio_vol(w)
                ret = self._portfolio_ret(w)
                sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
                row = {"ret": ret, "vol": vol, "sharpe": sharpe}
                row.update(dict(zip(self.tickers, w)))
                records.append(row)

        self._frontier = pd.DataFrame(records)
        return self._frontier

    # ------------------------------------------------------------------
    # Análisis comparativo
    # ------------------------------------------------------------------

    def compare_all(self) -> pd.DataFrame:
        """
        Ejecuta los tres portafolios base y compara sus métricas IS y OOS.

        Returns
        -------
        pd.DataFrame con índice = portafolio y columnas = métricas.
        """
        results = {
            "MSR": self.max_sharpe(),
            "GMV": self.min_variance(),
            "EW":  self.equal_weight(),
        }

        # Benchmark SPY
        if BENCHMARK in self._ret_oos.columns:
            bm_oos = self._ret_oos[BENCHMARK]
            bm_is  = self._ret_train[BENCHMARK]
            spy_metrics_is  = full_metrics(bm_is,  bm_is,  label="SPY")
            spy_metrics_oos = full_metrics(bm_oos, bm_oos, label="SPY")
        else:
            spy_metrics_is = spy_metrics_oos = {}

        rows_is, rows_oos = {}, {}

        for key, res in results.items():
            # IS: retorno / vol / sharpe calculados con parámetros estimados
            rows_is[res.name] = {
                "ret_annual": res.ret_annual,
                "vol_annual": res.vol_annual,
                "sharpe":     res.sharpe,
            }
            if res.metrics_oos:
                rows_oos[res.name] = res.metrics_oos

        if spy_metrics_is:
            rows_is["SPY (Benchmark)"] = {
                "ret_annual": spy_metrics_is["ret_annual"],
                "vol_annual": spy_metrics_is["vol_annual"],
                "sharpe":     spy_metrics_is["sharpe"],
            }
        if spy_metrics_oos:
            rows_oos["SPY (Benchmark)"] = spy_metrics_oos

        df_is  = pd.DataFrame(rows_is).T.rename_axis("Portafolio")
        df_oos = pd.DataFrame(rows_oos).T.rename_axis("Portafolio")

        return df_is, df_oos

    def weights_table(self) -> pd.DataFrame:
        """
        Tabla de pesos de los tres portafolios base.
        Filas: tickers, Columnas: MSR / GMV / EW
        Incluye sector de cada activo.
        """
        if not self._results:
            self.compare_all()

        dfs = {}
        for key, res in self._results.items():
            dfs[res.name] = res.weights.rename(res.name)

        wt = pd.DataFrame(dfs)
        wt.index.name = "Ticker"
        wt["Sector"] = wt.index.map(lambda t: SECTOR_LOOKUP.get(t, "Benchmark"))
        wt = wt.sort_values("Sector")
        return wt

    # ------------------------------------------------------------------
    # Utilidades de persistencia
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str) -> None:
        """
        Guarda resultados en CSV.

        Archivos generados:
        - markowitz_weights.csv    — pesos de los tres portafolios
        - markowitz_metrics_oos.csv — métricas OOS comparativas
        - efficient_frontier.csv  — puntos de la frontera eficiente
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        df_is, df_oos = self.compare_all()
        wt = self.weights_table()

        wt.to_csv(os.path.join(output_dir, "markowitz_weights.csv"))
        df_oos.to_csv(os.path.join(output_dir, "markowitz_metrics_oos.csv"))
        df_is.to_csv(os.path.join(output_dir, "markowitz_metrics_is.csv"))

        if self._frontier is not None:
            self._frontier.to_csv(
                os.path.join(output_dir, "efficient_frontier.csv"), index=False
            )

        print(f"✓ Resultados guardados en '{output_dir}/'")


# ---------------------------------------------------------------------------
# Función de conveniencia para notebooks
# ---------------------------------------------------------------------------

def run_markowitz(
    csv_path: str,
    tickers: Optional[Sequence[str]] = None,
    train_end: str = "2022-12-31",
    weight_bounds: Tuple[float, float] = (0.0, 0.25),
    fecha_inicio: str = "2015-01-02",
    shrinkage: bool = True,
    n_frontier: int = 100,
) -> Tuple[MarkowitzOptimizer, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de Markowitz listo para usar en notebooks.

    Parameters
    ----------
    csv_path : str
        Ruta a `datos.csv`.
    tickers : list of str, optional
        Subconjunto de tickers. Usa ALL_TICKERS por defecto.
    train_end : str
        Fecha de corte IS / OOS.
    weight_bounds : tuple
        (min, max) de pesos por activo.
    fecha_inicio : str
        Fecha inicial del índice temporal.
    shrinkage : bool
        Activar Ledoit-Wolf.
    n_frontier : int
        Número de puntos en la frontera eficiente.

    Returns
    -------
    optimizer : MarkowitzOptimizer
    metrics_is : pd.DataFrame     — métricas in-sample
    metrics_oos : pd.DataFrame    — métricas out-of-sample
    frontier : pd.DataFrame       — puntos de la frontera eficiente
    """
    import pandas as pd

    # Cargar precios
    df_raw = pd.read_csv(csv_path)
    business_days = pd.bdate_range(
        start=pd.Timestamp(fecha_inicio), periods=len(df_raw)
    )
    df_raw.index = business_days
    df_raw.index.name = "Date"
    df_raw = df_raw.drop(columns=["promedio"], errors="ignore")

    # Filtrar tickers disponibles
    use_tickers = list(tickers) if tickers else ALL_TICKERS
    if BENCHMARK not in use_tickers:
        use_tickers = use_tickers + [BENCHMARK]
    available = [t for t in use_tickers if t in df_raw.columns]
    prices = df_raw[available]

    print(f"Universo cargado: {len(available)-1} activos + {BENCHMARK}")
    print(f"Período total : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Período IS    : hasta {train_end}")
    print(f"Período OOS   : {pd.Timestamp(train_end) + pd.Timedelta('1d')} en adelante")

    # Optimizador
    opt = MarkowitzOptimizer(
        prices=prices,
        tickers=[t for t in available if t != BENCHMARK],
        train_end=train_end,
        weight_bounds=weight_bounds,
        shrinkage=shrinkage,
    )

    print("\nCalculando portafolios base...")
    metrics_is, metrics_oos = opt.compare_all()

    print("Calculando frontera eficiente...")
    frontier = opt.efficient_frontier(n_points=n_frontier)

    print("✓ Optimización completada.\n")
    return opt, metrics_is, metrics_oos, frontier


# ---------------------------------------------------------------------------
# Ejecución directa (test rápido)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    CSV = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "datos.csv")
    )

    print("=" * 60)
    print("MODELO BASE MARKOWITZ — Test rápido")
    print("=" * 60)

    opt, metrics_is, metrics_oos, frontier = run_markowitz(CSV)

    print("\n--- Métricas In-Sample (estimadas) ---")
    print(metrics_is.to_string())

    print("\n--- Métricas Out-of-Sample (2023-2025) ---")
    print(metrics_oos.to_string())

    print(f"\n--- Frontera eficiente: {len(frontier)} puntos ---")
    print(frontier[["ret", "vol", "sharpe"]].describe().round(4).to_string())

    print("\n--- Pesos de los portafolios ---")
    print(opt.weights_table().to_string())
