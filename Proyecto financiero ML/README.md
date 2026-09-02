# Quantitative Investment Portfolio: Tail Risk Mitigation & Advanced Optimization

## 📋 Introducción del Proyecto

Este repositorio contiene el desarrollo completo del proyecto de investigación para la **Estancia de Investigación en Finanzas Cuantitativas** del Tecnológico de Monterrey (Periodo AD2026), bajo la supervisión del Prof. Dr. Jonathan Montalvo-Urquizo.

El objetivo central de este proyecto es el diseño, implementación y evaluación rigurosa de un portafolio de inversión cuantitativo de alta resiliencia. A diferencia de las estrategias tradicionales basadas exclusivamente en la maximización media-varianza de Markowitz, este enfoque implementa una tesis de **mitigación de riesgo de cola (Tail Risk Mitigation) y convexidad asimétrica**. El universo de inversión comprende 16 activos líquidos seleccionados estratégicamente bajo cuatro pilares fundamentales:

1. **Acciones de Crecimiento y Alta Liquidez:** Para capturar la apreciación secular del mercado.
2. **Consumo Básico (Consumer Staples):** Para proveer flujos de caja inelásticos y estabilidad operativa.
3. **Sector Salud y Farmacéuticas:** Para garantizar demanda desvinculada del ciclo macroeconómico.
4. **Instrumentos de Renta Fija y Cobertura (Safe-Havens):** Para activar mecanismos de protección (*flight-to-quality*) durante periodos de contracción sistémica.

Mediante el uso de herramientas avanzadas en Python, programación convexa (optimización de **\$\\text{CVaR}\$**), econometría de series de tiempo y un flujo de trabajo reproducible basado en Git, la estrategia busca superar el desempeño ajustado por riesgo del índice de referencia (*benchmark* S&P 500) minimizando las caídas máximas (*maximum drawdown*).









portfolio-optimization-ml/
│
├── .gitignore                          # Exclusión de entornos virtuales, cachés y datos locales pesados
├── README.md                           # Documentación principal del proyecto, objetivos y metodología
├── requirements.txt                    # Dependencias y librerías científicas del entorno de Python
├── environment.yml                     # Configuración para recreación del entorno Conda
│
├── data/                               # Almacenamiento local de datos financieros
│   ├── raw/                            # Datos originales descargados de yfinance (.parquet / .csv)
│   └── processed/                      # Matrices limpias de precios ajustados, retornos y variables
│
├── src/                                # Código fuente modular y reutilizable (.py)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_pipeline.py            # Ingestión, limpieza y cálculo de retornos logarítmicos
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── technical_indicators.py     # Ingeniería de factores técnicos (RSI, MACD, volatilidad)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── markowitz_benchmark.py      # Modelo clásico de media-varianza (Benchmark base)
│   │   ├── time_series_models.py       # Modelos econométricos (ARIMA / GARCH para volatilidad)
│   │   ├── clustering.py               # Agrupamiento no supervisado (K-Means / Jerárquico)
│   │   └── machine_learning.py         # Modelos de regresión regularizada y ensambles (XGBoost / LSTM)
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── cvar_optimizer.py           # Optimización convexa en CVXPY para minimización de riesgo de cola
│   │   └── constraints.py              # Restricciones de pesos, apalancamiento y rotación
│   │
│   └── backtesting/
│       ├── __init__.py
│       ├── engine.py                   # Motor de simulación fuera de muestra (Out-of-Sample)
│       ├── stress_testing.py           # Pruebas de estrés ante escenarios de crisis y tasas
│       └── transaction_costs.py        # Modelado de comisiones, slippage y restricciones de liquidez
│
├── notebooks/                          # Cuadernos interactivos para experimentación y entregables
│   ├── 01_eda_and_correlations.ipynb   # Análisis exploratorio, estacionalidad y matrices de correlación
│   ├── 02_feature_engineering.ipynb    # Generación y validación de variables cuantitativas
│   ├── 03_benchmark_markowitz.ipynb    # Frontera eficiente inicial y comparación vs S&P 500
│   ├── 04_econometric_volatility.ipynb # Pruebas ADF, diagnóstico y modelado GARCH
│   ├── 05_unsupervised_clusters.ipynb  # Clustering de activos y selección de centroides
│   ├── 06_supervised_learning.ipynb    # Entrenamiento y métricas de modelos predictivos
│   ├── 07_cvar_portfolio_opt.ipynb     # Resolución del problema de optimización convexa
│   ├── 08_backtesting_evaluation.ipynb # Evaluación de curvas de equity, Sharpe, Sortino y Alpha/Beta
│   └── 09_stress_and_liquidity.ipynb   # Simulación de choques de mercado y costos netos
│
├── reports/                            # Artefactos finales y presentaciones ejecutivas
│   ├── figures/                        # Gráficos de alta resolución exportados
│   ├── draft_report.pdf                # Borrador del informe final de investigación
│   └── final_presentation_pitch.pdf    # Presentación ejecutiva tipo pitch de inversión
│
└── tests/                              # Pruebas unitarias para validación matemática del código
├── __init__.py
├── test_data_pipeline.py
└── test_optimization.py
