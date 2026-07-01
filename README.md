# Sistema de Análisis Multi-Timeframe

## Instalación
```bash
pip install -r requirements.txt
```

## Configuración
1. Pon tu API key de Polygon.io en `config.py` → `POLYGON_API_KEY`
2. Si ya tienes CSV de 1m, ponlos en `data/` con el nombre `SYMBOL_1m.csv`
   Columnas requeridas: `timestamp, open, high, low, close, volume`

## Uso
```bash
python main.py NVDA
python main.py AAPL 2024-01-01 2026-06-01
python main.py SPY
```

## Outputs generados en `output/SYMBOL/`
| Archivo | Contenido |
|---|---|
| `state_matrix.csv` | Estado completo de todos los indicadores en los 8 TFs por minuto |
| `univariate.csv` | Win rate individual de cada variable de evento |
| `patterns.csv` | Patrones detectados con estadísticas de resultado |
| `trades.csv` | Ciclo completo de trades con PnL, MAE, MFE |
| `by_regime.csv` | Estadísticas agrupadas por régimen de entrada |
| `by_quality.csv` | Estadísticas por calidad de tesis (ALTA/MEDIA/BAJA) |
| `by_direction.csv` | Win rate LONG vs SHORT |
| `by_exit_type.csv` | Estadísticas por tipo de salida |
| `patterns_ranked.csv` | Patrones rankeados por score (vecinos × win rate) |

## Módulos
| Módulo | Función |
|---|---|
| `config.py` | Parámetros centralizados |
| `downloader.py` | Descarga de datos Polygon.io |
| `resampler.py` | Construcción de 8 TFs (anti-lookahead) |
| `indicators.py` | MACD / Estocástico / Medias / Fibonacci |
| `state_engine.py` | Matriz de estados (merge_asof vectorizado) |
| `event_detector.py` | Detección de cambios 0→1 en variables binarias |
| `alignment_engine.py` | Sesgo por TF, cascadas, régimen |
| `thesis_engine.py` | Calidad de tesis LONG/SHORT (ALTA/MEDIA/BAJA) |
| `pattern_detector.py` | k-NN en espacio de estados binarios |
| `trade_cycle.py` | Simulación: entrada → tracking → salida → reversión |
| `univariate_analysis.py` | Ventaja estadística por variable individual |
| `statistics.py` | Resumen estadístico de trades y patrones |
| `main.py` | Orquestador principal |

## Regímenes del mercado
| Régimen | Significado |
|---|---|
| `IMPULSE_BULL/BEAR` | Cascada completa alineada — todos los TFs coherentes |
| `EXHAUST_BULL/BEAR` | Macro+context alineado pero phase/exec girando en contra |
| `FALSE_EXHAUST_BULL/BEAR` | Solo exec contrario, phase sigue alineado — posible ruido |
| `CONFLICT` | Macro y exec en direcciones opuestas |
| `INDECISION_MACRO` | Macro neutro, context alineado — zona gris |
| `RANGE` | Sin alineación clara en ningún TF |

## Calidad de Tesis
- **ALTA**: Cascada macro+context + ≥2 TFs con Stoch girando + confluencia Fib + régimen correcto
- **MEDIA**: Cascada macro+context + régimen correcto
- **BAJA**: Macro indeciso + context alineado (zona gris)
- **NONE**: Sin condiciones para operar
