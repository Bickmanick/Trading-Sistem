# Legacy — Sistema anterior (KNN + Regimen + Thesis)

Este directorio contiene el sistema de trading original.
No se usa en el pipeline activo (`main.py`).

## Archivos

| Archivo | Descripcion |
|---|---|
| `main_legacy.py` | Orquestador del sistema viejo |
| `event_detector.py` | Deteccion de eventos sobre state matrix |
| `alignment_engine.py` | Calculo de regimen y alineacion de TFs |
| `thesis_engine.py` | Calculo de thesis long/short |
| `pattern_detector.py` | Deteccion de patrones KNN |
| `pattern_stats.py` | Estadisticas de patrones KNN |
| `movement_detector.py` | Detector de movimientos sin CHoCH |
| `trade_cycle.py` | Simulacion de ciclo de trades |
| `univariate_analysis.py` | Analisis univariante |
| `combo_analyzer.py` | Analisis de combinaciones |
| `state_snapshot.py` | Snapshot de estado |
| `statistics.py` | Estadisticas de trades |
| `main_analysis.py` | Script de analisis auxiliar |
| `run_step9.py` | Script puntual paso 9 |
| `run.bat` | Lanzador Windows |

## Para ejecutar el sistema legacy

```bash
python legacy/main_legacy.py NVDA 2024-01-01
```
