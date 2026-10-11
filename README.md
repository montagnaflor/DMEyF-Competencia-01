# Competencia 01 — Predicción de bajas, agosto 2021

DMEyF — Florencia Montagna

Entrega final: 11.036 clientes (ganancia pública **101,12**).

## Cómo replicar la entrega final

| archivo | qué es |
|---|---|
| `prediccion_bajas_202108.ipynb` | Todo el proceso, desde el CSV de la competencia hasta el archivo de entrega |
| `orden_filas.csv.gz` | Orden de las filas de la corrida original (ver abajo) |

1. Copiar el dataset de la competencia a `data/competencia_01.csv` (al lado de la notebook).
2. Correr la notebook completa (~5 h en una PC de 8 núcleos y 8 GB de RAM). Por ejemplo:
   ```
   jupyter nbconvert --to notebook --execute prediccion_bajas_202108.ipynb --output prediccion_bajas_202108_ejecutada.ipynb --ExecutePreprocessor.timeout=-1
   ```
   En Google Colab: subir `competencia_01.csv` a `MyDrive/dmeyf/data/` y `orden_filas.csv.gz` a `MyDrive/dmeyf/`.
3. La entrega queda en `exp/final_202108/entrega01_202108.csv` y es idéntica al archivo entregado.

Versiones usadas: Python 3.13, numpy 2.4.3, pandas 3.0.1, pyarrow 25.0.1, duckdb 1.5.5, lightgbm 4.7.0,
optuna 4.9.0, scikit-learn 1.9.0. Semilla primigenia: 14072024.

### Por qué está `orden_filas.csv.gz`

El undersampling sortea el 25% de los CONTINUA según el orden de las filas del dataset. La entrega final se
generó con un parquet cuyo orden no era determinístico (DuckDB escribe en paralelo). La notebook genera ahora el
parquet ordenado por `foto_mes, numero_de_cliente` y, si encuentra `orden_filas.csv.gz`, reordena las filas como
en la corrida original; así se obtiene exactamente la misma muestra y la misma entrega. El archivo tiene solo
`foto_mes` y `numero_de_cliente`: no contiene información de la clase. Sin él, la notebook corre igual y produce
una entrega equivalente (mismo método, otra muestra de CONTINUA).

Verificado: los valores del parquet regenerado desde el CSV son idénticos a los del original, los datos cargados
con el archivo de orden son idénticos fila por fila a los de la corrida original, y esa corrida, ejecutada desde
cero (incluida la optimización bayesiana), reproduce la entrega final exacta.

## Metodología

**Validación con la misma distancia que el problema**: entre el último mes con clase completa (junio) y agosto
hay 2 meses; 202107 no se usa (sus BAJA+2 no se conocen).

| meses | uso |
|---|---|
| 202103–202104 | optimización bayesiana (5-fold) |
| 202106 | holdout: elección del trial y del corte; nunca entra en la optimización |
| 202103–202106 | modelo final |
| 202108 | predicción |

1. **Feature engineering**: deltas de 1 y 2 meses, ratio contra la media de 3 meses y pendiente del propio cliente;
   ranking de montos dentro del mes (contra la inflación); flags de "apagó un producto".
2. **Target**: BAJA+1 y BAJA+2 como positivos para entrenar; la ganancia se mide solo con BAJA+2.
3. **Undersampling** del 25% de los CONTINUA, con la ganancia corregida por ese factor.
4. **Drift**: se excluyen variables con PSI > 0,25 entre junio y agosto (9, casi todas tendencias de payroll
   distorsionadas por el aguinaldo de junio).
5. **Datos anómalos**: en junio los BAJA+2 tienen préstamos personales igual que los CONTINUA (22% contra 8–12% en
   marzo–mayo). Esas variables se ponen en NaN solo en junio. La ganancia en junio pasó de $395M a $449M.
6. **Canaritos**: se descartan las variables que no superan al mejor de 20 canaritos (quedan 74).
7. **LightGBM** con optimización bayesiana (Optuna, 100 trials), métrica = ganancia suavizada, early stopping.
8. **Maldición del ganador**: los 5 mejores trials se evalúan en junio con 10 semillas; se elige por ganancia media
   (trial 58), no el mejor de la optimización.
9. **Corte por cantidad de envíos**, el que maximiza la ganancia media en junio (11.000), trasladado a agosto como
   proporción de la cartera (11.036).
10. **Modelo final**: marzo–junio, `min_data_in_leaf` escalado por la cantidad de filas, promedio de 20 semillas,
    `max_bin = 255` (31 en la optimización).

## Todos los experimentos

Junio: ganancia media de 10 semillas en el mejor corte. Público: ganancia en el leaderboard público (25% de agosto),
en millones. "20 semillas" = 20 modelos de una semilla subidos juntos (media ± desvío).

| versión | cambio | junio | envíos | público |
|---|---|---|---|---|
| v1 | base | $395,3M | 12.541 | 94,85 |
| v1 | 9 cortes de 9.000 a 16.000 | | varios | 93,12 ± 6,02 |
| v1 | trial 62, 20 semillas | | 12.541 | 93,75 ± 1,69 |
| v1 | trial 61, 20 semillas | | 12.541 | 91,67 ± 2,24 |
| v1 | menos envíos | | 11.537 | 98,07 |
| v2 | sin canaritos (236 variables) | $397,2M | 11.537 | 98,23 |
| v2 | más envíos | | 12.541 | 95,98 |
| **v3** | **v1 + préstamos de junio en NaN** | **$449,0M** | 11.036 | **98,86** |
| v3 | más envíos | | 11.537 | 96,61 |
| v3 | 20 semillas | | 11.036 | 96,25 ± 2,14 |
| v4 | v3 + nulos (no 0/1) cuando no hay mes anterior | $435,9M | 11.036 | 94,13 |
| v5 | v3 con un modelo por mes, ensamblados | $444,1M | 11.036 | 90,97 |
| v6 | v3 + tendencias de payroll robustas al aguinaldo | $446,9M | 11.036 | 98,78 |
| v6 | 20 semillas | | 11.036 | 96,68 ± 2,01 |
| v7 | promedio de rankings v3 + v6 | | 11.036 | 99,14 |
| **final** | **v3 + `max_bin = 255` en el modelo final** | **$450,8M** | 11.036 | **101,12** |
| final | 20 semillas | | 11.036 | 96,76 ± 2,09 |
| v9 | v3 + optimización con validación temporal (marzo → mayo) | $451,4M | 11.036 | 92,32 |
| — | final + todos los CONTINUA en el modelo final | (+6,4M) | 11.036 | 95,56 |
| — | ídem, 20 semillas | | 11.036 | 91,53 ± 2,45 |
| — | misma notebook con otra muestra de CONTINUA (parquet ordenado) | $442,9M | 11.537 | 92,07 |

**Conclusiones**

* Lo que más movió la ganancia: neutralizar los préstamos de junio y el corte (con los mismos modelos, bajar de
  ~12.500 a ~11.000 envíos sumó 2–3M en el público).
* Junio predijo bien el efecto de los cambios en datos y variables (v3, v4, v5, `max_bin`), pero no el de los cambios
  en cómo se entrena el modelo final (v9, todos los CONTINUA): junio evalúa un modelo de marzo–abril, no uno de
  marzo–junio.
* El ensamble de 20 semillas suma ~2,5M sobre la semilla promedio.
* La variabilidad de una corrida completa (otra muestra de CONTINUA) es grande: el mismo método dio 101,12 y 92,07
  en el público.
