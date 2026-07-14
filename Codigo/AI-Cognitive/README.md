# Clasificador de carga cognitiva con EEG

Este proyecto mide, a partir de señales cerebrales (EEG), **cuánto está
trabajando mentalmente una persona** mientras hace una tarea. La idea de
fondo: si una computadora pudiera saber en tiempo real si estás
*relajado*, en tu *punto óptimo* o *sobrecargado*, podría adaptarse a ti
—por ejemplo, un tutor inteligente que baja la dificultad cuando te
satura, o un sistema de seguridad que avisa cuando un operador está
sobrepasado.

Este repositorio contiene el **código del análisis** y el **artículo
científico** (en `paper/`, versión inglés y español, listo para
Overleaf) que documenta todo lo que se probó.

---

## En una frase

Construimos una forma **honesta y realista** de estimar la carga mental
desde EEG: se adapta a cada persona con una calibración corta, se evalúa
sin hacer trampa (probando en personas que el modelo nunca vio), y usa
solo las señales cerebrales que la ciencia ya sabe que reflejan el
esfuerzo mental.

---

## Los datos

Usamos dos conjuntos públicos de EEG, para no depender de uno solo:

1. **Aritmética/Stroop** — 15 personas hacen tareas mentales
   (operaciones aritméticas y el test de Stroop) en cuatro niveles de
   dificultad. Se grabó con una **diadema inalámbrica de 8 canales tipo
   AURA** (similar al sistema AURA), en las posiciones estándar
   Fp1, Fp2, F3, **Fz**, F4, P3, **Pz**, P4.
2. **eegmat (PhysioNet)** — 36 personas hacen restas mentales; grabado
   con un equipo distinto (19 canales). Sirve para comprobar que el
   método **también funciona en otro hardware**.

---

## Cómo se hizo el análisis (explicado paso a paso)

1. **Cortar la señal en ventanas de 4 segundos.** El EEG es continuo; lo
   partimos en trozos cortos para analizar cada uno por separado (con
   50 % de solapamiento, uno cada 2 s).

2. **Limpiar cada ventana.** Se filtra la señal (1–40 Hz), se quita el
   ruido de la corriente eléctrica (50/60 Hz) y las derivas lentas. Esto
   deja solo la actividad cerebral útil.

3. **Extraer características con base fisiológica.** No inventamos
   medidas al azar: usamos las que la literatura de 30 años señala como
   marcadores de carga mental —sobre todo que la **theta en la frente
   sube** y la **alpha en la zona parietal baja** cuando la tarea cuesta
   más. Se calculan potencias por banda, el ratio θ/α, el índice de
   *engagement*, etc. **Quitamos** las medidas de asimetría frontal
   porque reflejan emoción, no carga mental (y verificamos que no
   aportaban nada).

4. **Calibrar a cada persona.** Cada cerebro es distinto. Antes de
   predecir, tomamos un tramo corto de *reposo* de la persona y medimos
   todo **en relación con su propio reposo**. Es como tarar una báscula
   a cero antes de pesar.

5. **Evaluar sin hacer trampa (LOSO).** Entrenamos con unas personas y
   probamos en **otra que el modelo nunca vio** (*leave-one-subject-out*),
   manteniendo separados los datos de calibración. Muchos trabajos
   reportan 95–99 % de acierto, pero mezclando datos; con evaluación
   honesta lo realista es 75–82 %, y ahí es donde caen nuestros números.

6. **Probar varios modelos y quedarnos con uno.** Comparamos Random
   Forest, gradient boosting, SVM, regresión logística, MLP y una red
   profunda (CNN-LSTM). El **Random Forest** fue el más fiable y es el
   que se conserva; los demás quedan solo mencionados.

7. **¿Cuántos electrodos hacen falta?** Probamos con 8, con 6 (frontal y
   parietal) y con solo 2 (Fz y Pz). Resultado: **con 6 canales se
   obtiene casi lo mismo que con 8**, e incluso con 2 se pierde poco. Es
   una buena noticia para un dispositivo barato y cómodo.

8. **Una métrica sencilla e interpretable (CLI).** Definimos un
   **Índice de Carga Cognitiva = theta(Fz) / alpha(Pz)**, normalizado al
   reposo de cada persona. Por sí solo ya sigue el nivel de carga casi
   tan bien como el modelo completo, y cualquiera puede entender qué
   significa.

9. **Tres niveles: baja / óptima / alta.** Reagrupamos las condiciones
   siguiendo la curva de Yerkes-Dodson (reposo → *baja*, tarea moderada
   → *óptima*, tarea difícil → *alta*).

---

## Qué encontramos

- **Distinguir reposo vs. tarea funciona bien** (≈0.82 en el primer
  conjunto, ≈0.76 en el segundo) y el método **transfiere entre equipos
  distintos**.
- **El nivel intermedio no se separa limpiamente.** Con estas
  características espectrales, la carga "media" se confunde con la alta.
  Lo comprobamos estadísticamente (no es un fallo del modelo, es un
  límite conocido del EEG) y por eso pasamos a una escala de 3 niveles y
  a una puntuación continua.
- **Menos electrodos bastan**, y hay **una métrica interpretable** que
  resume el resultado.

---

## Estructura del repositorio

```
AI-Cognitive/
├── paper/            # Artículo IEEE (main.tex inglés, main_es.tex español) + figuras
├── scripts/          # Todo el código del análisis
├── csv/              # Características por ventana y resultados intermedios
├── raw_data/         # Datos crudos (Arithmetic/Stroop, eegmat)
├── docs/             # Notas técnicas del proyecto
└── requirements.txt
```

Scripts principales (por función):

| Script | Qué hace |
|---|---|
| `eeg_convert_raw_to_clean.py` | Datos crudos → tabla limpia con etiquetas |
| `eeg_make_window_features_aura.py` | Ventanas de 4 s → características por ventana (datos tipo AURA) |
| `eeg_loso_calibrated.py` | Evaluación honesta LOSO + calibración por sujeto |
| `eeg_three_class_loso.py` | Clasificación en 3 niveles (baja/óptima/alta), compara 4 métodos |
| `eeg_channel_ablation.py` | Reducción de canales (8 vs 6 vs 2) + índice CLI θ(Fz)/α(Pz) |
| `eeg_ordinal_loso.py` | Puntuación ordinal (carga como escala continua) |
| `zyma_make_features.py` / `zyma_streaming_sweep.py` | Réplica en el conjunto eegmat |

## Cómo reproducir

```bash
pip install -r requirements.txt
# ablation de montaje + métrica interpretable:
python scripts/eeg_channel_ablation.py cli
python scripts/eeg_channel_ablation.py binary fp6
# clasificación en 3 niveles:
python scripts/eeg_three_class_loso.py rf
```

## Limitaciones y siguiente paso

- Pocas personas en el primer conjunto (15) y aún sin demo en hardware
  real.
- El nivel **alto** (sobrecarga) es el más difícil de aislar. El
  siguiente paso es probar características **no lineales** (entropía de
  muestra/permutación, coherencia entre canales) para recuperarlo.

## Dependencias

Python 3.9+, con numpy, pandas, scipy y scikit-learn (ver
`requirements.txt`).
