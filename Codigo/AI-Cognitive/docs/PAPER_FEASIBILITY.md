# ¿Es publicable este trabajo? — análisis honesto

Estado: 2026-06-25 · Autor: Rafael · Programa doctoral PHD

## TL;DR

**Sí, es viable como paper corto de conferencia IEEE/IEEE EMBC**, pero
**no como journal de alto impacto** ni como contribución algorítmica
novedosa. Lo que hay vendible es **una metodología limpia y
reproducible de carga cognitiva en tiempo real con calibración por
sujeto**, validada con LOSO honesto sobre un dataset público.

## Fortalezas

1. **Evaluación honesta (LOSO + streaming reservado)**. Gran parte de la
   literatura EEG reporta accuracy con split por ventana o por archivo
   sin separar por sujeto; eso infla resultados. Aquí el protocolo es
   explícito y los números son creíbles.
2. **Ganancia cuantificada por calibración de baseline**:
   - +8 puntos de accuracy y +28 puntos de recall en la clase minoritaria
     `normal` al pasar de LOSO crudo a LOSO + z-score con baseline propio.
   - Es un mensaje práctico y replicable para cualquier sistema BCI real.
3. **Arquitectura jerárquica + EMA** alineada con la operación real
   (calibrar 30 s, predecir cada 2 s). Latencia de 1.6 ms por inferencia
   demostrada.
4. **Resultado negativo reportado**: la clase `low` colapsa
   estructuralmente. Aporta valor metodológico al evitar que otros
   gasten tiempo en la misma trampa.
5. **Pipeline open-source y reproducible** (un comando por etapa).

## Debilidades para revisión por pares

| Problema | Impacto | Mitigación |
|---|---|---|
| N = 15 sujetos | Crítico para journal, aceptable para conferencia corta | Reportar intervalos de confianza, no ocultar varianza |
| Dataset público, no propio | Limita novedad de datos | Vender como benchmark replicable; comparar con literatura previa sobre el mismo dataset |
| Ganancia viene de metodología, no de algoritmo nuevo | Reduce el "wow factor" | Posicionar como **estudio de protocolo**, no como nuevo clasificador |
| Alta varianza intersujeto (acc 0.54 a 0.97) | Reviewer crítico la verá | Diagnóstico explícito + propuesta de recalibración online como future work |
| `low` recall = 0 | Anti-resultado fuerte | Reportarlo y argumentar por qué binario `normal vs task` es el problema bien planteado |
| Sin demo en hardware real | Limita claims de "real-time" | Llamarlo "real-time-ready simulation" y separar latencia computacional de latencia E2E |

## Posicionamiento sugerido

> **"Subject-calibrated hierarchical classification for real-time cognitive
> load estimation from EEG: a LOSO-validated pipeline on the
> Arithmetic/Stroop dataset"**

Mensajes principales (en orden):

1. La calibración por sujeto contra una baseline `natural` corta es la
   palanca dominante: +8 pts binario, +28 pts recall normal.
2. Una arquitectura jerárquica de dos etapas + EMA logra **81 % de
   accuracy binario con 1.6 ms de inferencia** en streaming honesto.
3. La separación de tres niveles de carga *con features espectrales
   estándar* no es viable: el nivel intermedio (`low`) colapsa. Línea
   abierta para futuros trabajos con regresión ordinal o features
   no lineales.

## Targets de conferencia razonables

| Venue | Tipo | Tasa de aceptación aprox | Comentario |
|---|---|---|---|
| **IEEE EMBC** | Conferencia engineering in medicine | ~55 % | Casa natural para este tema. Paper corto (4 págs). |
| **IEEE SMC** | Systems, man, cybernetics | ~45 % | Encaja por el lado human-machine interaction |
| **BCI Society Meeting** | Conferencia BCI | — | Si lo enmarcas como BCI/neuroergonomía |
| **IEEE Sensors** | Si añades algo de hardware | ~45 % | Solo si haces la demo con un Aura/Muse real |
| **Frontiers in Neuroergonomics** | Journal Q2 | open access | Si añades 2-3 sujetos más y la recalibración online |

**Recomendación primaria**: EMBC 2027 (deadline típico febrero).
Permite paper de 4 páginas y acepta estudios metodológicos con
N pequeño si están bien justificados.

## Qué falta antes de redactar

| Bloque | Esfuerzo | Beneficio |
|---|---|---|
| Recalibración online (sliding baseline) | 2-3 días | Resuelve el problema de subj 2/5/10 → mensaje más fuerte |
| LightGBM + tamaño/latencia del modelo | 1 día | Argumento "deployable" más sólido |
| Análisis de importancia de features | 1 día | Sección de interpretabilidad obligatoria en reviewer EMBC |
| Comparación contra 2 baselines de literatura | 2 días | Esencial para que el reviewer no diga "no hay state-of-the-art" |
| Demo opcional con Cognitive-load/electron + Aura | 3-4 días | Convierte el paper de "simulación" a "sistema funcional" |

Total estimado a paper enviable: **~2 semanas de trabajo enfocado**.

## Riesgos a vigilar

- Si EMBC pide un baseline reciente que use el mismo dataset y la
  diferencia con el nuestro es < 3 pts, el paper no pasa.
- Si los 15 sujetos del dataset están sesgados demográficamente y no
  podemos describir su demografía, reviewer crítico nos lo va a marcar.
- La asimetría log-α/β por pares opuestos sin montaje conocido es
  débil. Mejor quitarla o documentar el supuesto explícitamente.

## Próximo paso concreto

Decidir si:

- **(A)** Cerramos el experimento como está y redactamos **EMBC short
  paper** (objetivo: enviar antes de fin de año).
- **(B)** Invertimos 2 semanas más en **recalibración online + demo
  hardware** y apuntamos a **Frontiers in Neuroergonomics** como journal.

Mi recomendación: **(A) primero**, luego **(B)** como extensión.
