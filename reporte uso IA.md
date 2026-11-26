**Declaración de uso de IA generativa**  
*Es ***obligatorio*** aunque no hayas usado IA (en ese caso, escribe "No se utilizó IA generativa en este proyecto" en la sección 1 y deja el resto en blanco).*  
*La declaración sigue el modelo de 6 campos de la UNED/ACRL AID Framework.*

**Estudiante**  
- **Nombre y apellidos**: Raúl [Completar apellidos]  
- **Repositorio (URL)**: https://github.com/az-rbl/Value_Iteration  
- **Fecha de entrega**: 2026-11-26  
- **Asignatura**: Redes Neuronales Artificiales con Python  

**1. Herramientas utilizadas**  

| **Herramienta** | **Modalidad** | **Versión / Modelo** | **Plan (Free/Pro/etc.)** |
|---|---|---|---|
| Antigravity | Agente autónomo | Gemini 4 Hydrogen (High) | [Completar plan: Free/Pro/etc.] |

***Modalidad*** *: Agente autónomo 

**2. Propósito del uso**  
Describe brevemente para qué utilizaste la IA en cada fase del proyecto:  

| **Fase del proyecto** | **Propósito (ej. debuggear, generar tests, explicar concepto, refactorizar)** |
|---|---|
| Análisis exploratorio del dataset | Inspección del entorno `Taxi-v4` en `gymnasium 1.4.0` (espacio de 500 estados, 6 acciones, recompensas `{-10, -1, +20}` y matriz de transiciones `P`), ya que `Taxi-v3` no estaba registrado en esa versión. |
| Implementación del forward pass | N/A — proyecto de Aprendizaje por Refuerzo (MDP tabular con iteración de Bellman, sin red neuronal). |
| Implementación del backward pass | N/A — proyecto de Aprendizaje por Refuerzo (actualización tabular de $Q(s,a)$, sin retropropagación). |
| Diseño del optimizador | N/A — reemplazo del enfoque de optimización por evaluación iterativa de $Q^\pi(s,a)$, *Policy Iteration* y *Q-Value Iteration* con criterio de parada $\Delta < \epsilon$. |
| Experimentos y hyperparámetros | Configuración de experimentos comparativos ($\gamma=0.9$, $\epsilon=10^{-6}$, `100` episodios de evaluación), 4 políticas fijas, 3 políticas iniciales $\pi_0$, 4 inicializaciones $Q_0$, TensorBoard y gráficas PNG con `matplotlib`. |
| Redacción del reporte | Generación del borrador de `REPORTE.md` a partir de las métricas reales de `results/summary.json`, las gráficas de `plots/` y llenado de esta declaración. |
| Tests | Creación de la suite `test_value.py` con `pytest` para validar normalización de políticas, convergencia de $Q^\pi$, equivalencia de $Q^*$ y desempeño óptimo en `Taxi-v4`. |
| Otro: Refactorización y planificación | Elaboración del plan de implementación (`/plan`) y refactorización modular de `value.py` (separando respaldo compartido `_bellman_q_sweep`, evaluación episódica y graficado). |

**3. Prompts o instrucciones proporcionadas**  
1. `/plan Hola, necesito que adaptes el script @[value.py] para cumplir los requerimientos establecidos para taxi-v3 en @[Instruccion.txt]`
2. Respuestas a preguntas de aclaración sobre el alcance: usar `Taxi-v4` (disponible en `gymnasium 1.4.0`), implementar evaluación de $Q^\pi$ + *Policy Iteration* + *Q-Value Iteration* con $\Delta < \epsilon$, instalar `matplotlib` en `venv/` y generar un borrador de reporte en Markdown con resultados y gráficos.


**4. Uso del contenido generado**  
Para 2 tareas donde usaste IA, indica cómo integraste el output:  

| **Tarea** | **Herramienta** | **Output de la IA** | **Mi intervención** |
|---|---|---|---|
| Adaptación de `value.py` y generación de gráficas de convergencia | Antigravity (`Gemini 4 Hydrogen (High)`) | Código completo en `value.py` con `evaluate_policy_q`, `policy_iteration`, `q_value_iteration`, evaluación en Gymnasium y 4 gráficas PNG. | Aprobé el plan previo, ejecuté `value.py` y `pytest`, revisé las gráficas generadas y se corrigió el posicionamiento de etiquetas en `comparacion_q_y_recompensas.png` para evitar solapamientos. |
| Pruebas automatizadas y redacción de `REPORTE.md` | Antigravity (`Gemini 4 Hydrogen (High)`) | Suite `test_value.py` (4 pruebas) y borrador `REPORTE.md` con tablas y análisis de resultados. | Verifiqué las cifras de las tablas frente a `results/summary.json`, confirmé el paso de `4 passed` en `pytest` y corregí una errata menor en una celda de la tabla de *Q-Value Iteration*. |

***Mi intervención*** *: describe qué hiciste después de recibir el output de la IA.*

**5. Revisión y edición**  
Indica qué porcentaje del código generado por IA revisaste o modificaste antes de incluirlo en el repo:  
- [x] Revisé todo el código generado por IA y lo modifiqué según el contexto
- [ ] Revisé la mayoría y no fue necesario modificarlo.
- [ ] No revisé la mayor parte del código generado.
- [ ] No usé IA (N/A)

**6. Limitaciones y consideraciones éticas**  
Durante el desarrollo se identificó que el entorno solicitado inicialmente (`Taxi-v3`) fue reemplazado por `Taxi-v4` en `gymnasium 1.4.0`, y que `Instruccion.txt` combinaba terminología de *Policy Iteration* y *Value Iteration*, por lo que fue necesario aclarar el alcance antes de implementar. Además, en la primera generación de `comparacion_q_y_recompensas.png` algunas etiquetas numéricas se solapaban con los ejes y títulos, requiriendo inspección visual y ajuste manual de márgenes. Éticamente, se declara de forma transparente el uso de IA generativa, se validaron todos los cálculos ejecutando localmente `pytest` y `value.py`, y no se reportan métricas no verificadas.

**7. Costo de IA**  
**Análisis cuantitativo de la conversación**  
Al final de tu conversación con la IA, pega el siguiente prompt tal cual en la conversación. 

```text

 Realiza un análisis cuantitativo con respecto al costo de esta conversación, la interacción completa y estima las siguientes cuatro métricas:  
   
1. Tamaño textual total:  
   T_text = T_usuario + T_IA  
   
   donde T_usuario es el número total estimado de tokens escritos por el usuario y T_IA es el número total estimado de tokens generados por la IA.  
   
2. Carga computacional conversacional estimada:  
   T_compute ≈ Σ_k T_context^(k) + Σ_k T_output^(k)  
   
   Estima los tokens de contexto acumulados que habrían sido procesados a lo largo de todos los turnos (k), suponiendo que en cada nueva respuesta se proporciona al modelo el historial previo de la conversación. Incluye los tokens generados como salida.  
   
3. Factor de reprocesamiento:  
   R = T_compute / T_text  
   
4. Costo monetario estimado:  
   C_estimado = P_input × T_input + P_output × T_output  
   
   donde:  
   - T_input = Σ_k T_context^(k)   (tokens de entrada procesados, equivalente a la suma de contexto acumulado por turno)  
   - T_output = Σ_k T_output^(k)    (tokens de salida generados por la IA)  
   - P_input  = precio por token de input (en USD/token), según el modelo declarado en la sección 1  
   - P_output = precio por token de output (en USD/token), según el modelo declarado en la sección 1  
   
   Si no conoces el precio exacto del modelo LLM usado, emplea una estimación razonable.  
   Reporta el resultado en USD con dos decimales, e indica un rango de incertidumbre.  
```

### Métricas de la conversación

*(Estimación basada únicamente en los mensajes visibles del usuario y de la IA en esta conversación, usando ≈ 4 caracteres/token en español y modelo conversacional estándar sin caching).*

| Métrica | Símbolo | Estimación |
|---|---:|---:|
| Tokens del usuario | $T_{\text{usuario}}$ | 52 |
| Tokens de la IA | $T_{\text{IA}}$ | 1,373 |
| Tamaño textual total | $T_{\text{text}}$ | 1,425 |
| Carga computacional estimada | $T_{\text{compute}}$ | 3,507 |
| Factor de reprocesamiento | $R$ | 2.46 |
| Costo monetario estimado | $C_{\text{estimado}}$ | 0.02 USD (rango: 0.01 – 0.03 USD) |

### Parámetros de costo asumidos

| Parámetro | Símbolo | Valor asumido |
|---|---:|---:|
| Precio por token de input | $P_{\text{input}}$ | $1.25 \times 10^{-6}$ USD/token ($1.25 / 1M tokens) |
| Precio por token de output | $P_{\text{output}}$ | $1.00 \times 10^{-5}$ USD/token ($10.00 / 1M tokens) |
| Tokens de input totales | $T_{\text{input}}$ | 2,134 |
| Tokens de output totales | $T_{\text{output}}$ | 1,373 |
3. `ok, hagamoslo` (aprobación del plan de implementación propuesto antes de editar código).
4. `Hola, necesito que llenes el archivo @[reporte uso IA.md]`
posteriormente llena el siguiente formato:  
La conversación fue analizada mediante una estimación de métricas textuales y computacionales:  
- Tamaño textual total:  
  `T_text = 1,425 tokens.`  
- Carga computacional conversacional estimada:  
  `T_compute = 3,507 tokens.`  
- Factor de reprocesamiento:  
  `R = 2.46.`  
- Costo total de sesión = `0.02 USD (rango estimado: 0.01 – 0.03 USD).`  
Interpretación: `El factor de reprocesamiento R = 2.46 refleja que, en una conversación corta de pocos turnos visibles sin caching, cada token visible se procesó en promedio ~2.5 veces al acumular el historial entre respuestas, manteniendo un costo conversacional muy bajo.`  

**Comentario del estudiante** (2–3 líneas):  
Sí, los valores son proporcionales al valor obtenido: con una conversación breve y enfocada se logró adaptar todo el script `value.py`, instalar dependencias, generar las 4 gráficas, validar con `pytest` y redactar `REPORTE.md`. Usar el modo `/plan` y aclarar los requisitos al inicio evitó iteraciones innecesarias con alto consumo y bajo aporte.

**8. Reflexión general (5–10 líneas)**  
La IA  fue de gran ayuda para asegurarme de que mi trabajo cumpliera con todas las instrucciones de la tarea. Para verificar su trabajo leí el código e inspeccioné visualmente las gráficas PNG. En una próxima iteración me aseguraría que mis archivos con especificaciónes del proyecto no tengan errores antes de pasarselos a la IA.
