# Neurona artificial para el riego de plantas

Repositorio: https://github.com/Mans0089/neurona-riego-plantas

Proyecto de la actividad *Neurona artificial para el riego de plantas* (Universidad Santo Tomás, Tunja).

## Descripción del problema

Se construye una neurona artificial que decide si una planta necesita riego a partir de dos variables:

| Variable | Significado | Escala |
|---|---|---|
| x1 | Humedad del suelo | 0 a 100 % |
| x2 | Temperatura ambiental | 0 a 50 °C (aprox.) |
| y  | Salida | 1 = regar, 0 = no regar |

La neurona calcula `z = x1·w1 + x2·w2 + b`, aplica la función **sigmoide** para obtener una probabilidad entre 0 y 1 y la convierte en 0 o 1 con un umbral de 0.5. Se entrena con descenso de gradiente sobre el error cuadrático medio.

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Mans0089/neurona-riego-plantas.git
cd neurona-riego-plantas
uv sync
uv run main.py
```

## Estructura

```
neurona-riego-plantas/
├── main.py          # neurona, entrenamiento, pruebas y experimentos
├── pyproject.toml   # generado por uv
├── uv.lock          # generado por uv
└── README.md
```

## Normalización

La humedad llega hasta 100 y la temperatura hasta unos 50, así que cada columna se divide por su máximo esperado:

```python
escala = np.array([100, 50])
X_normalizado = X / escala
```

`X_normalizado` se usa en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente (`gradiente_pesos = X_normalizado.T @ gradiente_z`). El arreglo `X` original se conserva solo para mostrar los resultados en unidades reales. Los datos nuevos se dividen por **la misma** `escala`.

## Resultados del entrenamiento base (10000 épocas, tasa 0.5)

| Parámetro | Valor |
|---|---|
| Peso de la humedad (w1) | -19.5032 |
| Peso de la temperatura (w2) | 2.3101 |
| Sesgo (b) | 7.4022 |
| Error final (MSE) | 0.019645 |
| Aciertos | 10/10 |

| Caso | Humedad | Temperatura | Probabilidad | Respuesta | Esperado |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0006 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.0053 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.0183 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.1025 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.2951 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.7285 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.8496 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.9539 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.9941 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.9993 | 1 | 1 |

**Signo de los pesos:** el peso de la humedad es negativo, así que a mayor humedad baja la probabilidad de regar. El peso de la temperatura es positivo, así que a mayor temperatura sube la probabilidad de regar. La magnitud también importa: |w1| es unas 8 veces mayor que |w2|, así que la humedad es la variable que más pesa en la decisión.

## Pruebas con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 – No regar |
| 45 % | 34 °C | 0.5490 | 1 – Regar |
| 25 % | 22 °C | 0.9719 | 1 – Regar |
| 50 % | 25 °C | 0.2325 | 0 – No regar |
| 30 % | 40 °C | 0.9677 | 1 – Regar |

El caso 45 % / 34 °C queda muy cerca de 0.5. Tiene una humedad intermedia entre los ejemplos de "no regar" (≥ 50 %) y "regar" (≤ 40 %) y una temperatura alta, así que la neurona lo decide con poca seguridad.

## Experimentos con los parámetros

En cada prueba se cambió un solo parámetro y se usó la misma semilla (`default_rng(7)`) para los pesos iniciales, así los cambios se pueden atribuir al parámetro modificado.

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | Adecuado |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | Aceptable, incompleto |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | Mejora pequeña |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | Moderado |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | Muy rápido, estable |

Todos los experimentos aciertan los 10 casos porque los datos se separan con facilidad: existe una línea clara entre los casos de riego y los de no riego. Por eso **el error final** muestra mejor que los aciertos qué tan bien aprendió la neurona, porque mide qué tan cerca de 0 o 1 quedan las probabilidades.

## Prueba adicional con el umbral

Con los pesos del entrenamiento base, sin reentrenar:

| Humedad | Temperatura | Probabilidad | Umbral 0.4 | Umbral 0.5 | Umbral 0.6 |
|---|---|---|---|---|---|
| 45 % | 34 °C | 0.5490 | 1 | 1 | **0** |

Ese es el único caso que cambia (entre los 10 de entrenamiento y los 5 nuevos), porque es el único cuya probabilidad cae entre 0.4 y 0.6. Los demás están lejos de esa franja y mantienen su decisión.

Cambiar el umbral no modifica los pesos porque el umbral se aplica **después** del entrenamiento, solo para convertir la probabilidad en 0 o 1. Los pesos y el sesgo se ajustan con el gradiente del error, y en ese cálculo no interviene el umbral. Lo que cambia es la política de decisión: un umbral bajo (0.4) riega más a menudo y es más conservador con la planta, y un umbral alto (0.6) exige más seguridad antes de regar y ahorra agua.

## Análisis

**1. ¿Por qué fue necesario normalizar?**
La humedad (hasta 100) y la temperatura (hasta 50) tienen escalas distintas. Sin normalizar, la variable con valores más grandes produciría gradientes más grandes y dominaría la actualización de los pesos. Además, valores de z muy grandes saturan la sigmoide, su derivada se vuelve casi 0 y el aprendizaje se frena. Al dejar ambas entradas cerca de [0, 1], las dos variables contribuyen de forma comparable y el entrenamiento es más estable.

**2. ¿Dónde se usó X_normalizado y para qué se conservó X?**
`X_normalizado` se usó en la suma ponderada y en el gradiente de los pesos. `X` se conservó para mostrar los resultados en unidades reales (% y °C), que son más fáciles de interpretar. Los datos nuevos se normalizan con la misma `escala`, porque los pesos aprendidos solo tienen sentido para entradas en esa escala.

**3. ¿Qué ocurrió con 100 épocas?**
La neurona clasificó bien los 10 casos, pero el error quedó alto (0.166 frente a 0.020 en la prueba base). Las probabilidades quedaron cerca de 0.5, así que acertó "por poco" y sin seguridad. El aprendizaje fue insuficiente.

**4. ¿Más épocas siempre mejoraron mucho?**
No. De 100 a 1000 y de 1000 a 10000 épocas el error bajó bastante, pero de 10000 a 20000 solo pasó de 0.0196 a 0.0117 y los aciertos no cambiaron. Llega un punto de rendimientos decrecientes: más épocas cuestan más cómputo y aportan poco. Sobre-entrenar con pocos datos también puede volver la neurona demasiado confiada en regiones donde no vio ejemplos.

**5. ¿Qué efecto tuvo una tasa demasiado pequeña?**
Con 0.01 los pasos son tan pequeños que en 10000 épocas el error solo llegó a 0.130, parecido a entrenar 100 épocas con tasa 0.5. El aprendizaje es lento: iría en la dirección correcta, pero necesitaría muchas más épocas.

**6. ¿Qué efecto tuvo una tasa alta o muy alta?**
En este problema, las tasas 1.0 y 2.0 aprendieron más rápido y lograron el menor error, sin oscilaciones: el error bajó en todas las épocas. Algo notable es que la tasa 1.0 con 10000 épocas da exactamente el mismo resultado que la tasa 0.5 con 20000 épocas, porque dar pasos del doble de tamaño equivale a dar el doble de pasos (mientras el entrenamiento sea estable). No apareció inestabilidad porque el problema es sencillo y la derivada de la sigmoide (máximo 0.25) amortigua los gradientes. Aun así, en general una tasa demasiado alta puede hacer que los pesos salten por encima del mínimo y el error oscile o diverja, sobre todo con datos más complejos o sin normalizar.

**7. ¿Qué representa el signo del peso de la humedad?**
Es negativo: cuanto más húmedo está el suelo, menor es la probabilidad de regar. Esto coincide con la intuición: si el suelo ya tiene agua, no hace falta regar.

**8. ¿Qué representa el signo del peso de la temperatura?**
Es positivo: a mayor temperatura, mayor probabilidad de regar, porque el calor aumenta la evaporación y la planta pierde agua más rápido. Su magnitud es mucho menor que la de la humedad, así que la temperatura inclina la decisión en los casos límite pero no la domina.

**9. ¿Por qué convertir la probabilidad en 0 o 1 con un umbral?**
La sigmoide da un valor continuo (por ejemplo 0.73), pero la acción real es binaria: se riega o no se riega. El umbral traduce la probabilidad en una decisión concreta. Además permite ajustar la política según el costo de equivocarse (desperdiciar agua o dejar secar la planta) sin reentrenar.

**10. Limitaciones para representar una planta real**
- Solo hay 10 ejemplos didácticos, no mediciones reales.
- Una sola neurona solo traza una frontera lineal, y la necesidad de riego real puede depender de combinaciones no lineales.
- Faltan variables importantes: especie, tipo de suelo, etapa de crecimiento, luz, humedad relativa del aire, lluvia prevista, tiempo desde el último riego.
- No dice **cuánta** agua dar, solo sí o no.
- No se evaluó con datos distintos a los de entrenamiento con respuesta conocida, así que no se puede medir qué tan bien generaliza.
- La normalización con máximos fijos (100 y 50) no contempla valores fuera de ese rango.
