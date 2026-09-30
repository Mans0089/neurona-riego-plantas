import numpy as np

print("Neurona artificial para decidir si una planta necesita riego")
print("Entradas: x1 = humedad del suelo (%), x2 = temperatura ambiental (°C)")
print("Salida: 1 = regar, 0 = no regar")

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def entrenar(X_norm, y, tasa_aprendizaje, epocas, mostrar=False, semilla=7):
    """Entrena la neurona y devuelve pesos, sesgo y error final."""
    rng = np.random.default_rng(semilla)
    pesos = rng.normal(size=(2, 1))  # un peso para humedad y otro para temperatura
    sesgo = 0.0
    intervalo = max(epocas // 5, 1)

    for epoca in range(epocas):
        z = X_norm @ pesos + sesgo
        predicciones = sigmoide(z)

        # Error cuadrático medio y sus derivadas
        error = predicciones - y
        gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X_norm)
        gradiente_pesos = X_norm.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and epoca % intervalo == 0:
            perdida = np.mean(error ** 2)
            print(f"Epoca {epoca:6d} | error: {perdida:.6f}")

    probabilidades = sigmoide(X_norm @ pesos + sesgo)
    error_final = np.mean((probabilidades - y) ** 2)
    return pesos, sesgo, error_final

def predecir(datos_originales, pesos, sesgo, umbral=0.5):
    """Normaliza con la MISMA escala del entrenamiento y predice."""
    datos_norm = datos_originales / escala
    probabilidad = sigmoide(datos_norm @ pesos + sesgo)
    respuesta = (probabilidad >= umbral).astype(int)
    return probabilidad, respuesta

tasa_aprendizaje = 0.5
epocas = 10000

print(f"\n=== Entrenamiento base: {epocas} épocas, tasa {tasa_aprendizaje} ===")
pesos, sesgo, error_final = entrenar(X_normalizado, y, tasa_aprendizaje, epocas, mostrar=True)

print(f"\nPeso de la humedad:     {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo:                  {float(sesgo):.4f}")
print(f"Error final (MSE):      {error_final:.6f}")

print("\nSi la probabilidad es >= 0.5 la salida es 1 (regar); en caso contrario 0 (no regar)")
probabilidades, respuestas = predecir(X, pesos, sesgo)

print("\nCaso | Humedad | Temp | Probabilidad | Respuesta | Esperado")
for i, (entrada, p, r, esp) in enumerate(zip(X, probabilidades.ravel(), respuestas.ravel(), y.ravel()), 1):
    print(f"{i:4d} | {entrada[0]:6.0f}% | {entrada[1]:3.0f}° | {p:12.4f} | {r:9d} | {int(esp):8d}")

aciertos = int(np.sum(respuestas == y.astype(int)))
print(f"\nRespuestas correctas: {aciertos}/10")

print("\nInterpretación del signo de los pesos:")
print(" - Peso de la humedad negativo: a MÁS humedad, MENOR probabilidad de regar.")
print(" - Peso de la temperatura positivo: a MÁS temperatura, MAYOR probabilidad de regar.")

nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)

print("\n=== Pruebas con condiciones nuevas (normalizadas con la misma escala) ===")
prob_nuevos, resp_nuevos = predecir(nuevos, pesos, sesgo)
print("Humedad | Temp | Probabilidad | Decisión")
for entrada, p, r in zip(nuevos, prob_nuevos.ravel(), resp_nuevos.ravel()):
    decision = "Regar" if r == 1 else "No regar"
    print(f"{entrada[0]:6.0f}% | {entrada[1]:3.0f}° | {p:12.4f} | {r} ({decision})")

experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

caso_referencia = np.array([[45, 34]], dtype=float)

print("\n=== Experimentos (se cambia un solo parámetro a la vez) ===")
print(f"{'Experimento':22s} | {'Épocas':>6s} | {'Tasa':>5s} | {'Error final':>11s} | {'Aciertos':>8s} | {'P(45%,34°C)':>11s}")
for nombre, ep, tasa in experimentos:
    p_exp, s_exp, err_exp = entrenar(X_normalizado, y, tasa, ep)
    _, resp_exp = predecir(X, p_exp, s_exp)
    aciertos_exp = int(np.sum(resp_exp == y.astype(int)))
    prob_ref, _ = predecir(caso_referencia, p_exp, s_exp)
    print(f"{nombre:22s} | {ep:6d} | {tasa:5.2f} | {err_exp:11.6f} | {aciertos_exp:5d}/10 | {prob_ref[0, 0]:11.4f}")

print("\n=== Comparación de umbrales (mismos pesos del entrenamiento base) ===")
todos = np.vstack([X, nuevos])
prob_todos, _ = predecir(todos, pesos, sesgo)
print("Humedad | Temp | Probabilidad | U=0.4 | U=0.5 | U=0.6")
for entrada, p in zip(todos, prob_todos.ravel()):
    r4, r5, r6 = int(p >= 0.4), int(p >= 0.5), int(p >= 0.6)
    marca = "  <- cambia" if len({r4, r5, r6}) > 1 else ""
    print(f"{entrada[0]:6.0f}% | {entrada[1]:3.0f}° | {p:12.4f} | {r4:5d} | {r5:5d} | {r6:5d}{marca}")

assert np.array_equal(respuestas, y.astype(int))
print("\nVerificación superada: la neurona clasifica correctamente los 10 casos de entrenamiento.")