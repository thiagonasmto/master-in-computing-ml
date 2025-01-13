import numpy as np
import matplotlib.pyplot as plt

# Gradientes simulados
np.random.seed(42)
gradients = np.random.randn(100)

# EWMA
beta = 0.5
ewma = []
v_t = 0
for g_t in gradients:
    v_t = beta * v_t + (1 - beta) * g_t
    ewma.append(v_t)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(gradients, label="Gradientes Originais")
plt.plot(ewma, label="Gradientes Suavizados (EWMA)", linewidth=2)
plt.legend()
plt.title("Suavização de Gradientes com EWMA")
plt.xlabel("Iterações")
plt.ylabel("Valor do Gradiente")
plt.show()
