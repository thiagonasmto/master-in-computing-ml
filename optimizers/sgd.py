import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Definição da função de perda simulando dados
def generate_data(batch_size=1):
    x_data = torch.randn(batch_size, requires_grad=False) * 2.0  # Dados x
    y_data = torch.randn(batch_size, requires_grad=False) * 2.0  # Dados y
    return x_data, y_data

# Função que calcula a perda para os dados simulados
def function_to_minimize(x, y, x_data, y_data):
    return ((x - x_data)**2 + (y - y_data)**2).mean()  # Erro quadrático médio

# Inicialização dos parâmetros
x = torch.tensor([2.0], requires_grad=True)  # Ponto inicial x
y = torch.tensor([2.0], requires_grad=True)  # Ponto inicial y

# Configuração do otimizador SGD
optimizer = optim.SGD([x, y], lr=0.1)

# Salvando a trajetória
trajectory = []

# Otimização por SGD
for step in range(50):
    optimizer.zero_grad()                  # Zera os gradientes
    x_data, y_data = generate_data()       # Simula um minibatch
    loss = function_to_minimize(x, y, x_data, y_data)  # Calcula a perda
    loss.backward()                        # Calcula os gradientes
    optimizer.step()                       # Atualiza os parâmetros
    trajectory.append((x.item(), y.item(), loss.item(), x.item()**2 + y.item()**2))  # Salva a trajetória com a altura

# Preparação dos dados para visualização
trajectory = np.array(trajectory)
x_vals = np.linspace(-2.5, 2.5, 100)
y_vals = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2  # Superfície Z = f(x, y)

# Gráfico 2D
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', label="SGD Trajectory")
plt.title("SGD Optimization in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)  # Superfície
# Adiciona os pontos da trajetória na superfície
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 3], 'r-o', label="SGD Trajectory")  # Ajusta a altura dos pontos
ax.set_title("SGD Optimization in 3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.legend()
plt.show()
