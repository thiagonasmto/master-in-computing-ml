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

# Função para realizar a otimização usando SGD
def optimize_with_sgd(lr, momentum=0.0, nesterov=False, iterations=50):
    # Inicialização dos parâmetros
    x = torch.tensor([2.0], requires_grad=True)  # Ponto inicial x
    y = torch.tensor([2.0], requires_grad=True)  # Ponto inicial y

    # Configuração do otimizador (SGD com diferentes parâmetros)
    optimizer = optim.SGD([x, y], lr=lr, momentum=momentum, nesterov=nesterov)

    trajectory = []  # Lista para armazenar os pontos da trajetória

    # Otimização por SGD
    for step in range(iterations):
        optimizer.zero_grad()                  # Zera os gradientes
        x_data, y_data = generate_data()       # Simula um minibatch
        loss = function_to_minimize(x, y, x_data, y_data)  # Calcula a perda
        loss.backward()                        # Calcula os gradientes
        optimizer.step()                       # Atualiza os parâmetros
        trajectory.append((x.item(), y.item(), loss.item(), x.item()**2 + y.item()**2))  # Salva a trajetória com a altura

    return np.array(trajectory)

# Preparação dos dados para visualização (Superfície)
x_vals = np.linspace(-2.5, 2.5, 100)
y_vals = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2  # Superfície Z = f(x, y)

# Comparando os métodos de SGD
trajectories = {
    "SGD": optimize_with_sgd(lr=0.1),
    "SGD with Momentum": optimize_with_sgd(lr=0.1, momentum=0.9),
    "SGD with Nesterov Momentum": optimize_with_sgd(lr=0.1, momentum=0.9, nesterov=True),
}

# Criação dos subplots
fig = plt.figure(figsize=(18, 6))

# Subplot 1: SGD
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.plot(trajectories["SGD"][:, 0], trajectories["SGD"][:, 1], trajectories["SGD"][:, 3], 'r-o', label="SGD Trajectory")
ax1.set_title("SGD Optimization")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")
ax1.legend()

# Subplot 2: SGD com Momentum
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax2.plot(trajectories["SGD with Momentum"][:, 0], trajectories["SGD with Momentum"][:, 1], trajectories["SGD with Momentum"][:, 3], 'r-o', label="SGD with Momentum")
ax2.set_title("SGD with Momentum")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")
ax2.legend()

# Subplot 3: SGD com Nesterov Momentum
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax3.plot(trajectories["SGD with Nesterov Momentum"][:, 0], trajectories["SGD with Nesterov Momentum"][:, 1], trajectories["SGD with Nesterov Momentum"][:, 3], 'r-o', label="SGD with Nesterov Momentum")
ax3.set_title("SGD with Nesterov Momentum")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("f(x, y)")
ax3.legend()

plt.tight_layout()
plt.show()
