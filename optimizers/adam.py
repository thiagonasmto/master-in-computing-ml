import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo (a montanha)
def objective(x, y):
    return x**2.0 + y**2.0  # Forma de uma tigela convexa

# Gradiente da função objetivo
def gradients(x, y):
    grad_x = 2 * x  # Derivada parcial em relação a x
    grad_y = 2 * y  # Derivada parcial em relação a y
    return grad_x, grad_y

# Configuração inicial
np.random.seed(42)  # Para resultados reproduzíveis
x_current, y_current = np.random.uniform(-1.0, 1.0, 2)  # Posição inicial do "explorador"
learning_rate = 0.1  # Taxa de aprendizado

# Hiperparâmetros do Adam
beta1 = 0.9  # Decaimento para o momento (média dos gradientes)
beta2 = 0.999  # Decaimento para a variância (média dos quadrados dos gradientes)
epsilon = 1e-8  # Evita divisões por zero

# Inicialização de momentos
m_x, m_y = 0.0, 0.0  # Primeiros momentos (médias dos gradientes)
v_x, v_y = 0.0, 0.0  # Segundos momentos (médias dos quadrados dos gradientes)

# Listas para armazenar o caminho percorrido
path_x, path_y = [x_current], [y_current]

# Simulação de passos do Adam
for t in range(1, 51):  # 50 passos
    grad_x, grad_y = gradients(x_current, y_current)
    
    # Atualização dos momentos
    m_x = beta1 * m_x + (1 - beta1) * grad_x
    m_y = beta1 * m_y + (1 - beta1) * grad_y
    v_x = beta2 * v_x + (1 - beta2) * (grad_x**2)
    v_y = beta2 * v_y + (1 - beta2) * (grad_y**2)
    
    # Correção de viés
    m_x_hat = m_x / (1 - beta1**t)
    m_y_hat = m_y / (1 - beta1**t)
    v_x_hat = v_x / (1 - beta2**t)
    v_y_hat = v_y / (1 - beta2**t)
    
    # Atualização dos parâmetros
    x_current -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
    y_current -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
    
    # Armazenar o caminho
    path_x.append(x_current)
    path_y.append(y_current)

# Visualização em 3D
xaxis = np.arange(-1.0, 1.0, 0.1)
yaxis = np.arange(-1.0, 1.0, 0.1)
x, y = np.meshgrid(xaxis, yaxis)
results = objective(x, y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, results, cmap='jet', alpha=0.8)

# Plotar o caminho percorrido pelo Adam
z_path = [objective(px, py) for px, py in zip(path_x, path_y)]
ax.plot(path_x, path_y, z_path, color='red', marker='o', markersize=4, label='Caminho do Adam')

ax.set_title('Otimizador Adam Descendo a Montanha')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objetivo')
ax.legend()
plt.show()
