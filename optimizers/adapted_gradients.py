import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Configurações básicas
torch.manual_seed(42)

# Dados sintéticos para um problema de regressão
x = torch.linspace(-1, 1, 100).unsqueeze(1)  # Entrada
y = x.pow(3) + 0.1 * torch.randn(x.size())   # Saída com ruído

# Modelo simples: perceptron de uma camada
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
criterion = nn.MSELoss()

# Otimizador AdaGrad
optimizer = optim.Adagrad(model.parameters(), lr=0.1)

# Armazenar histórico de gradientes e parâmetros
gradient_history = []
param_history = []
loss_history = []

# Treinamento
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Zerar gradientes
    outputs = model(x)     # Forward pass
    loss = criterion(outputs, y)  # Calcular perda
    loss.backward()        # Backward pass

    # Capturar gradientes e parâmetros
    grads = [p.grad.clone() for p in model.parameters()]
    gradient_history.append(grads)
    params = [p.clone() for p in model.parameters()]
    param_history.append(params)
    loss_history.append(loss.item())

    optimizer.step()       # Atualizar parâmetros

# Função para calcular a escala dos gradientes adaptados
def compute_adaptive_gradients(grad_history, eps=1e-10):
    G = [torch.zeros_like(g) for g in grad_history[0]]
    scales = []

    for grads in grad_history:
        for i, g in enumerate(grads):
            G[i] += g.pow(2)  # Acumular gradiente ao quadrado
            adapted_grad = g / (torch.sqrt(G[i]) + eps)
            scales.append(adapted_grad.norm().item())  # Armazenar escala
    
    return np.array(scales).reshape(len(grad_history), -1)

# Calcular escalas dos gradientes adaptados
adaptive_grad_scales = compute_adaptive_gradients(gradient_history)

# Visualizações
plt.figure(figsize=(12, 8))

# Escalas dos gradientes adaptados
plt.subplot(2, 1, 1)
for i in range(adaptive_grad_scales.shape[1]):
    plt.plot(adaptive_grad_scales[:, i], label=f'Gradiente Adaptado {i+1}')
plt.xlabel('Época')
plt.ylabel('Norma do Gradiente Adaptado')
plt.title('Evolução dos Gradientes Adaptados (AdaGrad)')
plt.legend()

# Curva de perda
plt.subplot(2, 1, 2)
plt.plot(loss_history, color='red')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Curva de Perda Durante o Treinamento')

plt.tight_layout()
plt.show()
