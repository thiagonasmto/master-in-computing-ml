import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Definindo um modelo simples
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Inicializando o modelo, critério de perda e otimizador
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Usando o Learning Rate Scheduler - StepLR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Vamos monitorar a taxa de aprendizado
lr_history = []

# Simulando o treinamento por 50 epochs
for epoch in range(50):
    # Dummy input
    inputs = torch.randn(10)
    target = torch.randn(1)
    
    # Forward pass
    output = model(inputs)
    loss = criterion(output, target)
    
    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Atualizando a taxa de aprendizado
    scheduler.step()
    
    # Salvando o valor da taxa de aprendizado para visualização
    lr_history.append(optimizer.param_groups[0]['lr'])

# Plotando a taxa de aprendizado
plt.plot(lr_history)
plt.title('Evolução da Taxa de Aprendizado durante o Treinamento')
plt.xlabel('Epochs')
plt.ylabel('Taxa de Aprendizado')
plt.show()
