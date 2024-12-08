import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define uma CNN simples
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Saída: 16x28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Saída: 32x28x28
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)  # Reduz dimensão: 2x downsample

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten para o Fully Connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializa a rede
model = SimpleCNN()

# Dataset e DataLoader
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Monitorando as ativações (forward) e os gradientes (backward) usando hooks
activation = {}
gradients = {}

def get_activation(name):
    """Hook para monitorar as ativações no forward pass"""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_gradient(name):
    """Hook para monitorar os gradientes no backward pass"""
    def hook(model, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()  # grad_output é uma tupla
    return hook

# Registrando hooks nas camadas convolucionais
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv2.register_backward_hook(get_gradient('conv2'))  # Backward hook na conv2

# Pegando uma amostra para inspecionar
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Passando pelo forward
output = model(images)

# Calculando perda e realizando o backward
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
loss.backward()  # Backward pass para calcular os gradientes

# Inspecionando as ativações e gradientes
print("Ativação da Conv1 (forward):", activation['conv1'].shape)  # (batch_size, 16, 14, 14)
print("Ativação da Conv2 (forward):", activation['conv2'].shape)  # (batch_size, 32, 7, 7)
print("Gradiente da Conv2 (backward):", gradients['conv2'].shape)  # (batch_size, 32, 7, 7)

# Visualização das ativações da Conv1
act = activation['conv1'][0]  # Pegue o primeiro exemplo do batch
fig, axes = plt.subplots(1, 8, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(act[i].cpu().numpy(), cmap='viridis')
    ax.axis('off')
plt.suptitle("Ativações - Conv1")
plt.show()

# Visualização dos gradientes da Conv2
grad = gradients['conv2'][0]  # Pegue o primeiro exemplo do batch
fig, axes = plt.subplots(1, 8, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(grad[i].cpu().numpy(), cmap='viridis')
    ax.axis('off')
plt.suptitle("Gradientes - Conv2")
plt.show()
