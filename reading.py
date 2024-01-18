import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class GRUModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=250, batch_first=True, bidirectional=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Transpose input tensor to have the correct dimensions
        #x = x.permute(0, 2, 1)
        
        x, _ = self.gru1(x)
        x = self.softmax(x)
        return x

# Input shape
input_shape = (80, 636228,1)
num_classes = 2

# Create the PyTorch model
model = GRUModel(input_shape[1], 2)
print(model)

file_path = 'D:/EDU/Graduation Project/Project/Padded_Training.npz'
npz_file = np.load(file_path)

# Access individual arrays
TrainF = npz_file['features']
TrainL = npz_file['labels']

X_train_tensor = torch.tensor(TrainF, dtype=torch.float32)
y_train_tensor = torch.tensor(TrainL, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Validation data
val_file_path = 'D:/EDU/Graduation Project/Project/Padded_Testing.npz'
val_npz_file = np.load(val_file_path)
ValF = val_npz_file['features']
ValL = val_npz_file['labels']
X_val_tensor = torch.tensor(ValF, dtype=torch.float32)
y_val_tensor = torch.tensor(ValL, dtype=torch.long)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training, Validation, and Testing loops
num_epochs = 10

for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_correct_train = 0
    total_samples_train = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted_train = torch.max(outputs, 1)
        total_correct_train += (predicted_train == batch_y).sum().item()
        total_samples_train += batch_y.size(0)

    accuracy_train = total_correct_train / total_samples_train

    # Validation loop
    model.eval()
    total_correct_val = 0
    total_samples_val = 0

    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            outputs_val = model(batch_X_val)
            _, predicted_val = torch.max(outputs_val, 1)
            total_correct_val += (predicted_val == batch_y_val).sum().item()
            total_samples_val += batch_y_val.size(0)

    accuracy_val = total_correct_val / total_samples_val

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, '
          f'Training Accuracy: {accuracy_train * 100:.2f}%, '
          f'Validation Accuracy: {accuracy_val * 100:.2f}%')

