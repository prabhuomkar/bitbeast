import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot


NUM_EPOCHS = 10000
LEARNING_RATE = 0.01
SAVED_MODEL_FILE_NAME = "model.pth"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = Model()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), SAVED_MODEL_FILE_NAME)

make_dot(model(inputs)).render("model", format="png", cleanup=True)

with torch.no_grad():
    model = Model()
    model.load_state_dict(torch.load(SAVED_MODEL_FILE_NAME))
    model.eval()
    print(model(inputs).numpy())
    outputs = model(torch.tensor([0,0], dtype=torch.float32))
    print(outputs.numpy())
