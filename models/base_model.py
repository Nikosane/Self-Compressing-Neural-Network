import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], output_size=10):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x
