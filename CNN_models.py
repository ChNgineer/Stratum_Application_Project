import torch.nn as nn
import torch.nn.functional as F

# Base model, control case from provided repo
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.max_pool2d(x1, 2, 2)
        x2 = F.relu(self.conv2(x))
        x = F.max_pool2d(x2, 2, 2)
        x = x.view(-1, 4*4*50)
        # This extra softmax and relu activations are from the original paper's model but was removed
        # x3 = F.relu(self.fc1(x))
        # h = F.softmax(self.fc2(x3), dim=1) 
        # return h, x3, x2, x1
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x2, x1

# First new model
# Changes:
    # Num conv layers: 2 -> 3
    # Conv layers: (1,20,5,1) -> (1,16,4,1), (20,50,5,1) -> (16,32,4,1), NULL -> (32,64,4,1)
    # Linear layers: (4*4*50,500) -> (2*2*64,512), (500,10) -> (512,10)
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 1)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x1, 2, 2)
        x2 = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x2, 2, 2)
        x3 = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x3, 2, 2)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x3, x2, x1

# Second new model
# Changes:
    # Num conv layers: 2 -> 1
    # Conv layers: (1,20,5,1) -> (1,64,8,2), (20,50,5,1) -> NULL
    # Linear layers: (4*4*50,500) -> (4*4*64,256), (500,10) -> (256,512), NULL -> (512,10)
    # Pooling: max -> avg
    # Activation: relu -> softplus
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 2)
        self.fc1 = nn.Linear(5*5*32, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, x):
        x1 = F.softplus(self.conv1(x))
        x = F.avg_pool2d(x1, 2, 2)
        x = x.view(-1, 5*5*32)
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = self.fc3(x)
        return x, x1