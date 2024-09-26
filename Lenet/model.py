import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            nn.Linear(84,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1) # [N,C,H,W]
        x = self.classifier(x)
        return x