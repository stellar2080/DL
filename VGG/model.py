import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self,features,num_classes=10,init_weights=False):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self.initialize_weights()

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

cfgs = {
    'vgg11': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(cfg: list):
    layers = []
    in_channels = 3
    for m in cfg:
        if m == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, m, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = m
    return nn.Sequential(*layers)

def vgg(model_name="vgg11",**kwargs):
    assert model_name in cfgs, "model {} should be in cfgs".format(model_name)
    cfg = cfgs[model_name]
    return VGG(make_features(cfg),**kwargs)