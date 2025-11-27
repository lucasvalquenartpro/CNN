import torch
from torch import nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_early_layers=True):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        if freeze_early_layers:
            self._early_layers()

        self.fc1 = nn.Linear(513, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def _early_layers(self):
        layers = [self.conv1, self.bn1, self.layer1, self.layer2]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


    def forward(self, x, genre):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        genre = genre.unsqueeze(1)
        x = torch.cat((x, genre), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = x.squeeze(1)
        return x