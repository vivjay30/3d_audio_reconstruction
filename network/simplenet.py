import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.best_models = {}  # Filename to accuracy

        # Freq x Time x 8
        self.conv1 = nn.Conv2d(8, 64, 7, 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 7, 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 12)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3_bn(F.relu(self.conv3(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.log_softmax(x)

    def loss(self, prediction, label, reduction='elementwise_mean'):
        loss_val = F.cross_entropy(prediction, label, reduction=reduction)
        return loss_val
