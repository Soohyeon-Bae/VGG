# keras로 구현한 것을 일대일로 torch 형식으로 바꾼 것.
# conv 레이어를 일일이 적지 않아도 알맞는 파라미터가 적용되도록 할 수 있을 것 같은데 더 공부해봐야겠다. (22.01.14.)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(F.relu(self.conv10(x)))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(F.relu(self.conv13(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

model = VGG16().to("cuda:0")
summary(model, input_size=(3, 224, 224))


