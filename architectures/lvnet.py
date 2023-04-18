import torch
import torch.nn as nn
import torch.nn.functional as F

class LVNet(nn.Module):
    def __init__(self):
        super(LVNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, 7, padding=3)
        self.dropout1 = nn.Dropout2d()

        self.conv8 = nn.Conv2d(512, 512, 1)
        self.dropout2 = nn.Dropout2d()

        self.conv9 = nn.Conv2d(512, 4, 1)  # Change the output channels to 4

        self.deconv = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, output_padding=0, groups=4, bias=False, dilation=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = self.dropout1(x)

        x = F.relu(self.conv8(x))
        x = self.dropout2(x)

        x = F.relu(self.conv9(x))

        logits = self.deconv(x)

        return logits
        # prediction = torch.argmax(logits, dim=1)
        # probabilities = F.softmax(logits, dim=1)

        # return prediction, probabilities

if __name__ == "__main__":
    x= torch.rand(1,1,512,512)
    net=LVNet()
    yy=net(x)
    print('Out Shape :', yy.shape)