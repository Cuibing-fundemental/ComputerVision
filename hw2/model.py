import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet_Pretrain(nn.Module):
    def __init__(self, num_classes=37, pretrained=True):
        super(ResNet_Pretrain, self).__init__()
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

    @property
    def fc_parameters(self):
        return list(self.resnet.fc.parameters())
    
    @property
    def resnet_parameters(self):
        return [param for name, param in self.resnet.named_parameters() if not name.startswith('fc.')]
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class SEBasicBlock(models.resnet.BasicBlock):
    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.bn2.num_features, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet_SE(nn.Module):
    def __init__(self, num_classes=37):
        super(ResNet_SE, self).__init__()
        self.resnet = models.ResNet(SEBasicBlock, [2, 2, 2, 2])
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.load_state_dict(pretrained.state_dict(), strict=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    
    @property
    def fc_parameters(self):
        return list(self.resnet.fc.parameters())
    
    @property
    def se_parameters(self):
        return [param for name, param in self.resnet.named_parameters() if 'se' in name]
    
    @property
    def resnet_parameters(self):
        resnet_parameters = [param for name, param in self.resnet.named_parameters() if not name.startswith('fc.') and 'se' not in name]
        return resnet_parameters

class UnetEncoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias = False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        
        identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class UnetDecoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

class Unet(nn.Module):
    def __init__(self,nums_classes=3):
        super().__init__()
        self.en1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) #224->224
        self.en2 = UnetEncoder(64,128) #224->112
        self.en3 = UnetEncoder(128,256) #112->56
        self.en4 = UnetEncoder(256,512) #56->28

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.de1 = UnetDecoder(512,256) #28->56
        self.de2 = UnetDecoder(256,128) #56->112
        self.de3 = UnetDecoder(128,64) #112->224
        self.fc = nn.Conv2d(in_channels=128,out_channels=nums_classes,kernel_size=1)

    def forward(self,x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)

        c = self.conv(e4)

        d1 = self.de1(torch.cat([c,e4],dim=1))
        d2 = self.de2(torch.cat([d1,e3],dim=1))
        d3 = self.de3(torch.cat([d2,e2],dim=1))

        out = self.fc(torch.cat([d3,e1],dim=1))
        return out

class DiceLoss(nn.Module):
    def __init__(self,label_smoothing = 0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.epsilon = 1e-8

    def forward(self,pred,labels):
        prob = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        intersection = (prob * labels_one_hot).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1 - dice_score[:,:-1].mean()
        return dice_loss

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, label_smoothing=0):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dice_loss = DiceLoss(label_smoothing=label_smoothing)

    def forward(self, pred, labels):
        ce = self.ce_loss(pred, labels)
        dice = self.dice_loss(pred, labels)
        return self.alpha * ce + (1 - self.alpha) * dice