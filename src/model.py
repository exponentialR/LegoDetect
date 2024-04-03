import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F


class LegoDetector(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LegoDetector, self).__init__()
        self.num_classes = num_classes
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.bbox = nn.Linear(self.resnet18.fc.in_features, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        class_scores = self.classifier(x)
        bbox = torch.sigmoid(self.bbox(x))
        return class_scores, bbox


if __name__ == '__main__':
    model = LegoDetector(10)
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    class_scores, bbox = model(dummy_input)
    print(class_scores.shape, bbox.shape)
    print(class_scores, bbox)