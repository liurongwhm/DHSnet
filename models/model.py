import torch.nn as nn
from models.featureExtrator import FE

class DHSnet(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size):
        super(DHSnet, self).__init__()
        self.num_classes = num_classes
        self.feature_layers = FE(in_channels, num_classes, patch_size)
        self.hidden_channel = self.feature_layers.hidden_layer
        self.head = nn.Sequential(
            nn.Linear(self.hidden_channel, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, self.num_classes)
        )
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.hidden_channel, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, source, target):
        source_share = self.feature_layers(source)
        s_pred = self.head(source_share)
        target_share = self.feature_layers(target)
        t_label = self.head(target_share)
        y_pseudo = self.pseudo_head(target_share)

        return s_pred, t_label, source_share, target_share, y_pseudo

    def predict(self, x):
        f = self.feature_layers(x)
        x = self.head(f)
        return x
