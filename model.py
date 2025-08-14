import torch
import torch.nn as nn
import torch.nn.functional as F

class fconnCNN(nn.Module):
    def __init__(self, N, output_dim):  # N = number of ROIs
        super().__init__()
        self.name = "fconnCNN"

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),  # [B, 16, N, 10]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),                 # [B, 16, N//2, 5]

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1), # [B, 32, N//2, 5]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),                 # [B, 32, N//4, 2]

            nn.Conv2d(32, 64, kernel_size=(3, 2)),            # [B, 64, N//4 - 2, 1]
            nn.ReLU(),
        )

        # Compute the output shape dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, N, 10)  # [B=1, C=1, H=N, W=10]
            dummy_output = self.encoder(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # Squeeze for 1 trait, don't when computing regression on 5 traits
