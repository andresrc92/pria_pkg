import numpy as np
import torch

class NetworkCoarseSimple(torch.nn.Module):

    def __init__(self):
        super(NetworkCoarseSimple, self).__init__()

        # Define the network layers
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            torch.nn.ReLU(inplace=False),
            # torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # torch.nn.ReLU(inplace=False),
            # torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # torch.nn.ReLU(inplace=False),
            # torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # torch.nn.ReLU(inplace=False),

        )

        self.loss_coefficient = 0.01
        self.flat_size = 2 * 2 * 128
        self.mlp = torch.nn.Sequential(
            # torch.nn.Linear(29952,200),
            torch.nn.Linear(605472,200),
            torch.nn.ReLU(inplace=False),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(200, 200),
            # torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(200, 50),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(50, 7)
        )

    def forward(self, input_image):
        # print(input_image.shape)
        # Compute the cnn features
        image_features = self.conv(input_image)
        image_features_flat = torch.reshape(image_features, (input_image.shape[0], -1))
        # Compute the mlp features
        mlp_features = self.mlp(image_features_flat)
        # Make the prediction
        prediction = self.predictor(mlp_features)
        return prediction
    
    def compute_loss(self, pred, gt):
        translation_loss = torch.nn.MSELoss(reduction='none')(pred[:,:3], gt[:,:3]).mean()
        rotation_loss = torch.nn.MSELoss(reduction='none')(pred[:,3:], gt[:,3:]).mean()
        loss = translation_loss + self.loss_coefficient * rotation_loss
        return loss