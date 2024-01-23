from importlib import resources
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import math
from torch.utils.checkpoint import checkpoint
ASSETS_PATH = resources.files("assets")

class SinusoidalTimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 768  # Original input dimension
        self.time_encoding_dim = 768  # Dimension of time encoding
        self.concatenated_dim = self.input_dim + self.time_encoding_dim  # Total dimension after concatenation

        self.layers = nn.Sequential(
            nn.Linear(self.concatenated_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def sinusoidal_encoding(self, timesteps):
        # Normalize timesteps to be in the range [0, 1]
        timesteps = timesteps.float() / 50.0  # Assuming timesteps are provided as integers
    
        # Generate a series of frequencies
        frequencies = torch.exp(torch.arange(0, self.time_encoding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.time_encoding_dim))
        frequencies = frequencies.to(timesteps.device)

        # Apply the frequencies to the timesteps
        arguments = timesteps[:, None] * frequencies[None, :]
        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
        return encoding

    def forward(self, embed, timesteps):
        # Generate sinusoidal embeddings for the timesteps
        timestep_embed = self.sinusoidal_encoding(timesteps)

        # Concatenate the timestep embedding with the input tensor
        combined_input = torch.cat([embed, timestep_embed], dim=1)

        # Pass the combined input through the layers
        return self.layers(combined_input)

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed

if __name__ == "__main__":
    model = SinusoidalTimeMLP()
    embed = torch.randn(32, 768)
    timesteps = torch.randint(low=0, high=50, size=(32,))
    
    print(model.sinusoidal_encoding(timesteps).shape)
    print(model(embed, timesteps).shape)