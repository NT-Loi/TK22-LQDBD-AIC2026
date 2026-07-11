import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, device: str = None):
        super().__init__()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def forward(self, text: str):
        pass
