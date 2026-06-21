import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, device: str=None):
        super().__init__()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

    def forward(self, text: str):
        pass

class CLIPTextEncoder(TextEncoder):
    def __init__(self, device: str=None):
        super().__init__(device)

        self.model_name = "CLIP_H14"
        model_id = "ViT-H-14-378-quickgelu"
        pretrained = "dfn5b"
        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(model_id,
                                                                pretrained=pretrained)
        
        # delete image encoder
        del self.model.visual
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)

        logger.info("CLIPTextEncoder initialized successfully.")

    def forward(self, query: str):
        import torch.nn.functional as F
        text_inputs = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            if self.device  == "cuda":
                text_features = text_features.cpu()
            return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)

if __name__ == "__main__":
    encoder = CLIPTextEncoder()
    sample_text = "A person riding a horse on a beach."
    features = encoder(sample_text)
    print("Features:", features)