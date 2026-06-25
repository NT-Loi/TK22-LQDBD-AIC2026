import logging
import sys

# Configure logging to output to both console and a file called 'app.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        text_inputs = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        
        if self.device  == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)

class SigLIPTextEncoder(TextEncoder):
    def __init__(self, device: str = None):
        super().__init__(device)

        model_id = "google/siglip-so400m-patch14-384"
        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoProcessor, SiglipModel
        self.model = SiglipModel.from_pretrained(model_id)

        del self.model.vision_model

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info("SigLIPTextEncoder initialized successfully.")

    def forward(self, query: str):
        inputs = self.processor(text=[query], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32) 

if __name__ == "__main__":
    encoder = SigLIPTextEncoder()
    sample_text = "A person riding a horse on a beach."
    features = encoder(sample_text)
    print("Features:", features)
    print(features.shape)