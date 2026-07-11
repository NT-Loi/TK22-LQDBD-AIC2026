import logging
import torch
import numpy as np
import torch.nn.functional as F
from .base import TextEncoder

logger = logging.getLogger(__name__)

class CLIPTextEncoder(TextEncoder):
    def __init__(self, device: str = None):
        super().__init__(device)

        model_id = "ViT-H-14-378-quickgelu"
        pretrained = "dfn5b"
        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        
        # delete image encoder to save memory
        del self.model.visual
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)

        logger.info("CLIPTextEncoder initialized successfully.")

    def forward(self, query: str):
        text_inputs = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        
        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)
