import logging
import torch
import numpy as np
import torch.nn.functional as F
from .base import TextEncoder

logger = logging.getLogger(__name__)

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
