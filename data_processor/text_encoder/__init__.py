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
        inputs = self.processor(text=[query], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32) 

class SigLIP2TextEncoder(TextEncoder):
    def __init__(self, device: str = None):
        super().__init__(device)

        model_id = "google/siglip2-giant-opt-patch16-384"
        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoProcessor, SiglipModel
        self.model = SiglipModel.from_pretrained(model_id)

        del self.model.vision_model

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(self.processor.tokenizer)
        print(type(self.processor.tokenizer))
        logger.info("SigLIP2TextEncoder initialized successfully.")

    def forward(self, query: str):
        inputs = self.processor(text=[query], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32) 

class FGCLIP2TextEncoder(TextEncoder):
    def __init__(self, device: str = None, model_id: str = "qihoo360/fg-clip2-so400m"):
        super().__init__(device)

        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        if hasattr(self.model, "vision_model"):
            del self.model.vision_model

        # Ensure embedding masks are properly initialized (avoiding meta tensor issues on load)
        if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "embeddings"):
            embeddings = self.model.text_model.embeddings
            longtext_len = getattr(self.model.config.text_config, "longtext_len", 196) if hasattr(self.model.config, "text_config") else 196
            keep_len = getattr(self.model.config.text_config, "keep_len", 64) if hasattr(self.model.config, "text_config") else 64

            if hasattr(embeddings, "mask1") and embeddings.mask1.is_meta:
                mask1 = torch.zeros([longtext_len, 1])
                mask1[:keep_len, :] = 1
                embeddings.mask1 = mask1

            if hasattr(embeddings, "mask2") and embeddings.mask2.is_meta:
                mask2 = torch.zeros([longtext_len, 1])
                mask2[keep_len:, :] = 1
                embeddings.mask2 = mask2

        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        logger.info("FGCLIP2TextEncoder initialized successfully.")

    def forward(self, query: str, walk_type: str = "short"):
        max_len = 196 if walk_type == "long" else 64
        inputs = self.tokenizer(
            [query],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        seq_len = inputs["input_ids"].shape[-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            try:
                out = self.model.get_text_features(**inputs, position_ids=position_ids, walk_type=walk_type)
            except TypeError:
                out = self.model.get_text_features(**inputs, position_ids=position_ids)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)


if __name__ == "__main__":
    encoder = FGCLIP2TextEncoder()
    sample_text = "A person riding a horse on a beach."
    features = encoder(sample_text)
    print("Features:", features)
    print(features.shape)

