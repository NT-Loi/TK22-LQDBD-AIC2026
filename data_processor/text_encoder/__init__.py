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

import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN", None)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, List

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

class Qwen3VLTextEncoder(TextEncoder):
    def __init__(self, device: str = None, model_id: str = "Qwen/Qwen3-VL-Embedding-2B"):
        super().__init__(device)

        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoProcessor, AutoModel
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        )

        # Delete image encoder to save VRAM (text-only query encoder)
        if hasattr(self.model, "visual"):
            del self.model.visual

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info("Qwen3VLTextEncoder initialized successfully.")

    def forward(self, query: Union[str, List[str]]):
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        texts = []
        for q in queries:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": q}]}
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        inputs = self.processor(text=texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            last_positions = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
            rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
            text_features = hidden_state[rows, last_positions]

        if self.device == "cuda":
            text_features = text_features.cpu()

        return F.normalize(text_features.float(), p=2, dim=-1).detach().numpy().astype(np.float32)

if __name__ == "__main__":
    encoder = Qwen3VLTextEncoder()
    sample_text = "A person riding a horse on a beach."
    features = encoder(sample_text)
    print("Features:", features)
    print(features.shape)