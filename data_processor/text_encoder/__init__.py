import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN", None)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_source(model_id: str, local_dir_name: str = None) -> str:
    """Prefer a complete project-local model directory when one is available."""
    if local_dir_name:
        local_path = PROJECT_ROOT / "data" / "models" / local_dir_name
        if (local_path / "config.json").is_file():
            return str(local_path)
    return model_id


def _from_pretrained_local_first(loader, model_id: str, **kwargs):
    """Use the HF cache without a network probe, falling back to normal download behavior."""
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return loader.from_pretrained(model_id, **kwargs)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, List

# helper function for Qwen3EmbeddingTextEncoder
from torch import Tensor
def last_token_pool(last_hidden_states: Tensor,
                attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

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

        with torch.inference_mode():
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
        self.model = SiglipModel.from_pretrained(model_id, token=hf_token)

        del self.model.vision_model

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info("SigLIPTextEncoder initialized successfully.")

        self.max_length = 64

    def forward(self, query: str, max_length: int = None):
        max_length = max_length or self.max_length
        inputs = self.processor(text=[query], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
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
        self.model = _from_pretrained_local_first(SiglipModel, model_id, token=hf_token)

        del self.model.vision_model

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = _from_pretrained_local_first(AutoProcessor, model_id)
        logger.info("SigLIP2TextEncoder initialized successfully.")

        self.max_length = 64

    def forward(self, query: str, max_length: int = None):
        max_length = max_length or self.max_length
        inputs = self.processor(text=[query], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32) 

class Qwen3VLEmbeddingTextEncoder(TextEncoder):
    def __init__(self, device: str = None, model_id: str = "Qwen/Qwen3-VL-Embedding-2B"):
        super().__init__(device)

        model_source = _model_source(model_id, "Qwen3-VL-Embedding-2B")
        logger.info(f"Loading model '{model_source}' to device '{self.device}'...")
        from transformers import AutoProcessor, AutoModel
        self.model = _from_pretrained_local_first(
            AutoModel,
            model_source,
            # The checkpoint is BF16. Expanding it to FP32 on CPU roughly
            # doubles resident memory and can kill the app when SigLIP2 is
            # loaded at the same time.
            dtype=torch.float16 if self.device == "cuda" else torch.bfloat16,
            token=hf_token
        )

        # Delete image encoder to save VRAM (text-only query encoder)
        if hasattr(self.model, "visual"):
            del self.model.visual

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = _from_pretrained_local_first(AutoProcessor, model_source)

        logger.info("Qwen3VLTextEncoder initialized successfully.")

        self.task = "Retrieve video keyframes or images that depict the scene described in the query."
        self.max_length = 2048

    def forward(self, query: Union[str, List[str]], task: str = None, max_length: int = None):
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        texts = []
        task = task or self.task
        for q in queries:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": f"{task}"}]},
                {"role": "user", "content": [{"type": "text", "text": q}]}
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        max_length = max_length or self.max_length
        inputs = self.processor(text=texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            last_positions = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
            rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
            text_features = hidden_state[rows, last_positions]

        if self.device == "cuda":
            text_features = text_features.cpu()

        return F.normalize(text_features.float(), p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

class FGCLIP2TextEncoder(TextEncoder):
    def __init__(self, device: str = None, model_id: str = "qihoo360/fg-clip2-so400m"):
        super().__init__(device)

        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        except Exception:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=hf_token)

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

        with torch.inference_mode():
            try:
                out = self.model.get_text_features(**inputs, position_ids=position_ids, walk_type=walk_type)
            except TypeError:
                out = self.model.get_text_features(**inputs, position_ids=position_ids)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out

        if self.device == "cuda":
            text_features = text_features.cpu()
            
        return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)

class Qwen3EmbeddingTextEncoder(TextEncoder):
    def __init__(self, device: str = None, model_id: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__(device)

        logger.info(f"Loading model '{model_id}' to device '{self.device}'...")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = _from_pretrained_local_first(AutoTokenizer, model_id, padding_side='left')
        self.model = _from_pretrained_local_first(AutoModel, model_id, token=hf_token)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Qwen3EmbeddingTextEncoder initialized successfully.")

        self.task = "Given a video search query, retrieve relevant video shot descriptions that match the query"
        self.max_length = 2048

    def forward(self, query: Union[str, List[str]], task: str = None, max_length: int = None):
        queries = [query] if isinstance(query, str) else query
        task = task or self.task
        formatted_queries = [
            f"Instruct: {task}\nQuery: {q.strip()}" for q in queries
        ]
        max_length = max_length or self.max_length

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            formatted_queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            text_features = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        if self.device == "cuda":
            text_features = text_features.cpu()

        return F.normalize(text_features.float(), p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

if __name__ == "__main__":
    encoder = SigLIP2TextEncoder()
    sample_text = "A person riding a horse on a beach."
    features = encoder(sample_text)
    print("Features:", features)
    print(features.shape)
