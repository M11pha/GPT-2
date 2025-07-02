import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    # vocab_size refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
    "vocab_size": 50257,   # Vocabulary size
    # denotes the maximum number of input tokens the model can handle via the positional embeddings
    "context_length": 1024, # Shortened context length (orig: 1024)
    # emb_dim represents the embedding size, transforming each token into a 768-dimensional vector
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # in_idx.shape=torch.Size([2, 4])
        batch_size, seq_len = in_idx.shape #
        # tok_embeds.shape=torch.Size([2, 4, 768])
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds.shape=torch.Size([4, 768])
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # x.shape=torch.Size([2, 4, 768])
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # logits.shape = torch.Size([2, 4, 50257])
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)