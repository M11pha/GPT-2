import torch
import model as GPT2_model
import tiktoken

GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

start_context = "Hello, I am"
start_context1 = "Every effort moves you"
start_context2 = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context1)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context1)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

torch.manual_seed(123)
model = GPT2_model.GPTModel(GPT_CONFIG_124M)
batch = encoded_tensor
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

trf_params = sum(p.numel() for p in model.trf_blocks.parameters())
print(f"Total number of parameters: {trf_params:,}")
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# print("Output layer shape:", model.out_head2.weight.shape)

model_trans = GPT2_model.TransformerBlock(GPT_CONFIG_124M)
one_trans_params = sum(p.numel() for p in model_trans.parameters())
print(f"Total number of parameters in one transformer block: {one_trans_params:,}")
print(f"attention parameters {sum(p.numel() for p in model_trans.att.parameters()):,}:")
print(f"feed forward parameters {sum(p.numel() for p in model_trans.ff.parameters()):,}:")
print(one_trans_params *12)

# try:
#     first_param_dtype = next(model.parameters()).dtype
#     print(f"模型参数的浮点数类型为: {first_param_dtype}")
# except StopIteration:
#     print("模型中没有参数。")
