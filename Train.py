import os
import urllib.request
import tiktoken
from model import generate_text_simple, GPTModel
from Dataset import create_dataloader_v1
import torch
import train_util as tu

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
#
# print("\nValidation loader:")

# for x, y in val_loader:
# #     print(x.shape, y.shape)
# #
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
# #     torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader
# #     with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
# #         train_loss = tu.calc_loss_loader(train_loader, model, device)
# #         val_loss = tu.calc_loss_loader(val_loader, model, device)
# #
# #     print("Training loss:", train_loss)
# #     print("Validation loss:", val_loss)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
#
# num_epochs = 500
# # 开启自动混合精度 (AMP) 计算
# # scaler = torch.amp.GradScaler(enabled=False)  # 关闭 fp16 的 scale 机制
# # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
# #
# # start_event = torch.cuda.Event(enable_timing=True)
# # end_event = torch.cuda.Event(enable_timing=True)
#
# # start_event.record()  # 记录 GPU 训练起始时间
#
# # 开启自动混合精度 (AMP) 计算
# scaler = torch.amp.GradScaler(enabled=False)  # 关闭 fp16 的 scale 机制
# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#     train_losses, val_losses, tokens_seen = tu.train_model_simple(
#         model, train_loader, val_loader, optimizer, device,
#         num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#         start_context="Every effort moves you", tokenizer=tokenizer
#     )
#
# end_event.record()  # 记录 GPU 训练结束时间
# torch.cuda.synchronize()  # 确保所有 GPU 计算完成
#
# total_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
# print(f"total GPU training time: {total_time:.2f} 秒")
# print(f"per epoch trianing time: {total_time / num_epochs:.2f} 秒")