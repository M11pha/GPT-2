import os
import urllib.request
import tiktoken
from model import generate_text_simple, GPTModel
from Dataset import create_dataloader_v1
import torch
import train_util as tu

# 模型配置参数 ---------------------------------------------------------------------------------------
GPT_CONFIG_124M = {
    # vocab_size refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
    "vocab_size": 50257,   # Vocabulary size
    # denotes the maximum number of input tokens the model can handle via the positional embeddings
    "context_length": 256, # Shortened context length (orig: 1024)
    # emb_dim represents the embedding size, transforming each token into a 768-dimensional vector
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
# 读取txt文本并计算字符数与token数 ------------------------------------------------------------------------------
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


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters) # Characters: 20479
print("Tokens:", total_tokens) # Tokens: 5145 都是数字

# 将文本切分为 9:1 的训练集与验证集 ----------------------------------------------------------------------------
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

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# 初始化模型并查看初始损失 ----------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

for x, y in val_loader:
    print(x.shape, y.shape)

    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader
    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = tu.calc_loss_loader(train_loader, model, device)
        val_loss   = tu.calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
# 训练主体 ------------------------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = tu.train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# 保存训练损失曲线 ----------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, filename="loss_plot.png"):
    """
    绘制训练和验证损失，并将图像保存为文件。

    Args:
        epochs_seen: 已完成的epoch数据点。
        tokens_seen: 已处理的token数量数据点。
        train_losses: 训练损失列表。
        val_losses: 验证损失列表。
        filename (str): 保存图像的文件名。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制损失曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建第二个X轴以显示tokens数量
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 使用透明绘图来设置轴
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()

    # --- 主要改动 ---
    # 1. 将图像保存到指定文件，可以设置dpi以获得更高分辨率
    plt.savefig(filename, dpi=300)

    # 2. 关闭图像以释放内存，这在循环中绘图时尤其重要
    plt.close(fig)

# 准备数据
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# 调用修改后的函数，并指定输出文件名
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, filename="training_progress.png")

print("save loss curve at training_progress.png")

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