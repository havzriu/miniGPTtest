import torch

# 参数
batch_size = 32  # 一次处理多少文本序列
block_size = 8  # 序列的长度（上下文窗口大小）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
print(f'device: {device}')

with open('input.txt', 'r') as f:
    input_text = f.read()

# 这里我们需要统计输入文本中所有不同的字符，并计算它们的数量（即词汇表大小）
chars = sorted(list(set(input_text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# tokenizer | 这是逐字符的
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

data = torch.tensor(encode(input_text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# 现在我们已经有了一个包含输入文本的张量，我们可以将其分割成训练集和验证集
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# 
torch.manual_seed(9977)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 我们定义了一个简单的语言模型
# 它使用一个嵌入层来将输入的字符索引映射到一个向量空间中。
# 这个模型的输出是一个形状为 (batch_size, block_size, vocab_size) 的张量
# 其中每个位置的值表示该位置上每个字符的概率分布。
class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 即每个字符后面的字符的概率分布
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    # idx, targets 都是 (B,T) 的张量，返回 (B,T,C) 的张量
    # 其中 B 是批量大小，T 是块大小，C 是词汇表大小
    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # 计算交叉熵损失
            # 注意 cross_entropy 函数期待得到一个形状为 (N, C) 的输入，其中 N 是样本数量，C 是类别数量
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx 是 (B,T) 的张量，代表当前的上下文
        for _ in range(max_new_tokens):
            # 前向传播，得到下一个字符的分数
            logits, loss = self(idx)
            # 取最后一个时间步的 logits
            logits = logits[:, -1, :]  # (B,C)
            # 从概率分布中采样一个新的字符
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B,C)
            # 这里我们使用 torch.multinomial 从概率分布中采样一个新的字符索引
            idx_next = torch.multinomial(probs, num_samples=1)   # (B,1)
            # 将新采样的字符索引添加到当前的上下文中
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

# 计算当前的 loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel(vocab_size)
model = model.to(device)

content = torch.zeros((1,1), dtype=torch.long, device=device)
string = model.generate(content, max_new_tokens=100)
print(decode(string[0].tolist()))

# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 1000 == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

content = torch.zeros((1,1), dtype=torch.long, device=device)
string_trained = model.generate(content, max_new_tokens=100)
print(decode(string_trained[0].tolist()))