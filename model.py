import torch

# 参数
batch_size = 32  # 一次处理多少文本序列
block_size = 8  # 序列的长度（上下文窗口大小）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # 嵌入维度
head_size = 16
learning_rate = 1e-3 # 学习率
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

# 这是一个简单的注意力头
class Head(torch.nn.Module):
    def __init__(self, head_size):
        # 每个 token 对应一个 embedding 向量
        # 将其作为输入，用三个线性层来计算 Q, K, V
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # 计算 Q, K, V
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        # 计算注意力分数
        wei = q @ k.transpose(-2,-1) * head_size**-0.5  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        # 在训练过程中，我们需要确保模型只能访问当前时间步之前的上下文信息。
        # 为此，我们使用一个下三角矩阵（tril）来屏蔽掉未来的信息。
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = torch.nn.functional.softmax(wei, dim=-1)  # (B,T,T)

        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class multiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
# 这是一个简单的前馈网络
class FeedForward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, n_embd),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

# 我们定义了一个简单的语言模型
# 它使用一个嵌入层来将输入的字符索引映射到一个向量空间中。
# 这个模型的输出是一个形状为 (batch_size, block_size, vocab_size) 的张量
# 其中每个位置的值表示该位置上每个字符的概率分布。
class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 每个 token 对应一个 embedding 向量
        # 这里相当于一个映射表，输入是字符索引，输出是对应的嵌入向量
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        # 这也可以看成一个映射，内部其实是一个线性层
        # sa: self-attention
        # lm: language model
        self.sa_head = multiHeadAttention(4, n_embd//4)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
        self.ffwd = FeedForward(n_embd)

    # idx, targets 都是 (B,T) 的张量，返回 (B,T,C) 的张量
    # 其中 B 是批量大小，T 是块大小，C 是词汇表大小
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 算出每个位置的 token 嵌入和位置嵌入
        tok_emb = self.token_embedding_table(idx)  # (batch_size, block_size, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (block_size, n_embd)
        # 将 token 嵌入和位置嵌入相加，得到输入到注意力头的张量
        x = tok_emb + pos_emb  # (batch_size, block_size, n_embd)
        x = self.sa_head(x)  # (batch_size, block_size, n_embd)
        x = self.ffwd(x)  # (batch_size, block_size, n_embd)
        logits = self.lm_head(x)  # (batch_size, block_size, vocab_size)

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
            idx_cond = idx[:, -block_size:]  # (B,T) 取最后 block_size 个字符作为输入
            # 前向传播，得到下一个字符的分数
            logits, loss = self(idx_cond)
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

model = BigramLanguageModel()
model = model.to(device)

content = torch.zeros((1,1), dtype=torch.long, device=device)
string = model.generate(content, max_new_tokens=100)
print(decode(string[0].tolist()))

# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
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