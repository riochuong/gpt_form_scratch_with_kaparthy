import torch
import torch.nn as nn
from torch.nn import functional as F


# ------ hyperparamerts --------
batch_size = 64
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
# ----------------------------------

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from chars to int 
stoi = {s : i for i,s in enumerate(sorted(chars))}
itos = {i: s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[j] for j in i])

# Split data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
print("Data Info: ", data.shape, data.dtype)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

# data loading 
def get_batch(split):
    data = train if split == 'train' else val
    ix_in_batch = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix_in_batch])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix_in_batch])
    x, y = x.to(device), y.to(device)
    return x,y 


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Block(nn.Module):
    """ Transformer block: Communication follow by Computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # LayerNorm will normalie the rows instead of collumns like BatchNorm
        # it normalizes/transforms features vector , B, T are acting just like Batch Dimension
        # LayerNorm also has gamma, beta trainable params that could eventually create output that 
        # is not unit Gaussian but the optimization process will determine that  
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # residual network https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
        x = x + self.ffwd(self.ln2(x))  # residual network, fork off do some communication and comeback 
        return x
    

class FeedForward(nn.Module):
    """ Simple feed-forward layer followed by non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # projection back to the residual path way to be added with the input 
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table =  nn.Embedding(block_size, n_embd)
        # we are now having 4 communication channels with multi-head attentions
        # we concatenate all heads out put to give us 32 
        # this is similar to group convolution 
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim self-attention
        # self.blocks = nn.Sequential(
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) -> adding pos_emb to each batch of token
        #x = self.sa_heads(x) # apply multihead of self-attention. (B,T,C)
        x = self.blocks(x) # interpersing computation and communications many times with self-attention heads
        x = self.ln_f(x)
        # finally we decode the logics jjere         
        logits = self.lm_head(x) # (B, T, vocabsize)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions 
            logits, loss = self(idx_cond)
            # focus only on the last time step 
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, 1)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel(vocab_size=vocab_size)

class Head(nn.Module):
    """ one head of self-attentiion """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.droput = nn.Dropout(dropout)
        # tril is not a parameter, it's more like a helper matrix 
        # for us to prevent data from passing from the future to current token
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # key (B, T, C) @ (B, C, head_size) -> (B, T, head_size)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.droput(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # this give more channels for communication between tokens 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
        # projection back to input residual path way 
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # the projection is just the linear transformation of the self-attention layer
        return self.dropout(self.proj(out))


m = model.to(device)
# create a pytorch optimizer 
optimizer = torch.optim.AdamW()

#---- TRAINING LOOP -----
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch("train")

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generat from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))