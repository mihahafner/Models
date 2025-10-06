import torch, math, random
import torch.nn as nn
from Transformer import Seq2SeqForecaster

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- toy dataset: noisy sine waves ---
def make_series(n_series=1024, total_len=120):
    xs, ys = [], []
    for _ in range(n_series):
        freq = random.uniform(0.02, 0.08)
        phase = random.uniform(0, 2*math.pi)
        t = torch.arange(total_len, dtype=torch.float)
        s = torch.sin(freq*t + phase) + 0.1*torch.randn(total_len)
        xs.append(s.unsqueeze(-1))
    return torch.stack(xs)  # (N, T, 1)

def make_batches(series, in_len=60, out_len=20, batch=32):
    N, T, _ = series.shape
    while True:
        idx = torch.randint(0, N, (batch,))
        start = torch.randint(0, T - in_len - out_len - 1, (batch,))
        src = torch.stack([series[i, s:s+in_len, :] for i, s in zip(idx, start)], 0)
        tgt = torch.stack([series[i, s+in_len:s+in_len+out_len, :] for i, s in zip(idx, start)], 0)
        # teacher forcing input: start with last source value, then previous targets
        first = src[:, -1:, :]
        tgt_in = torch.cat([first, tgt[:, :-1, :]], dim=1)
        yield src.to(device), tgt_in.to(device), tgt.to(device)

# --- model, loss, optim ---
model = Seq2SeqForecaster(in_dim=1, out_dim=1, d_model=64, n_heads=4, d_ff=128, n_enc=2, n_dec=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

series = make_series()
loader = make_batches(series, in_len=60, out_len=20, batch=64)

# --- train a few steps ---
model.train()
for step in range(800):
    src, tgt_in, tgt = next(loader)
    pred = model(src, tgt_in)
    loss = loss_fn(pred, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step+1) % 100 == 0:
        print(f"step {step+1}: loss {loss.item():.4f}")

# --- quick inference ---
model.eval()
src, tgt_in, tgt = next(loader)
first_token = src[:, -1:, :]     # start decoder from last known point
pred_future = model.autoregressive(src, first_token, steps=20)  # (B, 20, 1)
print("Pred shape:", pred_future.shape)
