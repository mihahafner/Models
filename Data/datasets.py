import math, random, torch

def make_sine_dataset(n_series=1024, total_len=160):
    xs = []
    for _ in range(n_series):
        f = random.uniform(0.02, 0.08)
        phi = random.uniform(0, 2*math.pi)
        t = torch.arange(total_len, dtype=torch.float)
        s = torch.sin(f*t + phi) + 0.1*torch.randn(total_len)
        xs.append(s.unsqueeze(-1))
    return torch.stack(xs)

def timeseries_batcher(series, in_len=60, out_len=20, batch=64, device="cpu"):
    N, T, _ = series.shape
    def _next():
        idx = torch.randint(0, N, (batch,))
        start = torch.randint(0, T - in_len - out_len - 1, (batch,))
        src = torch.stack([series[i, s:s+in_len, :] for i, s in zip(idx, start)], 0)
        tgt = torch.stack([series[i, s+in_len:s+in_len+out_len, :] for i, s in zip(idx, start)], 0)
        first = src[:, -1:, :]
        tgt_in = torch.cat([first, tgt[:, :-1, :]], 1)
        return src.to(device), tgt_in.to(device), tgt.to(device)
    return _next
