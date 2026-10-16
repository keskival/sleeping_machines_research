"""Exact carried-state streaming for the deep keyed temporal-memory members (r1_deep_keyed_lm.Block stacks with the output
keyed token-slot bonus; r1_ptb_native.NativeWordLM).

The window scorer recomputes up to ctx tokens of history for every segment. Here the model consumes the stream once, in
segments, carrying:
- each layer's temporal-memory state (complex, per mode) indefinitely: the exact recurrence across segment boundaries;
- each layer's keyed-read keys/values and the output bonus keys and slot tokens for the last `mem` events (race reads and
  the slot bonus see that many past events; the window scorer's ctx plays the same role there);
- the last event's embedding (the predecessor message of the next segment's first event).
With mem >= stream length this equals scoring the whole stream as one lane (contract). Cached keys are detached: in the
online arm a segment's gradient flows through that segment's computation only, and cached history was computed with the
weights in force when it was read, as in recurrent dynamic evaluation.
"""
import math
import torch
import torch.nn.functional as F


def tm_step(tm, u, state):
    """ParallelTemporalMemory.forward with an initial complex state; returns (output, final state)."""
    B, L, _ = u.shape; g = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :tm.n]*g, w[..., tm.n:]*g
    rate, freq = tm.log_rate.exp(), tm.freq; C = min(tm.CHUNK, L)
    k = torch.arange(C, dtype=u.dtype); lag = k[:, None] - k[None, :]; causal = (lag >= 0).to(u.dtype); lagc = lag.clamp_min(0).unsqueeze(-1)
    mag = torch.exp(-rate*lagc)*causal.unsqueeze(-1); ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)
    steps = (k + 1).unsqueeze(-1); cr, ci = torch.exp(-rate*steps)*torch.cos(freq*steps), torch.exp(-rate*steps)*torch.sin(freq*steps)
    sr, si = state if state is not None else (u.new_zeros(B, tm.n), u.new_zeros(B, tm.n)); outs = []
    for c0 in range(0, L, C):
        ar, ai = wr[:, c0:c0 + C], wi[:, c0:c0 + C]; n_ = ar.shape[1]; Kr, Ki = kr[:n_, :n_], ki[:n_, :n_]
        zr = torch.einsum('ijn,bjn->bin', Kr, ar) - torch.einsum('ijn,bjn->bin', Ki, ai); zi = torch.einsum('ijn,bjn->bin', Kr, ai) + torch.einsum('ijn,bjn->bin', Ki, ar)
        zr = zr + cr[:n_]*sr.unsqueeze(1) - ci[:n_]*si.unsqueeze(1); zi = zi + cr[:n_]*si.unsqueeze(1) + ci[:n_]*sr.unsqueeze(1)
        outs.append(torch.cat([zr, zi], -1)); sr, si = zr[:, -1], zi[:, -1]
    return torch.cat(outs, 1), (sr, si)


def _cat(old, new, mem, dim):
    out = (new if old is None else torch.cat([old, new], dim)).detach()   # carried history never links segment graphs
    return out.narrow(dim, out.shape[dim] - min(mem, out.shape[dim]), min(mem, out.shape[dim]))


class StreamState:
    def __init__(self, model, mem):
        self.model, self.mem = model, mem; L = len(model.blocks)
        self.tm, self.kv = [None]*L, [None]*L; self.out_k = self.tokens = None; self.prev_e = None; self.seen = 0

    def segment(self, x):
        """Logits for every position of the input segment x [1, n] given the carried history; updates the carried state."""
        m = self.model; e = F.embedding(x, m.embed.weight); h = e; n = x.shape[1]
        first = torch.zeros_like(e[:, :1]) if self.prev_e is None else self.prev_e
        prev = torch.cat([first, e[:, :-1]], 1)
        for i, b in enumerate(m.blocks):
            y, st = tm_step(b.tm, b.n1(h), self.tm[i]); self.tm[i] = (st[0].detach(), st[1].detach()); h = h + b.drop(b.tm_out(y))
            r = b.read; u = b.n2(h); kv_in = torch.cat([u, prev], -1)
            q = r.q(u).view(1, n, r.h, r.dk).transpose(1, 2); k = r.k(kv_in).view(1, n, r.h, r.dk).transpose(1, 2); v = r.v(kv_in).view(1, n, r.h, r.dk).transpose(1, 2)
            pk, pv = self.kv[i] if self.kv[i] is not None else (k[:, :, :0], v[:, :, :0]); K, Vv = torch.cat([pk, k], 2), torch.cat([pv, v], 2); P = pk.shape[2]
            dist = (torch.arange(n)[:, None] + P - torch.arange(P + n)[None, :]).to(u.dtype)
            bias = (-r.log_rate.exp().view(1, -1, 1, 1)*dist.clamp_min(0)).masked_fill(dist < 0, float('-inf'))
            a = torch.softmax(q @ K.transpose(-1, -2)/math.sqrt(r.dk) + bias, -1)
            h = h + b.drop(r.o((a @ Vv).transpose(1, 2).reshape(1, n, -1)))
            self.kv[i] = (_cat(pk, k.detach(), self.mem, 2), _cat(pv, v.detach(), self.mem, 2))
            h = h + b.drop(b.mlp(b.n3(h)))
        h = m.out_norm(h)
        # output keyed token-slot bonus over carried + current events (KeyedTokenLM.keyed_scores with a cache)
        kk = m.key_w(torch.cat([h, prev], -1)).view(1, n, m.heads, m.dk); qq = m.query_w(h).view(1, n, m.heads, m.dk)
        pk = self.out_k if self.out_k is not None else kk[:, :0]; K = torch.cat([pk, kk], 1); P = pk.shape[1]
        toks = torch.cat([self.tokens, x[0]]) if self.tokens is not None else x[0]
        dist = (torch.arange(n)[:, None] + P - torch.arange(P + n)[None, :]).to(h.dtype); causal = dist >= 0
        S = h.new_zeros(1, n, P + n)
        for hd in range(m.heads):
            decay = torch.where(causal, torch.exp(-m.log_rate[hd].exp()*dist.clamp_min(0)), torch.zeros_like(dist))
            S = S + torch.einsum('btd,bjd->btj', qq[:, :, hd], K[:, :, hd])*decay/math.sqrt(m.dk)
        logits = (h @ m.embed.weight.T + m.bias).scatter_add(2, toks[None, None, :].expand(1, n, -1), S)
        self.out_k = _cat(pk, kk.detach(), self.mem, 1); self.tokens = toks[-self.mem:]
        self.prev_e = e[:, -1:].detach(); self.seen += n
        return logits


def stream_segments(model, stream, seg, start_token, mem):
    """Yield per-target NLL tensors for consecutive segments of `seg` targets over the whole stream (first target
    conditioned on start_token), carrying state across segments."""
    import numpy as np
    x = np.concatenate([[start_token], stream]); st = StreamState(model, mem)
    for t0 in range(0, len(stream), seg):
        t1 = min(t0 + seg, len(stream)); inp = torch.from_numpy(x[t0:t1][None])
        yield F.cross_entropy(st.segment(inp)[0], torch.from_numpy(x[t0 + 1:t1 + 1]), reduction='none')
