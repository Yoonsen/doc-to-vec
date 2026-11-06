"""Benchmark: document similarity on sparse CPU vs. sparse/dense/projected GPU.

Compares book word-frequency representations for one-vs-many similarity,
all-vs-all top-k, PMI transform and exact JSD/Hellinger/cosine, and measures
recall@k of approximate methods against the exact sparse result. Also
prototypes a hybrid: projected GPU candidates -> exact, traceable PartialFunc
rerank (explain_dot / explain_jsd_divergence).

Backends
  pf             PartialFunc (Roaring BitMap + aligned float64 ndarray) + SparseWorkspace,
                 single-threaded Python loop (from the partial-functions repo)
  scipy          scipy.sparse CSR (single-threaded SpMV / SpGEMM, numpy-vectorised JSD)
  numba_csr      CSR + scattered query workspace, Numba prange over books (40 threads)
  torch_sparse   torch CSR on CUDA (cuSPARSE SpMV/SpMM, gather + segment_reduce for JSD)
  dense{K}_{fp}  dense GPU matrix over the K most frequent words (fp32 / fp16)
  rp{d}          sparse JL / count-sketch projection of the full vocabulary to d dims (GPU)
  svd{d}         randomized truncated SVD (fit on <= 20k books, vocab capped at 200k), fold-in
  hybrid_*       projected candidates (GPU) -> exact PartialFunc rerank on CPU

Measures
  cos_tf     cosine on raw counts
  cos_ppmi   cosine on Shifted PPMI (alpha=0.85, k=1) against corpus background
  hellinger  H = sqrt(1 - sum_{I} sqrt(p q))   (a dot product on sqrt(p): GEMM-able)
  jsd        JSD in bits over the intersection (not GEMM-able)

Data sources
  synthetic  Zipf books: background = totals_500k frequencies, plus n/50 sparse topics,
             lognormal book length (median 60k tokens); ~15k distinct words per book;
             one topic per ~50 books so every book has more than k true neighbours
  shard      CSR .npz made by extract_shard_counts.py (real counts, no text)
  mat        mat_bundle.json (DDK 641, 338 books)

Only counts and integer ids are used; no text is read or printed.

    python bench_similarity_gpu.py --source synthetic --n-books 10000 --out results.jsonl
    python bench_similarity_gpu.py --report results.jsonl > SIMILARITY_BENCHMARK_tables.md
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

PF_DIR = os.environ.get("PARTIAL_FUNCTIONS_DIR", str(Path(__file__).resolve().parent.parent / "partial-functions"))
sys.path.insert(0, PF_DIR)
from partial_func import PartialFunc, SparseWorkspace  # noqa: E402

K_TOP = 10
MEASURES = ("cos_tf", "cos_ppmi", "hellinger", "jsd")
SMALLER_IS_BETTER = {"jsd"}


# ---------------------------------------------------------------------------
# Synthetic Zipf corpus (Numba, deterministic per book)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _book_tokens(b, seed, cum_bg, topic_words, topic_cum, mu_log_len):
    np.random.seed(seed * 1_000_003 + b)
    L = int(math.exp(np.random.normal(mu_log_len, 0.6)))
    L = max(5_000, min(L, 400_000))
    T = topic_words.shape[0]
    t1 = np.random.randint(T)
    t2 = (t1 + 1 + np.random.randint(3)) % T  # neighbouring topic -> graded similarity
    lam1 = np.random.uniform(0.05, 0.25)
    lam2 = np.random.uniform(0.0, 0.10)
    tok = np.empty(L, dtype=np.int32)
    for i in range(L):
        u = np.random.random()
        if u < lam1:
            tok[i] = topic_words[t1, np.searchsorted(topic_cum, np.random.random())]
        elif u < lam1 + lam2:
            tok[i] = topic_words[t2, np.searchsorted(topic_cum, np.random.random())]
        else:
            tok[i] = np.searchsorted(cum_bg, np.random.random())
    tok.sort()
    return tok


@njit(parallel=True, cache=True)
def _gen_counts(n, seed, cum_bg, topic_words, topic_cum, mu_log_len, nnz_out):
    for b in prange(n):
        tok = _book_tokens(b, seed, cum_bg, topic_words, topic_cum, mu_log_len)
        c = 1
        for i in range(1, len(tok)):
            if tok[i] != tok[i - 1]:
                c += 1
        nnz_out[b] = c


@njit(parallel=True, cache=True)
def _gen_fill(n, seed, cum_bg, topic_words, topic_cum, mu_log_len, indptr, indices, data):
    for b in prange(n):
        tok = _book_tokens(b, seed, cum_bg, topic_words, topic_cum, mu_log_len)
        p = indptr[b]
        indices[p] = tok[0]
        data[p] = 1.0
        for i in range(1, len(tok)):
            if tok[i] != tok[i - 1]:
                p += 1
                indices[p] = tok[i]
                data[p] = 1.0
            else:
                data[p] += 1.0


def make_synthetic(n, vocab, seed, data_dir, books_per_topic=50, topic_size=400, mu_len=60_000):
    tot = Path(data_dir) / "totals_500k.parquet"
    if tot.exists():
        import pandas as pd
        bg = pd.read_parquet(tot, columns=["total_freq"])["total_freq"].to_numpy(np.float64)
        bg = np.sort(bg)[::-1][:vocab]
    else:
        bg = 1.0 / np.arange(1, vocab + 1) ** 1.07
    vocab = len(bg)
    n_topics = max(10, n // books_per_topic)  # ~50 books share a primary topic
    cum_bg = np.cumsum(bg / bg.sum())
    cum_bg[-1] = 1.0
    rng = np.random.default_rng(seed)
    # Topic words: mid/low-frequency words (rank >= 500), biased towards bg^0.6.
    w = bg[500:] ** 0.6
    cdf = np.cumsum(w / w.sum())
    cdf[-1] = 1.0
    topic_words = (500 + np.searchsorted(cdf, rng.random(n_topics * topic_size))).astype(np.int32)
    topic_words = topic_words.reshape(n_topics, topic_size)
    tw = 1.0 / np.arange(1, topic_size + 1) ** 0.8
    topic_cum = np.cumsum(tw / tw.sum())
    topic_cum[-1] = 1.0
    mu_log = math.log(mu_len)
    nnz = np.zeros(n, dtype=np.int64)
    _gen_counts(n, seed, cum_bg, topic_words, topic_cum, mu_log, nnz)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nnz, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=np.int32)
    data = np.empty(indptr[-1], dtype=np.float32)
    _gen_fill(n, seed, cum_bg, topic_words, topic_cum, mu_log, indptr, indices, data)
    return indptr, indices, data, vocab


def load_source(args):
    if args.source == "synthetic":
        return make_synthetic(args.n_books, args.vocab, args.seed, args.data_dir)
    if args.source == "shard":
        z = np.load(args.shard_npz)
        indptr, indices, data = z["indptr"], z["indices"], z["data"]
        vocab = int(z["shape"][1])
        if args.n_books and args.n_books < len(indptr) - 1:
            n = args.n_books
            indptr = indptr[: n + 1].copy()
            indices, data = indices[: indptr[-1]].copy(), data[: indptr[-1]].copy()
        return indptr.astype(np.int64), indices.astype(np.int32), data.astype(np.float32), vocab
    if args.source == "mat":
        b = json.load(open(args.mat_bundle))
        c = b["csr"]
        tf_bg = np.array([w["tf_bg"] for w in b["words"]], dtype=np.float64)
        order = np.argsort(-tf_bg, kind="stable")
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))
        X = sp.csr_matrix((np.array(c["freq"], np.float32), rank_of[np.array(c["col_idx"])],
                           np.array(c["row_ptr"])), shape=(len(c["row_ptr"]) - 1, len(tf_bg)))
        X.sum_duplicates()  # the bundle can repeat a column within a row
        X.sort_indices()
        return X.indptr.astype(np.int64), X.indices.astype(np.int32), X.data.astype(np.float32), X.shape[1]
    raise ValueError(args.source)


# ---------------------------------------------------------------------------
# CSR weightings (Numba, parallel over books)
# ---------------------------------------------------------------------------

@njit(cache=True)
def k_colsum(indices, data, vocab):
    out = np.zeros(vocab, dtype=np.float64)
    for j in range(len(indices)):
        out[indices[j]] += data[j]
    return out


def make_scipy(indptr, indices, data, shape):
    """CSR without index upcasting copies (int32 indptr when nnz fits)."""
    ip = indptr.astype(np.int32) if indptr[-1] < 2**31 - 1 else indptr
    X = sp.csr_matrix(shape, dtype=np.float32)
    X.indptr, X.indices, X.data = ip, indices, data
    return X


@njit(parallel=True, cache=True)
def k_row_scale(indptr, data, mode):
    """mode 0: L2-normalise, 1: probability, 2: sqrt(probability)."""
    n = len(indptr) - 1
    out = np.empty(len(data), dtype=np.float32)
    for i in prange(n):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            v = data[j]
            s += v * v if mode == 0 else v
        if s <= 0.0:
            for j in range(indptr[i], indptr[i + 1]):
                out[j] = 0.0
            continue
        if mode == 0:
            s = math.sqrt(s)
        for j in range(indptr[i], indptr[i + 1]):
            v = data[j] / s
            out[j] = math.sqrt(v) if mode == 2 else v
    return out


@njit(parallel=True, cache=True)
def k_ppmi(indptr, indices, data, log2bg, alpha, log2k):
    """Shifted PPMI over CSR, positive entries only, rows L2-normalised. Two passes."""
    n = len(indptr) - 1
    cnt = np.zeros(n, dtype=np.int64)
    rs = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j]
        rs[i] = s
        c = 0
        for j in range(indptr[i], indptr[i + 1]):
            if math.log2(data[j] / s) - alpha * log2bg[indices[j]] - log2k > 0.0:
                c += 1
        cnt[i] = c
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        out_ptr[i + 1] = out_ptr[i] + cnt[i]
    out_idx = np.empty(out_ptr[-1], dtype=np.int32)
    out_val = np.empty(out_ptr[-1], dtype=np.float32)
    for i in prange(n):
        p = out_ptr[i]
        nrm = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            v = math.log2(data[j] / rs[i]) - alpha * log2bg[indices[j]] - log2k
            if v > 0.0:
                out_idx[p] = indices[j]
                out_val[p] = v
                nrm += v * v
                p += 1
        nrm = math.sqrt(nrm)
        if nrm > 0.0:
            for q in range(out_ptr[i], out_ptr[i + 1]):
                out_val[q] = out_val[q] / nrm
    return out_ptr, out_idx, out_val


@njit(parallel=True, cache=True)
def k_col_cap(indptr, indices, data, K):
    """Keep columns < K (vocabulary truncation to the K most frequent words)."""
    n = len(indptr) - 1
    cnt = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for j in range(indptr[i], indptr[i + 1]):
            if indices[j] < K:
                c += 1
        cnt[i] = c
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        out_ptr[i + 1] = out_ptr[i] + cnt[i]
    out_idx = np.empty(out_ptr[-1], dtype=np.int32)
    out_val = np.empty(out_ptr[-1], dtype=np.float32)
    for i in prange(n):
        p = out_ptr[i]
        for j in range(indptr[i], indptr[i + 1]):
            if indices[j] < K:
                out_idx[p] = indices[j]
                out_val[p] = data[j]
                p += 1
    return out_ptr, out_idx, out_val


# ---------------------------------------------------------------------------
# Exact one-vs-many kernels over CSR with a scattered query (fp64 accumulation)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def k_dot(indptr, indices, data, work, out):
    for i in prange(len(indptr) - 1):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * work[indices[j]]
        out[i] = s


@njit(parallel=True, cache=True)
def k_jsd(indptr, indices, data, work, out):
    for i in prange(len(indptr) - 1):
        acc = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            q = work[indices[j]]
            if q > 0.0:
                p = np.float64(data[j])
                if p > 0.0:
                    s = p + q
                    acc += p * math.log2(p / s) + q * math.log2(q / s)
        v = 1.0 + 0.5 * acc
        out[i] = min(1.0, max(0.0, v))


class NumbaCSR:
    def __init__(self, indptr, indices, data, vocab):
        self.indptr, self.indices, self.data = indptr, indices, data
        self.work = np.zeros(vocab, dtype=np.float64)
        self.out = np.empty(len(indptr) - 1, dtype=np.float64)

    def row(self, i):
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.data[s:e]

    def scores(self, i, measure):
        idx, val = self.row(i)
        self.work[idx] = val
        if measure == "jsd":
            k_jsd(self.indptr, self.indices, self.data, self.work, self.out)
        else:
            k_dot(self.indptr, self.indices, self.data, self.work, self.out)
        self.work[idx] = 0.0
        return self.out.copy()


def topk_cpu(scores, k, exclude, smaller):
    s = scores.astype(np.float64, copy=True)
    s[exclude] = np.inf if smaller else -np.inf
    if smaller:
        idx = np.argpartition(s, k)[:k]
        return idx[np.argsort(s[idx], kind="stable")]
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


def recall(approx, exact):
    return float(np.mean([len(set(a[:K_TOP]) & set(e[:K_TOP])) / K_TOP for a, e in zip(approx, exact)]))


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path, base):
        self.path, self.base = path, base

    def __call__(self, **kw):
        rec = {**self.base, **kw}
        line = json.dumps(rec)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def mb(nbytes):
    return round(nbytes / 2**20, 1)


def csr_bytes(indptr, indices, data):
    return indptr.nbytes + indices.nbytes + data.nbytes


def time_budget_loop(fn, items, budget_s):
    """Run fn(item) until all done or budget spent. Returns (elapsed, n_done, results)."""
    res = []
    t0 = time.perf_counter()
    for it in items:
        res.append(fn(it))
        if time.perf_counter() - t0 > budget_s:
            break
    return time.perf_counter() - t0, len(res), res


# ---------------------------------------------------------------------------
# PartialFunc backend
# ---------------------------------------------------------------------------

def pf_from_row(indptr, indices, data, i):
    s, e = indptr[i], indptr[i + 1]
    return PartialFunc.from_sorted_arrays(indices[s:e].view(np.uint32), data[s:e])


def pf_bytes(pf):
    return len(pf.dom.serialize()) + pf.val.nbytes


# ---------------------------------------------------------------------------
# GPU helpers (torch)
# ---------------------------------------------------------------------------

def torch_setup():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def sync():
    import torch
    torch.cuda.synchronize()


def gpu_peak_reset():
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb():
    import torch
    torch.cuda.synchronize()
    return mb(torch.cuda.max_memory_allocated())


def to_torch_csr(indptr, indices, data, vocab, dev):
    import torch
    idt = torch.int32 if len(indices) < 2**31 - 1 else torch.int64
    crow = torch.from_numpy(indptr).to(dev, idt)
    col = torch.from_numpy(indices).to(dev, idt)
    val = torch.from_numpy(data).to(dev)
    X = torch.sparse_csr_tensor(crow, col, val, size=(len(indptr) - 1, vocab), device=dev)
    return X, crow, col, val


def torch_row_ids(crow_np, s, e, dev):
    import torch
    lens = np.diff(crow_np[s:e + 1])
    return torch.repeat_interleave(torch.arange(s, e, device=dev), torch.from_numpy(lens).to(dev))


def row_chunks(indptr, max_nnz):
    """Yield (s, e) row ranges with at most ~max_nnz entries each."""
    n = len(indptr) - 1
    s = 0
    while s < n:
        e = int(np.searchsorted(indptr, indptr[s] + max_nnz, side="right")) - 1
        e = max(e, s + 1)
        e = min(e, n)
        yield s, e
        s = e


class TorchSparse:
    """CSR on GPU. Dot measures via cuSPARSE SpMV/SpMM, JSD via gather + segment_reduce."""

    def __init__(self, indptr, indices, data, vocab, dev, max_nnz=200_000_000):
        import torch
        self.t = torch
        self.dev, self.vocab, self.n = dev, vocab, len(indptr) - 1
        self.indptr_np = indptr
        self.X, self.crow, self.col, self.val = to_torch_csr(indptr, indices, data, vocab, dev)
        self.offsets = torch.from_numpy(indptr).to(dev)
        self.chunks = list(row_chunks(indptr, max_nnz))
        self.q = torch.zeros(vocab, device=dev, dtype=torch.float32)

    def _scatter(self, i):
        s, e = int(self.indptr_np[i]), int(self.indptr_np[i + 1])
        idx = self.col[s:e].long()
        self.q[idx] = self.val[s:e]
        return idx

    def scores(self, i, measure):
        t = self.t
        idx = self._scatter(i)
        if measure != "jsd":
            out = t.sparse.mm(self.X, self.q[:, None]).squeeze(1)
        else:
            parts = []
            for s, e in self.chunks:
                a, b = int(self.indptr_np[s]), int(self.indptr_np[e])
                p = self.val[a:b]
                qv = self.q[self.col[a:b].long()]
                ssum = p + qv
                term = t.where(qv > 0, p * t.log2(p / ssum) + qv * t.log2(qv / ssum.clamp_min(1e-30)),
                               t.zeros_like(p))
                parts.append(t.segment_reduce(term, "sum", offsets=self.offsets[s:e + 1] - a))
            out = (1.0 + 0.5 * t.cat(parts)).clamp(0.0, 1.0)
        self.q[idx] = 0.0
        return out

    def topk_block(self, rows, measure, k):
        """All-vs-all block: SpMM against a dense block of query columns."""
        t = self.t
        if measure == "jsd":
            res = []
            for i in rows:
                s = self.scores(i, measure)
                s[i] = float("inf")
                res.append(t.topk(s, k, largest=False).indices)
            return t.stack(res)
        B = len(rows)
        Q = t.zeros(self.vocab, B, device=self.dev, dtype=t.float32)
        for j, i in enumerate(rows):
            s, e = int(self.indptr_np[i]), int(self.indptr_np[i + 1])
            Q[self.col[s:e].long(), j] = self.val[s:e]
        S = t.sparse.mm(self.X, Q)  # N x B
        S[t.as_tensor(rows, device=self.dev), t.arange(B, device=self.dev)] = -float("inf")
        return t.topk(S, k, dim=0).indices.T


class TorchDense:
    """Dense N x K matrix on GPU over the K most frequent words."""

    def __init__(self, indptr, indices, data, K, dev, dtype, measure, max_nnz=100_000_000):
        import torch
        t = self.t = torch
        self.measure = measure
        cp, ci, cv = k_col_cap(indptr, indices, data, K)
        if measure in ("hellinger", "jsd"):  # renormalise the truncated distribution
            cv = k_row_scale(cp, cv * cv if measure == "hellinger" else cv, 1)
            if measure == "hellinger":
                cv = np.sqrt(cv)
        else:
            cv = k_row_scale(cp, cv, 0)
        n = len(cp) - 1
        self.M = t.zeros(n, K, device=dev, dtype=dtype)
        for s, e in row_chunks(cp, max_nnz):
            a, b = int(cp[s]), int(cp[e])
            rows = torch_row_ids(cp, s, e, dev)
            cols = t.from_numpy(ci[a:b]).to(dev).long()
            self.M[rows, cols] = t.from_numpy(cv[a:b]).to(dev, dtype)
        self.n = n

    def nbytes(self):
        return self.M.numel() * self.M.element_size()

    def scores(self, i, measure, row_block=8192):
        t = self.t
        q = self.M[i]
        if measure != "jsd":
            return (self.M @ q).float()
        qf = q.float()
        out = []
        for s in range(0, self.n, row_block):
            P = self.M[s:s + row_block].float()
            Ssum = P + qf
            both = (P > 0) & (qf > 0)
            term = t.where(both, P * t.log2(P.clamp_min(1e-30) / Ssum) + qf * t.log2(qf.clamp_min(1e-30) / Ssum),
                           t.zeros((), device=P.device))
            out.append(term.sum(1))
        return (1.0 + 0.5 * t.cat(out)).clamp(0, 1)

    def topk_block(self, rows, measure, k):
        t = self.t
        if measure == "jsd":
            res = []
            for i in rows:
                s = self.scores(i, measure)
                s[i] = float("inf")
                res.append(t.topk(s, k, largest=False).indices)
            return t.stack(res)
        r = t.as_tensor(rows, device=self.M.device)
        S = (self.M[r] @ self.M.T).float()
        S[t.arange(len(rows), device=S.device), r] = -float("inf")
        return t.topk(S, k, dim=1).indices


def project_rp(indptr, indices, data, vocab, d, dev, seed=0, s_hash=4, max_nnz=100_000_000):
    """Sparse JL / count-sketch: each word hits s_hash random dims with random signs."""
    import torch as t
    g = t.Generator(device=dev).manual_seed(seed)
    dims = t.randint(0, d, (vocab, s_hash), device=dev, generator=g)
    sign = (t.randint(0, 2, (vocab, s_hash), device=dev, generator=g) * 2 - 1).float() / math.sqrt(s_hash)
    n = len(indptr) - 1
    E = t.zeros(n * d, device=dev)
    for s, e in row_chunks(indptr, max_nnz):
        a, b = int(indptr[s]), int(indptr[e])
        rows = torch_row_ids(indptr, s, e, dev)
        cols = t.from_numpy(indices[a:b]).to(dev).long()
        vals = t.from_numpy(data[a:b]).to(dev)
        for h in range(s_hash):
            E.index_add_(0, rows * d + dims[cols, h], vals * sign[cols, h])
    E = E.view(n, d)
    return t.nn.functional.normalize(E, dim=1)


def project_svd(indptr, indices, data, d, dev, fit_rows=20_000, vocab_cap=200_000, seed=0, n_iter=2):
    """Randomized truncated SVD on a sample of rows (vocab capped), then fold-in X @ V_k."""
    import torch as t
    cp, ci, cv = k_col_cap(indptr, indices, data, vocab_cap)
    n = len(cp) - 1
    V = min(vocab_cap, int(ci.max()) + 1 if len(ci) else 1)
    rng = np.random.default_rng(seed)
    fit = np.sort(rng.choice(n, size=min(n, fit_rows), replace=False))
    Xs = make_scipy(cp, ci, cv, (n, V))[fit]
    Xt = Xs.T.tocsr()
    Xs_t, *_ = to_torch_csr(Xs.indptr.astype(np.int64), Xs.indices.astype(np.int32), Xs.data, V, dev)
    Xt_t, *_ = to_torch_csr(Xt.indptr.astype(np.int64), Xt.indices.astype(np.int32), Xt.data, Xs.shape[0], dev)
    d = min(d, Xs.shape[0], V)  # rank cannot exceed the sample/vocab size
    l = min(d + 16, Xs.shape[0], V)
    g = t.Generator(device=dev).manual_seed(seed)
    Om = t.randn(Xs.shape[0], l, device=dev, generator=g)
    Y = t.sparse.mm(Xt_t, Om)  # V x l (row space)
    for _ in range(n_iter):
        Y, _ = t.linalg.qr(Y)
        Y = t.sparse.mm(Xt_t, t.sparse.mm(Xs_t, Y))
    Qb, _ = t.linalg.qr(Y)  # V x l
    B = t.sparse.mm(Xs_t, Qb)  # S x l  (= Xs Qb)
    _, _, Vh = t.linalg.svd(B, full_matrices=False)
    Vk = Qb @ Vh[:d].T  # V x d
    del Xs_t, Xt_t, Y, Qb, B, Om
    E = t.empty(n, d, device=dev)
    for s, e in row_chunks(cp, 100_000_000):
        a, b = int(cp[s]), int(cp[e])
        sub, *_ = to_torch_csr((cp[s:e + 1] - a).astype(np.int64), ci[a:b], cv[a:b], V, dev)
        E[s:e] = t.sparse.mm(sub, Vk)
        del sub
    return t.nn.functional.normalize(E, dim=1)


def emb_topk(E, rows, k, block=4096):
    import torch as t
    out = []
    for s in range(0, len(rows), block):
        r = t.as_tensor(rows[s:s + block], device=E.device)
        S = E[r] @ E.T
        S[t.arange(len(r), device=E.device), r] = -float("inf")
        out.append(t.topk(S, k, dim=1).indices)
    return t.cat(out)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def weighting(measure, indptr, indices, data, log2bg):
    if measure == "cos_tf":
        return indptr, indices, k_row_scale(indptr, data, 0)
    if measure == "cos_ppmi":
        return k_ppmi(indptr, indices, data, log2bg, 0.85, 0.0)
    if measure == "hellinger":
        return indptr, indices, k_row_scale(indptr, data, 2)
    if measure == "jsd":
        return indptr, indices, k_row_scale(indptr, data, 1)
    raise ValueError(measure)


def run(args):
    t0 = time.perf_counter()
    indptr, indices, data, vocab = load_source(args)
    n = len(indptr) - 1
    nnz = int(indptr[-1])
    t_load = time.perf_counter() - t0
    label = f"{args.source}_{n}"
    log = Log(args.out, {"source": args.source, "n_books": n, "vocab": vocab, "nnz": nnz, "label": label})
    log(backend="data", op="load_or_generate", seconds=round(t_load, 3), csr_mb=mb(csr_bytes(indptr, indices, data)),
        mean_distinct=round(nnz / n, 1))

    rng = np.random.default_rng(args.seed + 1)
    qe = np.sort(rng.choice(n, size=min(n, args.eval_queries), replace=False))
    budget = args.budget

    # Background distribution for PMI: corpus column totals.
    colsum = k_colsum(indices, data, vocab)
    log2bg = np.log2(np.maximum(colsum, 1e-300) / colsum.sum())

    use_gpu = not args.no_gpu
    if use_gpu:
        torch = torch_setup()
        dev = torch.device("cuda")
        torch.zeros(1, device=dev)

    # ---------------- PMI transform timings ----------------
    t = time.perf_counter()
    pp = k_ppmi(indptr, indices, data, log2bg, 0.85, 0.0)  # warm (jit)
    t = time.perf_counter()
    pp = k_ppmi(indptr, indices, data, log2bg, 0.85, 0.0)
    log(backend="numba_csr", op="ppmi_transform", seconds=round(time.perf_counter() - t, 3),
        out_nnz=int(pp[0][-1]))
    del pp
    if nnz <= args.max_numpy_nnz:
        t = time.perf_counter()
        rs = np.add.reduceat(data.astype(np.float64), indptr[:-1])
        lens = np.diff(indptr)
        v = np.log2(data / np.repeat(rs, lens)) - 0.85 * log2bg[indices]
        X = sp.csr_matrix((np.maximum(v, 0).astype(np.float32), indices.copy(), indptr.copy()), shape=(n, vocab))
        X.eliminate_zeros()
        X = sp.csr_matrix(X.multiply(1.0 / np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel() + 1e-30)[:, None]))
        log(backend="scipy", op="ppmi_transform", seconds=round(time.perf_counter() - t, 3), out_nnz=int(X.nnz))
        del X, v, rs
    # PartialFunc: per-book .pmi() against a background PartialFunc (sample, extrapolated).
    bg_pf = PartialFunc.from_sorted_arrays(np.flatnonzero(colsum).astype(np.uint32),
                                           colsum[colsum > 0] / colsum.sum())
    samp = qe[: min(len(qe), 200)]
    rows_pf = [pf_from_row(indptr, indices, data, i) for i in samp]
    rows_pf[0].pmi(bg_pf, alpha=0.85, shift_k=1.0)
    t = time.perf_counter()
    for f in rows_pf:
        f.pmi(bg_pf, alpha=0.85, shift_k=1.0)
    dt = (time.perf_counter() - t) / len(rows_pf)
    log(backend="pf", op="ppmi_transform", seconds=round(dt * n, 3), extrapolated=True, per_book_ms=round(dt * 1e3, 3))
    del rows_pf
    if use_gpu:
        Xg = TorchSparse(indptr, indices, data, vocab, dev)
        lb = torch.from_numpy(log2bg.astype(np.float32)).to(dev)

        def gpu_ppmi():
            rs = torch.segment_reduce(Xg.val, "sum", offsets=Xg.offsets)
            out = []
            for s, e in Xg.chunks:
                a, b = int(indptr[s]), int(indptr[e])
                rid = torch_row_ids(indptr, s, e, dev)
                v = torch.log2(Xg.val[a:b] / rs[rid]) - 0.85 * lb[Xg.col[a:b].long()]
                out.append(v.clamp_min(0))
            v = torch.cat(out)
            nrm = torch.segment_reduce(v * v, "sum", offsets=Xg.offsets).sqrt()
            keep = v > 0
            return keep.sum()

        gpu_ppmi()
        gpu_peak_reset()
        sync()
        t = time.perf_counter()
        nk = gpu_ppmi()
        sync()
        log(backend="torch_sparse", op="ppmi_transform", seconds=round(time.perf_counter() - t, 3),
            gpu_peak_mb=gpu_peak_mb(), out_nnz=int(nk))
        del Xg, lb
        torch.cuda.empty_cache()

    emb_cache = {}
    measures = args.measures.split(",")
    for measure in measures:
        smaller = measure in SMALLER_IS_BETTER
        t = time.perf_counter()
        wp, wi, wv = weighting(measure, indptr, indices, data, log2bg)
        log(backend="numba_csr", op="weighting", measure=measure, seconds=round(time.perf_counter() - t, 3))
        W = NumbaCSR(wp, wi, wv, vocab)

        # ---- Exact ground truth (numba_csr, fp64) + one-vs-many timing ----
        W.scores(int(qe[0]), measure)  # jit warm-up
        t = time.perf_counter()
        exact = [topk_cpu(W.scores(int(i), measure), K_TOP, int(i), smaller) for i in qe]
        dt = (time.perf_counter() - t) / len(qe)
        log(backend="numba_csr", op="one_vs_many", measure=measure, per_query_ms=round(dt * 1e3, 3),
            repr_mb=mb(csr_bytes(wp, wi, wv)), threads=int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count())))
        log(backend="numba_csr", op="all_vs_all_topk", measure=measure, seconds=round(dt * n, 2),
            extrapolated=True, n_queries_measured=len(qe), recall_at_10=1.0)

        # ---- scipy ----
        Xs = make_scipy(wp, wi, wv, (n, vocab))
        if measure != "jsd":
            qv = np.zeros(vocab, dtype=np.float32)

            def sc(i):
                idx, val = W.row(i)
                qv[idx] = val
                r = Xs @ qv
                qv[idx] = 0
                return r
            sc(int(qe[0]))
            el, nd, res = time_budget_loop(lambda i: sc(int(i)), qe, budget)
            log(backend="scipy", op="one_vs_many", measure=measure, per_query_ms=round(el / nd * 1e3, 3),
                repr_mb=mb(csr_bytes(wp, wi, wv)))
            # all-vs-all via SpGEMM blocks
            if nnz <= args.max_numpy_nnz:
                XT = Xs.T.tocsr()
                B = 64
                blocks = [qe[s:s + B] for s in range(0, len(qe), B)]

                def blk(rows):
                    S = (Xs[rows] @ XT).toarray()
                    S[np.arange(len(rows)), rows] = -np.inf
                    return np.argpartition(-S, K_TOP, axis=1)[:, :K_TOP]
                el, nd, _ = time_budget_loop(blk, blocks, budget)
                done = sum(len(b) for b in blocks[:nd])
                log(backend="scipy", op="all_vs_all_topk", measure=measure, seconds=round(el / done * n, 2),
                    extrapolated=done < n, n_queries_measured=done)
                del XT
        elif nnz <= args.max_numpy_nnz:
            qv = np.zeros(vocab, dtype=np.float64)

            def sj(i):
                idx, val = W.row(i)
                qv[idx] = val
                q = qv[wi]
                m = q > 0
                p = wv[m].astype(np.float64)
                qq = q[m]
                s = p + qq
                term = np.zeros(len(wi))
                term[m] = p * np.log2(p / s) + qq * np.log2(qq / s)
                r = 1.0 + 0.5 * np.add.reduceat(term, wp[:-1])
                qv[idx] = 0
                return r
            el, nd, _ = time_budget_loop(lambda i: sj(int(i)), qe, budget)
            log(backend="scipy", op="one_vs_many", measure=measure, per_query_ms=round(el / nd * 1e3, 3),
                note="numpy-vectorised gather over CSR")
        del Xs

        # ---- PartialFunc + SparseWorkspace (single-threaded) ----
        npf = min(n, args.pf_books)
        t = time.perf_counter()
        pfs = [pf_from_row(wp, wi, wv, i) for i in range(npf)]
        tb = time.perf_counter() - t
        pf_mb = sum(pf_bytes(f) for f in pfs[:2000]) / min(npf, 2000) * n / 2**20
        log(backend="pf", op="build", measure=measure, seconds=round(tb / npf * n, 2), extrapolated=npf < n,
            repr_mb=round(pf_mb, 1))
        ws = SparseWorkspace(vocab)

        def pf_one(i):
            ws.scatter(pf_from_row(wp, wi, wv, i))
            if measure == "jsd":
                return [ws.dot_and_jsd_with_active(g)[1] for g in pfs]
            return [ws.dot_with_active(g) for g in pfs]
        pf_one(int(qe[0]))
        el, nd, res = time_budget_loop(lambda i: pf_one(int(i)), qe, budget)
        per_q = el / nd * n / npf
        # consistency with numba_csr on the first query (restricted to the pf sample)
        ref = W.scores(int(qe[0]), measure)[:npf]
        maxdiff = float(np.max(np.abs(np.asarray(res[0]) - ref)))
        log(backend="pf", op="one_vs_many", measure=measure, per_query_ms=round(per_q * 1e3, 3),
            extrapolated=npf < n, max_abs_diff_vs_numba=maxdiff)
        log(backend="pf", op="all_vs_all_topk", measure=measure, seconds=round(per_q * n, 2), extrapolated=True,
            recall_at_10=1.0)
        del pfs
        gc.collect()

        if not use_gpu:
            continue

        # ---- torch sparse (cuSPARSE) ----
        gpu_peak_reset()
        t = time.perf_counter()
        G = TorchSparse(wp, wi, wv, vocab, dev)
        sync()
        log(backend="torch_sparse", op="build", measure=measure, seconds=round(time.perf_counter() - t, 3),
            gpu_peak_mb=gpu_peak_mb())
        G.scores(int(qe[0]), measure)
        gpu_peak_reset()

        def g_one(i):
            s = G.scores(int(i), measure)
            sync()
            return s
        el, nd, _ = time_budget_loop(g_one, qe, budget)
        log(backend="torch_sparse", op="one_vs_many", measure=measure, per_query_ms=round(el / nd * 1e3, 3),
            gpu_peak_mb=gpu_peak_mb())
        B = args.gpu_block if measure != "jsd" else 1
        B = min(B, max(1, int(4e9 / 4 / vocab)))  # dense query block <= 4 GB
        all_rows = np.arange(n)
        blocks = [all_rows[s:s + B] for s in range(0, n, B)]
        gpu_peak_reset()
        approx = {}

        def g_blk(rows):
            r = G.topk_block(rows, measure, K_TOP).cpu().numpy()
            return rows, r
        el, nd, res = time_budget_loop(g_blk, blocks, budget * 2)
        done = sum(len(b) for b in blocks[:nd])
        log(backend="torch_sparse", op="all_vs_all_topk", measure=measure, seconds=round(el / done * n, 2),
            extrapolated=done < n, n_queries_measured=done, gpu_peak_mb=gpu_peak_mb())
        # recall check of torch_sparse on the eval queries
        ap = G.topk_block(list(qe), measure, K_TOP).cpu().numpy() if measure != "jsd" else \
            np.stack([G.topk_block([int(i)], measure, K_TOP).cpu().numpy()[0] for i in qe])
        log(backend="torch_sparse", op="recall", measure=measure, recall_at_10=recall(ap, exact))
        del G
        torch.cuda.empty_cache()

        # ---- dense GPU over top-K vocabulary ----
        for dt_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            K = min(args.dense_k, vocab)
            need = n * K * (4 if dt_name == "fp32" else 2)
            if need > args.max_gpu_bytes:
                log(backend=f"dense{K // 1000}k_{dt_name}", op="skip", measure=measure,
                    note=f"needs {need / 2**30:.1f} GB > limit")
                continue
            gpu_peak_reset()
            t = time.perf_counter()
            D = TorchDense(wp, wi, wv, K, dev, dtype, measure)
            sync()
            name = f"dense{K // 1000}k_{dt_name}"
            log(backend=name, op="build", measure=measure, seconds=round(time.perf_counter() - t, 3),
                repr_mb=mb(D.nbytes()), gpu_peak_mb=gpu_peak_mb())
            D.scores(int(qe[0]), measure)
            gpu_peak_reset()

            def d_one(i):
                s = D.scores(int(i), measure)
                sync()
                return s
            el, nd, _ = time_budget_loop(d_one, qe, budget)
            log(backend=name, op="one_vs_many", measure=measure, per_query_ms=round(el / nd * 1e3, 3),
                gpu_peak_mb=gpu_peak_mb())
            Bd = 4096 if measure != "jsd" else 1
            blocks = [all_rows[s:s + Bd] for s in range(0, n, Bd)]
            gpu_peak_reset()
            el, nd, _ = time_budget_loop(lambda r: D.topk_block(r, measure, K_TOP).cpu(), blocks, budget * 2)
            done = sum(len(b) for b in blocks[:nd])
            log(backend=name, op="all_vs_all_topk", measure=measure, seconds=round(el / done * n, 2),
                extrapolated=done < n, n_queries_measured=done, gpu_peak_mb=gpu_peak_mb())
            if measure == "jsd":
                ap = np.stack([D.topk_block([int(i)], measure, K_TOP).cpu().numpy()[0] for i in qe])
            else:
                ap = D.topk_block(list(qe), measure, K_TOP).cpu().numpy()
            log(backend=name, op="recall", measure=measure, recall_at_10=recall(ap, exact))
            del D
            torch.cuda.empty_cache()

        # ---- projections (dense GPU, d dims) ----
        if measure in ("cos_ppmi", "hellinger", "jsd"):
            src = "hellinger" if measure == "jsd" else measure
            for kind in ("rp", "svd"):
                for d in args.proj_dims:
                    key = (src, kind, d)
                    if key not in emb_cache:
                        gpu_peak_reset()
                        t = time.perf_counter()
                        E = project_rp(wp, wi, wv if measure != "jsd" else np.sqrt(wv), vocab, d, dev) if kind == "rp" \
                            else project_svd(wp, wi, wv if measure != "jsd" else np.sqrt(wv), d, dev)
                        sync()
                        log(backend=f"{kind}{d}", op="build", measure=src, seconds=round(time.perf_counter() - t, 3),
                            repr_mb=mb(E.numel() * 2), gpu_peak_mb=gpu_peak_mb(), note="fp16 storage size")
                        emb_cache[key] = E.half()
                        del E
                        torch.cuda.empty_cache()
                    E = emb_cache[key]
                    gpu_peak_reset()
                    sync()
                    t = time.perf_counter()
                    nb = emb_topk(E, np.arange(n), K_TOP).cpu().numpy()
                    sync()
                    log(backend=f"{kind}{d}", op="all_vs_all_topk", measure=measure,
                        seconds=round(time.perf_counter() - t, 3), gpu_peak_mb=gpu_peak_mb())
                    log(backend=f"{kind}{d}", op="recall", measure=measure, recall_at_10=recall(nb[qe], exact))

            # ---- hybrid: projected candidates -> exact PartialFunc rerank + explanation ----
            for kind, d in args.hybrid_proj:
                E = emb_cache.get((src, kind, d))
                if E is None:
                    continue
                for C in args.hybrid_c:
                    t = time.perf_counter()
                    cand = emb_topk(E, qe, C).cpu().numpy()
                    sync()
                    t_cand = (time.perf_counter() - t) / len(qe)
                    pf_cache = {}
                    t = time.perf_counter()
                    for i, cs in zip(qe, cand):
                        for j in (int(i), *map(int, cs)):
                            if j not in pf_cache:
                                pf_cache[j] = pf_from_row(wp, wi, wv, j)
                    t_build = (time.perf_counter() - t) / len(qe)
                    approx_nb = []
                    t = time.perf_counter()
                    for i, cs in zip(qe, cand):
                        ws.scatter(pf_cache[int(i)])
                        if measure == "jsd":
                            sc = np.array([ws.dot_and_jsd_with_active(pf_cache[int(j)])[1] for j in cs])
                            order = np.argsort(sc, kind="stable")
                        else:
                            sc = np.array([ws.dot_with_active(pf_cache[int(j)]) for j in cs])
                            order = np.argsort(-sc, kind="stable")
                        approx_nb.append(cs[order[:K_TOP]])
                    t_rerank = (time.perf_counter() - t) / len(qe)
                    # Traceability: decompose the score of each final neighbour into per-word terms.
                    t = time.perf_counter()
                    nexp = min(len(qe), 20)
                    for i, nbs in zip(qe[:nexp], approx_nb[:nexp]):
                        f = pf_cache[int(i)]
                        for j in nbs:
                            g = pf_cache[int(j)]
                            h = f.explain_jsd_divergence(g) if measure == "jsd" else f.explain_dot(g)
                            h.top_k(10)
                    t_explain = (time.perf_counter() - t) / nexp
                    log(backend=f"hybrid_{kind}{d}_C{C}", op="hybrid_query", measure=measure,
                        per_query_ms=round((t_cand + t_rerank) * 1e3, 3), cand_ms=round(t_cand * 1e3, 3),
                        rerank_ms=round(t_rerank * 1e3, 3), pf_build_ms=round(t_build * 1e3, 3),
                        explain_top10_ms=round(t_explain * 1e3, 3),
                        recall_at_10=recall(approx_nb, exact))
                    log(backend=f"hybrid_{kind}{d}_C{C}", op="all_vs_all_topk", measure=measure,
                        seconds=round((t_cand + t_rerank + t_build) * n, 2), extrapolated=True,
                        recall_at_10=recall(approx_nb, exact), note="incl. lazy PartialFunc build from CSR")
                    del pf_cache
        if measure == "jsd":
            emb_cache.clear()
        del W, wp, wi, wv
        gc.collect()

    if use_gpu:
        log(backend="gpu", op="total_peak", gpu_peak_mb=mb(torch.cuda.max_memory_allocated()))
    import resource
    log(backend="process", op="max_rss", rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        total_seconds=round(time.perf_counter() - t0, 1))


# ---------------------------------------------------------------------------
# Report: JSONL -> Markdown tables
# ---------------------------------------------------------------------------

def report(paths):
    recs = []
    for p in paths:
        recs += [json.loads(line) for line in open(p) if line.strip()]
    labels = list(dict.fromkeys(r["label"] for r in recs))
    out = []

    def fmt_s(r):
        if r is None:
            return ""
        if "per_query_ms" in r:
            return f"{r['per_query_ms']:.2f} ms"
        s = r.get("seconds")
        if s is None:
            return ""
        txt = f"{s:.3g} s" if s < 600 else f"{s / 60:.3g} min" if s < 36000 else f"{s / 3600:.3g} h"
        return txt + (" (est)" if r.get("extrapolated") else "")

    for lab in labels:
        rs = [r for r in recs if r["label"] == lab]
        d = next(r for r in rs if r["op"] == "load_or_generate")
        out.append(f"\n### {lab}: {d['n_books']:,} books, vocab {d['vocab']:,}, nnz {d['nnz']:,} "
                   f"(mean {d['mean_distinct']:,.0f} distinct words/book), CSR {d['csr_mb']:,} MB\n")
        out.append("PMI transform (Shifted PPMI, alpha=0.85):\n")
        out.append("| backend | time |\n|---|---|")
        for r in rs:
            if r["op"] == "ppmi_transform":
                out.append(f"| {r['backend']} | {fmt_s(r)} |")
        for m in MEASURES:
            mr = [r for r in rs if r.get("measure") == m]
            if not mr:
                continue
            backends = list(dict.fromkeys(r["backend"] for r in mr if r["op"] != "weighting"))
            out.append(f"\n**{m}**\n")
            out.append("| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |")
            out.append("|---|---|---|---|---|---|---|")
            for b in backends:
                br = {r["op"]: r for r in mr if r["backend"] == b}
                if "skip" in br:
                    out.append(f"| {b} | skipped: {br['skip'].get('note', '')} | | | | | |")
                    continue
                one = br.get("one_vs_many") or br.get("hybrid_query")
                rec = next((br[o]["recall_at_10"] for o in ("recall", "hybrid_query", "all_vs_all_topk")
                            if o in br and "recall_at_10" in br[o]), None)
                build = br.get("build")
                if build is None and b.startswith(("rp", "svd")):
                    src = "hellinger" if m == "jsd" else m
                    build = next((r for r in rs if r["backend"] == b and r["op"] == "build" and r.get("measure") == src), None)
                repr_ = (build or {}).get("repr_mb") or (one or {}).get("repr_mb") or ""
                peak = max([r.get("gpu_peak_mb", 0) for r in br.values()] or [0]) or ""
                extra = ""
                if "hybrid_query" in br:
                    h = br["hybrid_query"]
                    extra = f" (cand {h['cand_ms']:.2f} + rerank {h['rerank_ms']:.2f}; explain top-10 {h['explain_top10_ms']:.1f} ms)"
                out.append(f"| {b} | {fmt_s(build)} | {fmt_s(one)}{extra} | {fmt_s(br.get('all_vs_all_topk'))} | "
                           f"{'' if rec is None else f'{rec:.3f}'} | {repr_} | {peak} |")
        tail = [r for r in rs if r["op"] in ("max_rss", "total_peak")]
        out.append("\n" + ", ".join(f"{r['backend']} {r['op']}: {r.get('rss_mb', r.get('gpu_peak_mb'))} MB" for r in tail))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "shard", "mat"], default="synthetic")
    ap.add_argument("--n-books", type=int, default=1000)
    ap.add_argument("--vocab", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--shard-npz", default="data/shard0_counts.npz")
    ap.add_argument("--mat-bundle", default=str(Path(__file__).resolve().parent / "mat_bundle.json"))
    ap.add_argument("--measures", default=",".join(MEASURES))
    ap.add_argument("--eval-queries", type=int, default=100)
    ap.add_argument("--pf-books", type=int, default=20_000, help="books in the PartialFunc one-vs-many loop")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per measured loop before extrapolating")
    ap.add_argument("--gpu-block", type=int, default=512)
    ap.add_argument("--dense-k", type=int, default=50_000)
    ap.add_argument("--max-gpu-bytes", type=float, default=30e9)
    ap.add_argument("--max-numpy-nnz", type=float, default=3e8)
    ap.add_argument("--proj-dims", type=lambda s: [int(x) for x in s.split(",")], default=[256, 1024])
    ap.add_argument("--hybrid-proj", type=lambda s: [(m[1], int(m[2])) for m in
                                                     (re.fullmatch(r"(rp|svd)(\d+)", x) for x in s.split(","))],
                    default=[("rp", 256), ("svd", 256)])
    ap.add_argument("--hybrid-c", type=lambda s: [int(x) for x in s.split(",")], default=[50, 200])
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="+", default=None, help="render JSONL result files as Markdown")
    args = ap.parse_args()
    if args.report:
        print(report(args.report))
        return
    run(args)


if __name__ == "__main__":
    main()
