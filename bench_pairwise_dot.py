"""Benchmark: pairwise dot products <f, g> as building blocks for A @ A.T.

Compares every way partial-functions computes <f, g> via I = dom(f) & dom(g)
against matrix multiplication on CPU and GPU, on the same data as
bench_similarity_gpu.py (values = L2-normalised counts, so <f, g> = cosine).

Per-pair (Python loop, one call per pair):
  pf_dot                 PartialFunc.dot (Roaring & + gallop extraction + np.dot)
  align_gallop           align_intersection(method="gallop") + np.dot
  align_twopointer       align_intersection(method="twopointer") + np.dot
  align_rank             align_intersection(method="rank"): dom.rank(w) - 1 for each w in I
  bitblock               _bitblock_intersect_dot_and_jsd on cached 64-bit blocks (POPCNT indexing)
  ws_dot / ws_dot_jsd    SparseWorkspace: scatter(f) once, dot_with_active(g) / dot_and_jsd_with_active(g)

Batched in Numba (one call for a block of query rows x all targets, prange over targets):
  nb_merge               two-pointer merge of CSR rows (f[I] * g[I] without Python per pair)
  nb_bitblock            64-bit block masks + POPCNT, concatenated per-book blocks
  nb_workspace           Gustavson: scatter query into a flat array, gather over each target row

Matrix multiplication:
  scipy_spgemm           scipy CSR A[block] @ A.T (single-threaded)
  torch_spmm             torch CSR (cuSPARSE) A @ Q_block (dense query block)
  torch_dense_fp32/fp16  dense GEMM over the full used vocabulary (exact) when it fits on the GPU

All methods are checked against a float64 reference on sampled pairs.
All-vs-all = N^2 dot evaluations (pairwise methods could halve this by symmetry).

    python bench_pairwise_dot.py --source synthetic --n-books 10000 --out pairwise.jsonl
    python bench_pairwise_dot.py --report pairwise.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from numba import njit, prange

import bench_similarity_gpu as bs
from partial_func import PartialFunc, SparseWorkspace, _bitblock_intersect_dot_and_jsd, _build_bitblocks


@njit(parallel=True, cache=True)
def k_merge_block(indptr, indices, data, qrows, targets, out):
    for t in prange(len(targets)):
        j = targets[t]
        for a in range(len(qrows)):
            i = qrows[a]
            p, pe = indptr[i], indptr[i + 1]
            q, qe = indptr[j], indptr[j + 1]
            s = 0.0
            while p < pe and q < qe:
                wi = indices[p]
                wj = indices[q]
                if wi == wj:
                    s += np.float64(data[p]) * data[q]
                    p += 1
                    q += 1
                elif wi < wj:
                    p += 1
                else:
                    q += 1
            out[a, t] = s


@njit(cache=True)
def _popcnt(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(parallel=True, cache=True)
def k_bitblock_block(bptr, blk, msk, off, vptr, val, qrows, targets, out):
    for t in prange(len(targets)):
        j = targets[t]
        for a in range(len(qrows)):
            i = qrows[a]
            p, pe = bptr[i], bptr[i + 1]
            q, qe = bptr[j], bptr[j + 1]
            vi, vj = vptr[i], vptr[j]
            s = 0.0
            while p < pe and q < qe:
                if blk[p] == blk[q]:
                    mf = msk[p]
                    mg = msk[q]
                    m = mf & mg
                    while m != np.uint64(0):
                        low = m & (np.uint64(0) - m)
                        before = low - np.uint64(1)
                        s += np.float64(val[vi + off[p] + _popcnt(mf & before)]) * val[vj + off[q] + _popcnt(mg & before)]
                        m ^= low
                    p += 1
                    q += 1
                elif blk[p] < blk[q]:
                    p += 1
                else:
                    q += 1
            out[a, t] = s


@njit(parallel=True, cache=True)
def k_workspace_block(indptr, indices, data, work, targets, out_row):
    for t in prange(len(targets)):
        j = targets[t]
        s = 0.0
        for q in range(indptr[j], indptr[j + 1]):
            s += data[q] * work[indices[q]]
        out_row[t] = s


def build_all_bitblocks(indptr, indices, rows):
    parts, bptr = [], [0]
    for i in rows:
        b, m, o = _build_bitblocks(indices[indptr[i]:indptr[i + 1]].view(np.uint32))
        parts.append((b, m, o))
        bptr.append(bptr[-1] + len(b))
    blk = np.concatenate([p[0] for p in parts])
    msk = np.concatenate([p[1] for p in parts])
    off = np.concatenate([p[2] for p in parts]).astype(np.int64)
    return np.asarray(bptr, dtype=np.int64), blk, msk, off


def run(args):
    indptr, indices, data, vocab = bs.load_source(args)
    n = len(indptr) - 1
    nnz = int(indptr[-1])
    data = bs.k_row_scale(indptr, data, 0)  # L2-normalised: <f, g> = cosine
    log = bs.Log(args.out, {"source": args.source, "n_books": n, "vocab": vocab, "nnz": nnz,
                            "label": f"{args.source}_{n}"})
    log(backend="data", op="load", mean_distinct=round(nnz / n, 1))
    rng = np.random.default_rng(args.seed)
    n_all = float(n) * n
    budget = args.budget

    # ---- sample of books for per-pair methods ----
    S = min(n, args.sample_books)
    srows = np.sort(rng.choice(n, size=S, replace=False))
    pfs = [bs.pf_from_row(indptr, indices, data, int(i)) for i in srows]
    for f in pfs:
        f.bitblocks  # cache
    pairs = rng.integers(0, S, size=(args.pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    def ref(a, b):  # float64 reference via sorted-array intersection
        fa, fb = pfs[a], pfs[b]
        _, ia, ib = np.intersect1d(fa.dom_array, fb.dom_array, assume_unique=True, return_indices=True)
        return float(np.dot(fa.val[ia], fb.val[ib]))
    refv = np.array([ref(a, b) for a, b in pairs])
    inter = np.array([len(pfs[a].dom & pfs[b].dom) for a, b in pairs[:200]])
    log(backend="data", op="pair_stats", mean_dom=round(float(np.mean([len(f) for f in pfs])), 1),
        mean_intersection=round(float(inter.mean()), 1))

    def per_pair(name, fn, max_pairs=None):
        P = pairs if max_pairs is None else pairs[:max_pairs]
        fn(*P[0])  # warm-up / JIT
        t = time.perf_counter()
        vals, k = [], 0
        for a, b in P:
            vals.append(fn(a, b))
            k += 1
            if time.perf_counter() - t > budget:
                break
        el = time.perf_counter() - t
        err = float(np.max(np.abs(np.array(vals) - refv[:k])))
        log(backend=name, op="pair", per_pair_us=round(el / k * 1e6, 2), pairs_measured=k,
            all_vs_all_s=round(el / k * n_all, 1), extrapolated=True, max_abs_err=err)

    per_pair("pf_dot", lambda a, b: pfs[a].dot(pfs[b]))
    for m in ("gallop", "twopointer", "rank"):
        def fn(a, b, m=m):
            _, vf, vg = pfs[a].align_intersection(pfs[b], method=m)
            return float(np.dot(vf, vg))
        per_pair(f"align_{m}", fn, max_pairs=200 if m == "rank" else None)

    def bb(a, b):
        f, g = pfs[a], pfs[b]
        bf, mf, of = f.bitblocks
        bg_, mg, og = g.bitblocks
        return _bitblock_intersect_dot_and_jsd(bf, mf, of, f.val, bg_, mg, og, g.val)[0]
    per_pair("bitblock", bb)

    # SparseWorkspace: scatter each query once, then many g (pairs grouped by query)
    ws = SparseWorkspace(vocab)
    for name, call in (("ws_dot", ws.dot_with_active),
                       ("ws_dot_jsd", lambda g: ws.dot_and_jsd_with_active(g)[0])):
        qs = rng.choice(S, size=min(S, 20), replace=False)
        ws.scatter(pfs[0]); call(pfs[1])
        t = time.perf_counter()
        k, err = 0, 0.0
        for a in qs:
            ws.scatter(pfs[a])
            for b in range(S):
                v = call(pfs[b])
                k += 1
                if b % 97 == 0 and b != a:
                    err = max(err, abs(v - ref(a, b)))
            if time.perf_counter() - t > budget:
                break
        el = time.perf_counter() - t
        log(backend=name, op="pair", per_pair_us=round(el / k * 1e6, 2), pairs_measured=k,
            all_vs_all_s=round(el / k * n_all, 1), extrapolated=True, max_abs_err=err)
    ws.clear()
    del pfs

    # ---- batched Numba kernels: block of queries x target set ----
    T = min(n, args.targets)
    targets = np.sort(rng.choice(n, size=T, replace=False)).astype(np.int64)
    qrows = targets[: args.qblock]
    exact_blk = np.empty((len(qrows), T))
    k_merge_block(indptr, indices, data, qrows[:1], targets[:10], np.empty((1, 10)))
    t = time.perf_counter()
    k_merge_block(indptr, indices, data, qrows, targets, exact_blk)
    el = time.perf_counter() - t
    ref_blk = exact_blk.copy()
    log(backend="nb_merge", op="batched", per_pair_us=round(el / exact_blk.size * 1e6, 4),
        all_vs_all_s=round(el / exact_blk.size * n_all, 1), extrapolated=True, threads=bs.os.cpu_count(),
        max_abs_err=0.0)

    rows_bb = np.unique(np.concatenate([targets, qrows]))
    pos = {int(r): i for i, r in enumerate(rows_bb)}
    t = time.perf_counter()
    bptr, blk, msk, off = build_all_bitblocks(indptr, indices, rows_bb)
    vptr = np.array([indptr[r] for r in rows_bb], dtype=np.int64)
    t_build = time.perf_counter() - t
    lq = np.array([pos[int(r)] for r in qrows], dtype=np.int64)
    lt = np.array([pos[int(r)] for r in targets], dtype=np.int64)
    out = np.empty((len(qrows), T))
    k_bitblock_block(bptr, blk, msk, off, vptr, data, lq[:1], lt[:10], np.empty((1, 10)))
    t = time.perf_counter()
    k_bitblock_block(bptr, blk, msk, off, vptr, data, lq, lt, out)
    el = time.perf_counter() - t
    log(backend="nb_bitblock", op="batched", per_pair_us=round(el / out.size * 1e6, 4),
        all_vs_all_s=round(el / out.size * n_all, 1), extrapolated=True, build_s_for_targets=round(t_build, 2),
        bitblock_mb=bs.mb(bptr.nbytes + blk.nbytes + msk.nbytes + off.nbytes),
        max_abs_err=float(np.max(np.abs(out - ref_blk))))
    del blk, msk, off

    work = np.zeros(vocab)
    row = np.empty(T)
    k_workspace_block(indptr, indices, data, work, targets[:10], row[:10])
    t = time.perf_counter()
    err = 0.0
    for a, i in enumerate(qrows):
        s, e = indptr[i], indptr[i + 1]
        work[indices[s:e]] = data[s:e]
        k_workspace_block(indptr, indices, data, work, targets, row)
        work[indices[s:e]] = 0.0
        err = max(err, float(np.max(np.abs(row - ref_blk[a]))))
    el = time.perf_counter() - t
    log(backend="nb_workspace", op="batched", per_pair_us=round(el / (len(qrows) * T) * 1e6, 4),
        all_vs_all_s=round(el / (len(qrows) * T) * n_all, 1), extrapolated=True, max_abs_err=err)

    # ---- scipy SpGEMM ----
    if nnz <= args.max_scipy_nnz:
        A = bs.make_scipy(indptr, indices, data, (n, vocab))
        At = A.T.tocsr()
        sub = A[qrows]
        t = time.perf_counter()
        R = (sub @ At).toarray()
        el = time.perf_counter() - t
        log(backend="scipy_spgemm", op="matmul", per_pair_us=round(el / R.size * 1e6, 4),
            all_vs_all_s=round(el / R.size * n_all, 1), extrapolated=True,
            max_abs_err=float(np.max(np.abs(R[:, targets] - ref_blk))))
        del A, At, sub, R

    if args.no_gpu:
        return
    torch = bs.torch_setup()
    dev = torch.device("cuda")
    G = bs.TorchSparse(indptr, indices, data, vocab, dev)
    B = min(args.gpu_block, max(1, int(4e9 / 4 / vocab)))
    blocks = [np.arange(s, min(n, s + B)) for s in range(0, n, B)]

    def spmm(rows):
        Q = torch.zeros(vocab, len(rows), device=dev)
        for j, i in enumerate(rows):
            s, e = int(indptr[i]), int(indptr[i + 1])
            Q[G.col[s:e].long(), j] = G.val[s:e]
        return torch.sparse.mm(G.X, Q)
    spmm(blocks[0][:2])
    bs.gpu_peak_reset()
    t = time.perf_counter()
    done = 0
    for rows in blocks:
        spmm(rows)
        done += len(rows)
        torch.cuda.synchronize()
        if time.perf_counter() - t > budget * 2:
            break
    el = time.perf_counter() - t
    chk = spmm(qrows)[torch.as_tensor(targets, device=dev)].T.double().cpu().numpy()
    log(backend="torch_spmm", op="matmul", per_pair_us=round(el / (done * n) * 1e6, 5),
        all_vs_all_s=round(el / done * n, 2), extrapolated=done < n, gpu_peak_mb=bs.gpu_peak_mb(),
        max_abs_err=float(np.max(np.abs(chk - ref_blk))))
    del G
    torch.cuda.empty_cache()

    used = np.unique(indices) if nnz < 5e8 else np.arange(vocab)
    Vu = len(used)
    remap = np.full(vocab, -1, dtype=np.int64)
    remap[used] = np.arange(Vu)
    for name, dt, sz in (("torch_dense_fp32", torch.float32, 4), ("torch_dense_fp16", torch.float16, 2)):
        need = n * Vu * sz
        if need > args.max_gpu_bytes:
            log(backend=name, op="skip", note=f"dense {n}x{Vu} needs {need / 2**30:.1f} GB")
            continue
        bs.gpu_peak_reset()
        M = torch.zeros(n, Vu, device=dev, dtype=dt)
        for s, e in bs.row_chunks(indptr, 100_000_000):
            a, b = int(indptr[s]), int(indptr[e])
            r = bs.torch_row_ids(indptr, s, e, dev)
            M[r, torch.from_numpy(remap[indices[a:b]]).to(dev)] = torch.from_numpy(data[a:b]).to(dev, dt)
        Bd = 2048
        _ = M[:16] @ M.T
        torch.cuda.synchronize()
        t = time.perf_counter()
        done = 0
        for s in range(0, n, Bd):
            S_ = M[s:s + Bd] @ M.T
            torch.topk(S_.float(), 11, dim=1)
            done += S_.shape[0]
            torch.cuda.synchronize()
            if time.perf_counter() - t > budget * 2:
                break
        el = time.perf_counter() - t
        qt = torch.as_tensor(qrows, device=dev)
        chk = (M[qt] @ M.T).double()[:, torch.as_tensor(targets, device=dev)].cpu().numpy()
        log(backend=name, op="matmul", per_pair_us=round(el / (done * n) * 1e6, 6),
            all_vs_all_s=round(el / done * n, 2), extrapolated=done < n, gpu_peak_mb=bs.gpu_peak_mb(),
            repr_mb=bs.mb(need), note="exact: full used vocabulary, incl. top-10 per row",
            max_abs_err=float(np.max(np.abs(chk - ref_blk))))
        del M, S_
        torch.cuda.empty_cache()


def report(paths):
    recs = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    out = []
    for lab in dict.fromkeys(r["label"] for r in recs):
        rs = [r for r in recs if r["label"] == lab]
        st = next(r for r in rs if r["op"] == "pair_stats")
        out.append(f"\n### {lab}: {rs[0]['n_books']:,} books, mean |dom(f)| = {st['mean_dom']:,.0f}, "
                   f"mean |I| = {st['mean_intersection']:,.0f}\n")
        out.append("| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |")
        out.append("|---|---|---|---|---|")
        for r in rs:
            if r["op"] in ("load", "pair_stats"):
                continue
            if r["op"] == "skip":
                out.append(f"| {r['backend']} | skipped | {r['note']} | | |")
                continue
            us = r["per_pair_us"]
            t = f"{us:.3g} µs" if us >= 0.01 else f"{us * 1000:.3g} ns"
            s = r["all_vs_all_s"]
            tot = f"{s:.3g} s" if s < 600 else f"{s / 60:.3g} min" if s < 36000 else f"{s / 3600:.3g} h" if s < 86400 * 3 else f"{s / 86400:.3g} d"
            tot += " (est)" if r.get("extrapolated") else ""
            out.append(f"| {r['backend']} | {r['op']} | {t} | {tot} | {r.get('max_abs_err', 0):.1e} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "shard", "mat"], default="synthetic")
    ap.add_argument("--n-books", type=int, default=1000)
    ap.add_argument("--vocab", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--shard-npz", default="data/shard0_counts.npz")
    ap.add_argument("--mat-bundle", default="mat_bundle.json")
    ap.add_argument("--sample-books", type=int, default=2000)
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--targets", type=int, default=20_000)
    ap.add_argument("--qblock", type=int, default=64)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--gpu-block", type=int, default=512)
    ap.add_argument("--max-gpu-bytes", type=float, default=30e9)
    ap.add_argument("--max-scipy-nnz", type=float, default=3e8)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="+", default=None)
    args = ap.parse_args()
    if args.report:
        print(report(args.report))
        return
    run(args)


if __name__ == "__main__":
    main()
