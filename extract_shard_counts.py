"""Extract per-book word counts (book_id, cf_id, tf) from a Roaring shard into a CSR .npz.

Read-only on the shard. Only integer ids and counts are written: no word strings, no text.
Word ids are renumbered 0..M-1 by descending total frequency in the shard, so the
vocabulary order matches the frequency-ranked convention used by partial-functions
(rank 0 = most frequent word).

    python extract_shard_counts.py SHARD.db OUT.npz
"""
import sqlite3
import sys
import time

import numpy as np
import scipy.sparse as sp


def main(db_path: str, out_path: str) -> None:
    t0 = time.time()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    n = con.execute("SELECT COUNT(*) FROM unigrams").fetchone()[0]
    print(f"rows: {n:,}", flush=True)
    books = np.empty(n, dtype=np.int64)
    words = np.empty(n, dtype=np.int64)
    tfs = np.empty(n, dtype=np.float32)
    cur = con.execute("SELECT book_id, cf_id, tf FROM unigrams")
    i = 0
    while True:
        rows = cur.fetchmany(1_000_000)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.int64)
        m = len(a)
        books[i:i + m], words[i:i + m], tfs[i:i + m] = a[:, 0], a[:, 1], a[:, 2]
        i += m
        print(f"  {i:,} rows  {time.time() - t0:.0f}s", flush=True)
    books, words, tfs = books[:i], words[:i], tfs[:i]

    book_ids, b_idx = np.unique(books, return_inverse=True)
    cf_ids, w_idx = np.unique(words, return_inverse=True)
    totals = np.bincount(w_idx, weights=tfs)
    order = np.argsort(-totals, kind="stable")  # order[r] = old index with rank r
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))
    X = sp.csr_matrix((tfs, (b_idx, rank_of[w_idx])), shape=(len(book_ids), len(cf_ids)))
    X.sum_duplicates()
    X.sort_indices()
    np.savez(out_path, indptr=X.indptr.astype(np.int64), indices=X.indices.astype(np.int32),
             data=X.data.astype(np.float32), shape=np.array(X.shape), book_ids=book_ids,
             cf_id_by_rank=cf_ids[order])
    print(f"books={X.shape[0]:,} vocab={X.shape[1]:,} nnz={X.nnz:,} "
          f"mean distinct/book={X.nnz / X.shape[0]:.0f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
