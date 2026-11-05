import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.nb.no/dhlab"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_one_urn_chunks(idx: int, urn: str, chunk_size: int = 300, cutoff: int = 5):
    """Henter chunks for én URN og beregner både per-bok burstiness-statistikk og chunk-postings."""
    session = requests.Session()
    for attempt in range(3):
        try:
            resp = session.get(
                f"{BASE_URL}/chunks",
                params={"urn": urn, "chunk_size": chunk_size},
                timeout=90,
            )
            resp.raise_for_status()
            chunks = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                return idx, urn, 0, [], []
            time.sleep(1.5 * (attempt + 1))

    if not isinstance(chunks, list) or len(chunks) == 0:
        return idx, urn, 0, [], []

    K = len(chunks)
    # word -> list of (chunk_idx, count)
    word_postings = defaultdict(list)
    for k, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            continue
        lc_counts = defaultdict(int)
        for w, c in ch.items():
            wl = str(w).lower()
            lc_counts[wl] += int(c)
        for wl, c in lc_counts.items():
            if c > 0:
                word_postings[wl].append((k, c))

    burst_rows = []
    posting_rows = []

    log2_K = np.log2(K) if K > 1 else 0.0

    for wl, k_counts in word_postings.items():
        tot = sum(c for _, c in k_counts)
        if tot < cutoff:
            continue

        arr = np.array([c for _, c in k_counts], dtype=np.float64)
        df = len(arr)
        disp = df / K
        exp_df = K * (1.0 - np.exp(-tot / K))
        df_ratio = df / exp_df if exp_df > 0 else 1.0

        p = arr / tot
        entropy = float(-np.sum(p * np.log2(p)))
        internal_pmi = float(max(0.0, log2_K - entropy))

        mu = tot / K
        var = float((np.sum(arr * arr) / K) - (mu * mu))
        fano = float(var / mu) if mu > 0 else 1.0
        burst_tf_df = float(tot / df)

        burst_rows.append((
            idx,
            urn,
            wl,
            int(tot),
            int(df),
            int(K),
            float(disp),
            float(exp_df),
            float(df_ratio),
            entropy,
            internal_pmi,
            fano,
            burst_tf_df,
        ))

        for k, c in k_counts:
            posting_rows.append((idx, k, wl, c))

    return idx, urn, K, burst_rows, posting_rows


def main():
    burst_path = DATA_DIR / "book_word_burstiness.parquet"
    postings_path = DATA_DIR / "chunk_postings_f5.parquet"

    if burst_path.exists() and postings_path.exists():
        print(f"Ferdige filer finnes allerede: {burst_path} og {postings_path}")
        return

    corpus = pd.read_csv(BASE_DIR / "vann.csv")
    urns = corpus["urn"].dropna().unique().tolist()
    print(f"Henter 300-ords chunks for {len(urns)} bøker i vann.csv...")

    t0 = time.time()
    all_burst = []
    all_postings = []
    total_chunks = 0
    done = 0

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(fetch_one_urn_chunks, idx, urn, 300, 5): (idx, urn)
            for idx, urn in enumerate(urns)
        }
        for fut in as_completed(futures):
            idx, urn, K, b_rows, p_rows = fut.result()
            total_chunks += K
            all_burst.extend(b_rows)
            all_postings.extend(p_rows)
            done += 1
            if done % 50 == 0 or done == len(urns):
                elapsed = time.time() - t0
                print(
                    f"  Ferdig med {done}/{len(urns)} bøker ({total_chunks:,} chunks så langt) "
                    f"på {elapsed:.1f}s..."
                )

    print("Bygger DataFrames og lagrer til Parquet...")
    df_burst = pd.DataFrame(
        all_burst,
        columns=[
            "doc_idx",
            "urn",
            "word",
            "freq",
            "df_chunks",
            "n_chunks",
            "disp",
            "exp_df",
            "df_ratio",
            "entropy",
            "internal_pmi",
            "fano",
            "burst_tf_df",
        ],
    )
    df_burst.to_parquet(burst_path, index=False)

    df_post = pd.DataFrame(
        all_postings,
        columns=["doc_idx", "chunk_idx", "word", "count"],
    )
    df_post["doc_idx"] = df_post["doc_idx"].astype(np.int16)
    df_post["chunk_idx"] = df_post["chunk_idx"].astype(np.int32)
    df_post["count"] = df_post["count"].astype(np.int16)
    df_post["word"] = df_post["word"].astype("category")
    df_post.to_parquet(postings_path, index=False)

    print(
        f"Ferdig på {time.time() - t0:.1f}s!\n"
        f"  Totalt chunks: {total_chunks:,}\n"
        f"  Burstiness-tabell: {len(df_burst):,} rader -> {burst_path}\n"
        f"  Chunk-postings:    {len(df_post):,} rader -> {postings_path}"
    )


if __name__ == "__main__":
    main()
