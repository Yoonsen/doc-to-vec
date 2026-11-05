import json
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

BBI_DIR = Path("/Users/larsj/Github/bbi_diversity")


def load_bbi_luminon_and_targets():
    r1 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_annotations.jsonl", encoding="utf-8")]
    r2 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_round2_annotations.jsonl", encoding="utf-8")]
    for r in r1:
        r["domain"] = (
            "funksjon"
            if r["subgroup"].startswith(("syn_", "horsel_", "bevegelse_", "kognitiv_"))
            else "religion"
        )
    all_recs = r1 + r2

    target_words = set()
    urns = set()
    for r in all_recs:
        if r.get("urn"):
            urns.add(r["urn"].strip())
        for k in ("matched_word", "target_token"):
            if r.get(k):
                target_words.add(str(r[k]).strip().lower())

    basic_Path = BBI_DIR / "basic_categories.json"
    if basic_Path.exists():
        bc = json.loads(basic_Path.read_text(encoding="utf-8"))
        for _, wlist in bc.items():
            for w in wlist:
                target_words.add(str(w).strip().lower())

    return all_recs, sorted(urns), target_words


def fetch_one_urn_chunks(
    idx: int,
    urn: str,
    target_words: set[str],
    chunk_size: int = 250,
    general_cutoff: int = 3,
):
    session = requests.Session()
    for attempt in range(3):
        try:
            resp = session.get(
                f"{BASE_URL}/chunks",
                params={"urn": urn, "chunk_size": chunk_size},
                timeout=60,
            )
            resp.raise_for_status()
            chunks = resp.json()
            break
        except Exception:
            if attempt == 2:
                return idx, urn, 0, [], []
            time.sleep(1.0 * (attempt + 1))

    if not isinstance(chunks, list) or len(chunks) == 0:
        return idx, urn, 0, [], []

    K = len(chunks)
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
        is_target = wl in target_words
        if tot < general_cutoff and not is_target:
            continue

        arr = np.array([c for _, c in k_counts], dtype=np.float64)
        df_c = len(arr)
        disp = df_c / K

        p = arr / tot
        H = -np.sum(p * np.log2(p))
        pmi_intern = max(0.0, log2_K - H)

        mu = tot / K
        sum_sq = np.sum(arr * arr)
        var = (sum_sq / K) - (mu * mu)
        fano = (var / mu) if mu > 0 else 1.0

        exp_df = K * (1.0 - np.exp(-mu))
        df_over_exp = (df_c / exp_df) if exp_df > 0 else 1.0

        burst_rows.append(
            (
                idx,
                wl,
                int(tot),
                int(df_c),
                int(K),
                np.float32(disp),
                np.float32(pmi_intern),
                np.float32(fano),
                np.float32(df_over_exp),
            )
        )

        if tot >= general_cutoff and wl.isalpha() and len(wl) >= 3:
            for k, c in k_counts:
                posting_rows.append((idx, k, wl, int(c)))

    return idx, urn, K, burst_rows, posting_rows


def main():
    all_recs, urns, target_words = load_bbi_luminon_and_targets()
    print(f"Loaded {len(all_recs)} Luminon annotations across {len(urns)} unique BBI URNs.")
    print(f"Tracking {len(target_words)} target/dictionary word forms down to TF>=1 (general cutoff TF>=3).")

    all_burst = []
    all_postings = []
    urn_chunks = {}

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {
            ex.submit(fetch_one_urn_chunks, idx, u, target_words, 250, 3): (idx, u)
            for idx, u in enumerate(urns)
        }
        for fut in as_completed(futs):
            idx, u, K, b_rows, p_rows = fut.result()
            urn_chunks[u] = K
            all_burst.extend(b_rows)
            all_postings.extend(p_rows)
            done += 1
            if done % 200 == 0 or done == len(urns):
                elapsed = time.time() - t0
                print(
                    f"  [{done}/{len(urns)}] elapsed={elapsed:.1f}s | burst_rows={len(all_burst):,} | postings={len(all_postings):,}"
                )

    burst_df = pd.DataFrame(
        all_burst,
        columns=[
            "doc_idx",
            "word",
            "tf",
            "chunk_df",
            "num_chunks",
            "chunk_disp",
            "pmi_intern",
            "fano",
            "df_over_exp",
        ],
    )
    urn_map = pd.Series(urns, name="urn")
    burst_df["urn"] = burst_df["doc_idx"].map(urn_map)

    out_burst = DATA_DIR / "bbi_book_word_burstiness.parquet"
    burst_df.to_parquet(out_burst, index=False)
    print(f"Saved {out_burst} ({len(burst_df):,} rows)")

    post_df = pd.DataFrame(
        all_postings, columns=["doc_idx", "chunk_idx", "word", "count"]
    )
    out_post = DATA_DIR / "bbi_chunk_postings_f3.parquet"
    post_df.to_parquet(out_post, index=False)
    print(f"Saved {out_post} ({len(post_df):,} rows)")


if __name__ == "__main__":
    main()
