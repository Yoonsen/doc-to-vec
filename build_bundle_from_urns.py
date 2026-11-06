#!/usr/bin/env python3
"""
Bygger en doc-to-vec korpuspakke (.json) direkte fra en fil eller liste med URN-er.

Eksempel på bruk:
    python build_bundle_from_urns.py min_urn_liste.csv --name "Bachelor-korpus: Anbefalinger" --out anbefalinger_bundle.json
"""
import argparse
import json
import re
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

URN_REGEX = re.compile(r"URN:NBN:no-nb_[a-zA-Z0-9_]+")
DIGIBOK_NUM_REGEX = re.compile(r"\bdigibok_[0-9]{10,16}\b")


def extract_urns_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    seen = set()
    urns = []
    for m in URN_REGEX.findall(text):
        if m not in seen:
            seen.add(m)
            urns.append(m)
    for m in DIGIBOK_NUM_REGEX.findall(text):
        full = f"URN:NBN:no-nb_{m}"
        if full not in seen:
            seen.add(full)
            urns.append(full)
    return urns


def fetch_metadata_for_urns(urns: list[str]) -> list[dict]:
    books_raw = []
    seen = set()
    batch_size = 500
    for start in range(0, len(urns), batch_size):
        batch = urns[start : start + batch_size]
        resp = requests.post(
            f"{BASE_URL}/get_metadata",
            json={"urns": batch},
            timeout=45,
        )
        resp.raise_for_status()
        c = resp.json()
        urn_map = c.get("urn", {})
        title_map = c.get("title", {})
        auth_map = c.get("authors", {})
        year_map = c.get("year", {})
        for k, u in urn_map.items():
            if u in seen:
                continue
            seen.add(u)
            try:
                yr = int(float(year_map.get(k, 0)))
            except Exception:
                yr = 0
            books_raw.append(
                {
                    "urn": u,
                    "title": str(title_map.get(k, "Uten tittel")),
                    "authors": str(auth_map.get(k, "")),
                    "year": yr,
                }
            )
    # Fallback for URN-er som eventuelt ikke returnerte metadata
    for u in urns:
        if u not in seen:
            seen.add(u)
            books_raw.append({"urn": u, "title": u, "authors": "", "year": 0})
    return books_raw


def fetch_one_book_chunks(idx: int, book: dict, chunk_size: int = 250, cutoff: int = 3):
    urn = book["urn"]
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{BASE_URL}/chunks",
                params={"urn": urn, "chunk_size": chunk_size},
                timeout=60,
            )
            resp.raise_for_status()
            chunks = resp.json()
            break
        except Exception:
            if attempt == 2:
                return idx, 0, 0, []
            time.sleep(1.0 * (attempt + 1))

    if not isinstance(chunks, list) or len(chunks) < 5:
        return idx, 0, 0, []

    K = len(chunks)
    word_postings = defaultdict(list)
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        lc_counts = defaultdict(int)
        for w, c in ch.items():
            wl = str(w).lower()
            lc_counts[wl] += int(c)
        for wl, c in lc_counts.items():
            if c > 0:
                word_postings[wl].append(c)

    rows = []
    doc_tokens = 0
    for wl, counts in word_postings.items():
        tot = sum(counts)
        if tot < cutoff or not wl.isalpha() or len(wl) < 3:
            continue
        doc_tokens += tot
        arr = np.array(counts, dtype=np.float64)
        df_c = len(arr)
        disp = df_c / K

        mu = tot / K
        sum_sq = np.sum(arr * arr)
        var = (sum_sq / K) - (mu * mu)
        fano = (var / mu) if mu > 0 else 1.0

        exp_df = K * (1.0 - np.exp(-mu))
        df_over_exp = (df_c / exp_df) if exp_df > 0 else 1.0

        rows.append((idx, wl, int(tot), float(disp), float(fano), float(df_over_exp)))

    return idx, K, doc_tokens, rows


def main():
    parser = argparse.ArgumentParser(description="Bygg doc-to-vec korpuspakke (.json) fra en fil med URN-er.")
    parser.add_argument("input_file", type=Path, help="Sti til .csv, .txt eller .json som inneholder URN-er")
    parser.add_argument("--name", type=str, default="Tilpasset URN-korpus", help="Visningsnavn på korpuset")
    parser.add_argument("--out", type=Path, default=Path("custom_urn_bundle.json"), help="Utfil (.json)")
    parser.add_argument("--chunk-size", type=int, default=250, help="Antall ord per chunk (default: 250)")
    parser.add_argument("--max-words", type=int, default=5500, help="Maks antall ord i vokabularet (default: 5500)")
    parser.add_argument("--workers", type=int, default=12, help="Antall parallelle tråder mot NB API")
    args = parser.parse_args()

    urns = extract_urns_from_file(args.input_file)
    if not urns:
        raise SystemExit(f"Fant ingen URN-er i {args.input_file}")

    print(f"Fant {len(urns)} unike URN-er i {args.input_file}. Henter metadata fra NB DHlab...")
    books_raw = fetch_metadata_for_urns(urns)
    print(f"Hentet metadata for {len(books_raw)} bøker. Henter {args.chunk_size}-ords tekstblokker...")

    totals = pd.read_parquet(DATA_DIR / "totals_500k.parquet").dropna(subset=["word", "total_freq"]).copy()
    totals["word"] = totals["word"].astype(str).str.lower()
    totals_lc = totals.groupby("word", as_index=False)["total_freq"].sum()
    totals_map = dict(zip(totals_lc["word"], totals_lc["total_freq"]))
    total_bg_sum = float(totals_lc["total_freq"].sum())

    all_rows = []
    book_meta_out = {}
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_one_book_chunks, i, b, args.chunk_size, 3): (i, b)
            for i, b in enumerate(books_raw)
        }
        for fut in as_completed(futs):
            i, K, doc_tokens, rows = fut.result()
            done += 1
            if K >= 5 and doc_tokens > 0 and len(rows) > 0:
                book_meta_out[i] = (K, doc_tokens)
                all_rows.extend(rows)
            if done % 50 == 0 or done == len(books_raw):
                print(
                    f"  [{done}/{len(books_raw)}] elapsed={time.time()-t0:.1f}s | aktive bøker={len(book_meta_out)} | rader={len(all_rows):,}"
                )

    df_burst = pd.DataFrame(
        all_rows, columns=["doc_idx", "word", "freq", "disp", "fano", "df_ratio"]
    )
    active_indices = sorted(book_meta_out.keys())
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(active_indices)}

    books = []
    for new_i, old_i in enumerate(active_indices):
        b = books_raw[old_i]
        K, d_tok = book_meta_out[old_i]
        books.append(
            {
                "id": new_i,
                "urn": b["urn"],
                "title": b["title"],
                "authors": b["authors"],
                "year": b["year"],
                "n_chunks": int(K),
                "doc_tokens": int(d_tok),
            }
        )

    min_df_stop = max(5, int(len(books) * 0.12))
    auto_func_words = set(
        df_burst.groupby("word")
        .filter(lambda g: len(g) >= min_df_stop and (g["disp"].median() >= 0.22 or g["df_ratio"].median() >= 0.89))["word"]
        .unique()
    )
    paratext_and_pronouns = {
        "jeg", "mig", "meg", "du", "dig", "deg", "han", "ham", "hun", "ho", "henne", "vi", "oss", "dere", "dem", "dei",
        "sin", "sine", "sitt", "seg", "sig", "av", "af", "på", "paa", "til", "till", "og", "och", "ikke", "ikkje",
        "var", "har", "hadde", "havde", "ble", "blev", "skal", "skulle", "kan", "kunne", "vil", "ville", "må", "måtte",
        "side", "sider", "kap", "bind", "utg", "isbn", "foto", "forlag", "cappelen", "gyldendal", "aschehoug",
        "the", "and", "with", "for", "from",
    }
    stop_words = auto_func_words | paratext_and_pronouns

    df_valid = df_burst[df_burst["word"].isin(totals_map)].copy()
    w_agg = (
        df_valid.groupby("word")
        .agg(
            n_books=("doc_idx", "nunique"),
            corpus_freq=("freq", "sum"),
            median_disp=("disp", "median"),
            median_df_ratio=("df_ratio", "median"),
            median_fano=("fano", "median"),
            mean_fano=("fano", "mean"),
        )
        .reset_index()
    )
    min_books_vocab = 2 if len(books) < 60 else 3
    w_agg = w_agg[w_agg["n_books"] >= min_books_vocab].copy()
    w_agg["priority"] = np.log2(1 + w_agg["corpus_freq"]) * np.maximum(0.2, w_agg["mean_fano"])

    top_words_df = w_agg.sort_values("priority", ascending=False).head(args.max_words).reset_index(drop=True)
    vocab_list = top_words_df["word"].tolist()
    word_to_id = {w: i for i, w in enumerate(vocab_list)}

    words_meta = []
    for _, r in top_words_df.iterrows():
        w = r["word"]
        words_meta.append(
            {
                "w": w,
                "tf_bg": int(totals_map.get(w, 200)),
                "nb": int(r["n_books"]),
                "cf": int(r["corpus_freq"]),
                "md": round(float(r["median_disp"]), 4),
                "mr": round(float(r["median_df_ratio"]), 3),
                "mf": round(float(r["median_fano"]), 3),
                "stop": 1 if w in stop_words else 0,
            }
        )

    sub = df_valid[df_valid["word"].isin(word_to_id)].copy()
    sub["d_id"] = sub["doc_idx"].map(old_to_new)
    sub["w_id"] = sub["word"].map(word_to_id)
    sub = sub.sort_values(["d_id", "w_id"])

    n_books = len(books)
    counts_per_doc = sub.groupby("d_id").size().reindex(range(n_books), fill_value=0).values
    row_ptr = [0] + np.cumsum(counts_per_doc).astype(int).tolist()

    col_idx = sub["w_id"].astype(int).tolist()
    freqs = sub["freq"].astype(int).tolist()
    fano_x100 = np.round(sub["fano"].values * 100).astype(int).tolist()
    dfratio_x100 = np.round(sub["df_ratio"].values * 100).astype(int).tolist()

    bundle = {
        "meta": {
            "corpus_name": args.name,
            "n_books": n_books,
            "n_words": len(words_meta),
            "nnz": len(col_idx),
            "total_bg_sum": total_bg_sum,
        },
        "books": books,
        "words": words_meta,
        "csr": {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "freq": freqs,
            "fano_x100": fano_x100,
            "dfratio_x100": dfratio_x100,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Skrev {args.out} ({size_mb:.2f} MB): {n_books} bøker, {len(words_meta):,} ord, {len(col_idx):,} ikke-null celler.")


if __name__ == "__main__":
    main()
