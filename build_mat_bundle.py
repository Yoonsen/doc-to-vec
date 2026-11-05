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


def fetch_stratified_cookbooks():
    """Henter et bredt og historisk spennende utvalg kokebøker (DDK 641*) fra 1840 til 2022."""
    eras = [
        (1840, 1929, 75),
        (1930, 1969, 75),
        (1970, 1994, 85),
        (1995, 2009, 85),
        (2010, 2022, 90),
    ]
    seen_urns = set()
    books_raw = []
    for y0, y1, lim in eras:
        resp = requests.post(
            f"{BASE_URL}/build_corpus",
            json={
                "doctype": "digibok",
                "ddk": "641*",
                "from_year": y0,
                "to_year": y1,
                "limit": lim,
            },
            timeout=30,
        )
        resp.raise_for_status()
        c = resp.json()
        urn_map = c.get("urn", {})
        title_map = c.get("title", {})
        auth_map = c.get("authors", {})
        year_map = c.get("year", {})
        for k, u in urn_map.items():
            if u in seen_urns:
                continue
            seen_urns.add(u)
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
    books_raw.sort(key=lambda x: (x["year"], x["urn"]))
    print(f"Hentet {len(books_raw)} unike kokebøker (DDK 641*) fra 1840–2022.")
    return books_raw


def fetch_one_cookbook_chunks(idx: int, book: dict, chunk_size: int = 250, cutoff: int = 4):
    urn = book["urn"]
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
                return idx, 0, 0, []
            time.sleep(1.0 * (attempt + 1))

    if not isinstance(chunks, list) or len(chunks) == 0:
        return idx, 0, 0, []

    K = len(chunks)
    if K < 5:
        return idx, 0, 0, []

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
    books_raw = fetch_stratified_cookbooks()
    totals = pd.read_parquet(DATA_DIR / "totals_500k.parquet").dropna(subset=["word", "total_freq"]).copy()
    totals["word"] = totals["word"].astype(str).str.lower()
    totals_lc = totals.groupby("word", as_index=False)["total_freq"].sum()
    totals_map = dict(zip(totals_lc["word"], totals_lc["total_freq"]))
    total_bg_sum = float(totals_lc["total_freq"].sum())

    all_rows = []
    book_meta_out = {}
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {
            ex.submit(fetch_one_cookbook_chunks, i, b, 250, 4): (i, b)
            for i, b in enumerate(books_raw)
        }
        for fut in as_completed(futs):
            i, K, doc_tokens, rows = fut.result()
            done += 1
            if K >= 5 and doc_tokens > 0 and len(rows) > 0:
                book_meta_out[i] = (K, doc_tokens)
                all_rows.extend(rows)
            if done % 80 == 0 or done == len(books_raw):
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

    # Stoppord: funksjonsord med høy spredning + vanlige oppskrifts-måleord/paratekst
    auto_func_words = set(
        df_burst.groupby("word")
        .filter(lambda g: len(g) >= 25 and (g["disp"].median() >= 0.22 or g["df_ratio"].median() >= 0.89))["word"]
        .unique()
    )
    paratext_and_pronouns = {
        "jeg", "mig", "meg", "du", "dig", "deg", "han", "ham", "hun", "ho", "henne", "vi", "oss", "dere", "dem", "dei",
        "sin", "sine", "sitt", "seg", "sig", "av", "af", "på", "paa", "til", "till", "og", "och", "ikke", "ikkje",
        "var", "har", "hadde", "havde", "ble", "blev", "skal", "skulle", "kan", "kunne", "vil", "ville", "må", "måtte",
        "side", "sider", "kap", "bind", "utg", "isbn", "foto", "forlag", "cappelen", "gyldendal", "aschehoug",
        "the", "and", "with", "for", "from",
    }
    # Behold sentrale mat-basisord i tabellen selv om de er høyfrekvente
    culinary_keep = {
        "salt", "pepper", "smør", "sukker", "vann", "mel", "hvetemel", "egg", "melk", "fløte", "olje", "olivenolje",
        "løk", "hvitløk", "poteter", "potet", "tomater", "tomat", "ost", "fisk", "torsk", "laks", "sild", "lutefisk",
        "klippfisk", "reker", "krabbe", "blåskjell", "kjøtt", "oksekjøtt", "svinekjøtt", "lam", "lammekjøtt", "fårikål",
        "pinnekjøtt", "ribbe", "elg", "hjort", "rein", "rype", "kylling", "kalkun", "pølser", "kjøttkaker", "kraft",
        "buljong", "suppe", "saus", "brød", "deig", "deigen", "gjær", "surdeig", "kake", "kaker", "sjokolade", "krem",
        "mandler", "marsipan", "is", "bær", "jordbær", "epler", "sitron", "ingefær", "chili", "karri", "koriander",
        "basilikum", "timian", "persille", "kanel", "kardemomme", "vanilje", "soyasaus", "ris", "pasta", "spaghetti",
        "pizza", "vin", "rødvin", "hvitvin", "øl", "malt", "humle", "akevitt", "kaffe", "ovn", "ovnen", "stekeovn",
    }
    stop_words = (auto_func_words | paratext_and_pronouns) - culinary_keep

    df_valid = df_burst[df_burst["word"].isin(totals_map) | df_burst["word"].isin(culinary_keep)].copy()
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
    w_agg = w_agg[(w_agg["n_books"] >= 4) | w_agg["word"].isin(culinary_keep)].copy()
    w_agg["priority"] = np.log2(1 + w_agg["corpus_freq"]) * np.maximum(0.2, w_agg["mean_fano"])

    top_words_df = pd.concat(
        [
            w_agg[w_agg["word"].isin(culinary_keep)],
            w_agg[~w_agg["word"].isin(culinary_keep)].sort_values("priority", ascending=False).head(5500),
        ]
    ).drop_duplicates(subset=["word"]).reset_index(drop=True)

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
            "corpus_name": "Norske kokebøker & matkultur (DDK 641, 1840–2022)",
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

    out_path = BASE_DIR / "mat_bundle.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Skrev {out_path} ({size_mb:.2f} MB): {n_books} bøker, {len(words_meta):,} ord, {len(col_idx):,} ikke-null celler.")


if __name__ == "__main__":
    main()
