import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BBI_DIR = Path("/Users/larsj/Github/bbi_diversity")


def main():
    barn = pd.read_csv(BBI_DIR / "data/barn.csv", low_memory=False)
    meta_by_urn = {}
    for row in barn.itertuples(index=False):
        u = str(getattr(row, "urn", "")).strip()
        if u:
            yr = getattr(row, "year", 0)
            try:
                yr = int(float(yr))
            except Exception:
                yr = 0
            meta_by_urn[u] = {
                "title": str(getattr(row, "title", "")) if pd.notna(getattr(row, "title", None)) else "Uten tittel",
                "authors": str(getattr(row, "authors", "")) if pd.notna(getattr(row, "authors", None)) else "",
                "year": yr,
            }

    df_burst = pd.read_parquet(DATA_DIR / "bbi_book_word_burstiness.parquet")
    totals = pd.read_parquet(DATA_DIR / "totals_500k.parquet").dropna(subset=["word", "total_freq"]).copy()

    # Keep case-sensitive dictionary for proper-name detection
    raw_tot_dict = dict(zip(totals["word"].astype(str), totals["total_freq"]))

    totals["word"] = totals["word"].astype(str).str.lower()
    totals_lc = totals.groupby("word", as_index=False)["total_freq"].sum()
    totals_map = dict(zip(totals_lc["word"], totals_lc["total_freq"]))
    total_bg_sum = float(totals_lc["total_freq"].sum())

    def is_proper_name(w: str) -> bool:
        c_low = raw_tot_dict.get(w, 0)
        c_cap = raw_tot_dict.get(w.capitalize(), 0)
        return (c_low + c_cap > 0) and (c_cap / (c_low + c_cap) > 0.55)

    # Collect BBI diversity words from Luminon + basic_categories
    r1 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_annotations.jsonl", encoding="utf-8")]
    r2 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_round2_annotations.jsonl", encoding="utf-8")]
    bbi_div_words = set()
    for r in r1 + r2:
        for k in ("matched_word", "target_token"):
            if r.get(k):
                bbi_div_words.add(str(r[k]).strip().lower())
    bc = json.loads((BBI_DIR / "basic_categories.json").read_text(encoding="utf-8"))
    for wlist in bc.values():
        for w in wlist:
            bbi_div_words.add(str(w).strip().lower())

    discovered_div_words = {
        "islam", "muslimene", "bønnen", "profeten", "imam", "bønnerommet", "arabisk", "muslimsk", "islamske",
        "funksjonshemmede", "hjelpemidler", "likeverd", "døve", "blinde", "venterommet",
        "homofili", "homser", "homoer", "paraden", "gay", "smittsomt", "unormal",
        "joiker", "joiken", "joike", "same", "finsk", "reinsdyr", "karasjok", "nordlyset", "kautokeino",
        "tater", "somby", "reinen", "reinene", "skallene", "lasso",
        "asyl", "udi", "søknaden", "mottaket", "asylmottak", "asylmottaket", "asylsøker", "asylsøkere",
    }
    must_keep = bbi_div_words | discovered_div_words

    # High-dispersion function words & narrative stop words + proper first names
    auto_func_words = set(
        df_burst.groupby("word")
        .filter(lambda g: len(g) >= 30 and (g["chunk_disp"].median() >= 0.14 or g["df_over_exp"].median() >= 0.88))["word"]
        .unique()
    )
    narrative_and_paratext = {
        "jeg", "mig", "meg", "du", "dig", "deg", "han", "ham", "hun", "ho", "hon", "henne", "hennes",
        "vi", "oss", "dere", "dem", "dei", "deira", "sin", "sine", "sitt", "seg", "sig",
        "ikkje", "ein", "eitt", "kor", "kva", "noko", "nokon", "so", "eg", "me", "dykk", "itte", "inte", "fekk",
        "mannen", "moren", "faren", "gutten", "jenta", "onkel", "tante", "mormor", "morfar", "farmor", "farfar",
        "bestemor", "bestefar", "fru", "herr", "frøken", "muttern", "fattern", "pappa", "mamma",
        "forlag", "gyldendal", "cappelen", "aschehoug", "samlaget", "isbn", "printed", "copyright", "side", "kapittel",
        "the", "and", "you", "will", "was", "that", "for", "with",
    }
    stop_words = (auto_func_words | narrative_and_paratext) - (must_keep - {"tora", "kven", "lam"})

    doc_stats = (
        df_burst.groupby("doc_idx")
        .agg(doc_tokens=("tf", "sum"), n_chunks=("num_chunks", "first"), urn=("urn", "first"))
        .to_dict("index")
    )

    books = []
    active_doc_indices = sorted(df_burst["doc_idx"].unique())
    old_to_new_doc = {old_idx: new_idx for new_idx, old_idx in enumerate(active_doc_indices)}

    for new_idx, old_idx in enumerate(active_doc_indices):
        ds = doc_stats[old_idx]
        u = str(ds["urn"])
        m = meta_by_urn.get(u, {"title": u, "authors": "", "year": 2010})
        books.append({
            "id": new_idx,
            "urn": u,
            "title": m["title"],
            "authors": m["authors"],
            "year": m["year"],
            "n_chunks": int(ds["n_chunks"]),
            "doc_tokens": int(ds["doc_tokens"]),
        })

    # For general vocabulary, keep words with tf >= 6 in a book (or any word in must_keep with tf >= 2)
    df_valid = df_burst[
        df_burst["word"].str.isalpha()
        & (df_burst["word"].str.len() >= 3)
        & ((df_burst["tf"] >= 6) | (df_burst["word"].isin(must_keep) & (df_burst["tf"] >= 2)))
        & (df_burst["word"].isin(totals_map) | df_burst["word"].isin(must_keep))
    ].copy()

    w_agg = (
        df_valid.groupby("word")
        .agg(
            n_books=("doc_idx", "nunique"),
            corpus_freq=("tf", "sum"),
            median_disp=("chunk_disp", "median"),
            median_df_ratio=("df_over_exp", "median"),
            median_fano=("fano", "median"),
            mean_fano=("fano", "mean"),
        )
        .reset_index()
    )
    w_agg = w_agg[(w_agg["n_books"] >= 5) | w_agg["word"].isin(must_keep)].copy()
    w_agg["priority"] = np.log2(1 + w_agg["corpus_freq"]) * np.maximum(0.2, w_agg["mean_fano"])

    top_words_df = pd.concat([
        w_agg[w_agg["word"].isin(must_keep)],
        w_agg[~w_agg["word"].isin(must_keep)].sort_values("priority", ascending=False).head(4800),
    ]).drop_duplicates(subset=["word"]).reset_index(drop=True)

    vocab_list = top_words_df["word"].tolist()
    word_to_id = {w: i for i, w in enumerate(vocab_list)}

    words_meta = []
    for _, r in top_words_df.iterrows():
        w = r["word"]
        is_stop = (w in stop_words) or (is_proper_name(w) and w not in must_keep)
        words_meta.append({
            "w": w,
            "tf_bg": int(totals_map.get(w, 250)),
            "nb": int(r["n_books"]),
            "cf": int(r["corpus_freq"]),
            "md": round(float(r["median_disp"]), 4),
            "mr": round(float(r["median_df_ratio"]), 3),
            "mf": round(float(r["median_fano"]), 3),
            "stop": 1 if is_stop else 0,
        })

    sub = df_valid[df_valid["word"].isin(word_to_id)].copy()
    sub["d_id"] = sub["doc_idx"].map(old_to_new_doc)
    sub["w_id"] = sub["word"].map(word_to_id)
    sub = sub.sort_values(["d_id", "w_id"])

    n_books = len(books)
    counts_per_doc = sub.groupby("d_id").size().reindex(range(n_books), fill_value=0).values
    row_ptr = [0] + np.cumsum(counts_per_doc).astype(int).tolist()

    col_idx = sub["w_id"].astype(int).tolist()
    freqs = sub["tf"].astype(int).tolist()
    fano_x100 = np.round(sub["fano"].values * 100).astype(int).tolist()
    dfratio_x100 = np.round(sub["df_over_exp"].values * 100).astype(int).tolist()

    bundle = {
        "meta": {
            "corpus_name": "BBI Mangfold – Barne- og ungdomsbøker 2000–2024 (Luminon-utvalg)",
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

    out_path = BASE_DIR / "bbi_bundle.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Skrev {out_path} ({size_mb:.2f} MB): {n_books} bøker, {len(words_meta):,} ord, {len(col_idx):,} ikke-null celler.")


if __name__ == "__main__":
    main()
