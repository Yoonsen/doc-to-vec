import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def main():
    corpus = pd.read_csv(BASE_DIR / "vann.csv")
    df_burst = pd.read_parquet(DATA_DIR / "book_word_burstiness.parquet")
    totals = pd.read_parquet(DATA_DIR / "totals_500k.parquet")

    # Lowercase og summer totals
    totals = totals.dropna(subset=["word", "total_freq"]).copy()
    totals["word"] = totals["word"].astype(str).str.lower()
    totals_lc = totals.groupby("word", as_index=False)["total_freq"].sum()
    totals_map = dict(zip(totals_lc["word"], totals_lc["total_freq"]))
    total_bg_sum = float(totals_lc["total_freq"].sum())

    # Automatisk deteksjon av funksjonsord + paratekst
    auto_func_words = set(
        df_burst.groupby("word")
        .filter(lambda g: len(g) >= 20 and (g["disp"].median() >= 0.22 or g["df_ratio"].median() >= 0.88))["word"]
        .unique()
    )
    paratext_and_pronouns = {
        "jeg", "mig", "meg", "du", "dig", "deg", "han", "ham", "hannem", "hann", "hun", "ho", "hon",
        "hende", "henne", "hendes", "hennes", "vi", "oss", "os", "dere", "dem", "dei", "deira",
        "sin", "av", "af", "aff", "på", "paa", "til", "till", "og", "och", "oc", "udi", "ikke", "ikkje", "icke",
        "var", "war", "har", "hadde", "havde", "ble", "blev", "bleve", "vart", "skal", "skulle", "skulde",
        "kan", "kunne", "kunde", "vil", "ville", "vilde", "må", "måtte", "maatte", "fikk", "fik", "fekk",
        "flg", "fgg", "fg", "ff", "fig", "side", "sider", "kap", "bd", "bind", "nr", "utg",
        "øverst", "nederst", "venstre", "høyre", "høire", "midten", "foto", "scanpix", "ill", "tabell",
        "the", "and", "of", "in", "to", "und", "der", "die", "das", "was", "item", "ibid", "op", "opp",
    }
    stop_words = auto_func_words | paratext_and_pronouns

    # Dokumentlengder (sum av ord med freq >= 5 i boka) og antall chunks
    doc_stats = (
        df_burst.groupby("doc_idx")
        .agg(doc_tokens=("freq", "sum"), n_chunks=("n_chunks", "first"))
        .to_dict("index")
    )

    books = []
    active_doc_indices = sorted(df_burst["doc_idx"].unique())
    old_to_new_doc = {old_idx: new_idx for new_idx, old_idx in enumerate(active_doc_indices)}

    for new_idx, old_idx in enumerate(active_doc_indices):
        row = corpus.iloc[old_idx]
        ds = doc_stats.get(old_idx, {"doc_tokens": 1, "n_chunks": 1})
        books.append({
            "id": new_idx,
            "urn": str(row["urn"]),
            "title": str(row["title"]) if pd.notna(row["title"]) else "Uten tittel",
            "authors": str(row["authors"]) if pd.notna(row["authors"]) else "",
            "year": int(row["year"]) if pd.notna(row["year"]) else 0,
            "n_chunks": int(ds["n_chunks"]),
            "doc_tokens": int(ds["doc_tokens"]),
        })

    # Velg vokabular: ord med minst 2 tegn (alfanumeriske) som finnes i totals_map og i minst 4 bøker,
    # prioritert etter en kombinasjon av antall bøker og burstiness (topp 6 500 ord)
    df_valid = df_burst[
        df_burst["word"].str.isalnum()
        & (df_burst["word"].str.len() >= 2)
        & df_burst["word"].isin(totals_map)
    ].copy()

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
    w_agg = w_agg[w_agg["n_books"] >= 4].copy()
    w_agg["priority"] = np.log2(1 + w_agg["corpus_freq"]) * np.maximum(0.2, w_agg["mean_fano"])
    # Sørg også for at viktige land/geografiske ord er med uansett
    must_keep = {
        "norge", "norges", "noreg", "norsk", "norske", "danmark", "sverige", "england", "storbritannia",
        "island", "europa", "russland", "sovjet", "tyskland", "preussen", "frankrike", "italia", "rom",
        "spania", "nederland", "holland", "finland", "amerika", "usa", "afrika", "asia", "india", "kina",
    }
    top_words_df = pd.concat([
        w_agg[w_agg["word"].isin(must_keep)],
        w_agg[~w_agg["word"].isin(must_keep)].sort_values("priority", ascending=False).head(6500),
    ]).drop_duplicates(subset=["word"]).reset_index(drop=True)

    vocab_list = top_words_df["word"].tolist()
    word_to_id = {w: i for i, w in enumerate(vocab_list)}

    words_meta = []
    for i, r in top_words_df.iterrows():
        w = r["word"]
        words_meta.append({
            "w": w,
            "tf_bg": int(totals_map.get(w, 100)),
            "nb": int(r["n_books"]),
            "cf": int(r["corpus_freq"]),
            "md": round(float(r["median_disp"]), 4),
            "mr": round(float(r["median_df_ratio"]), 3),
            "mf": round(float(r["median_fano"]), 3),
            "stop": 1 if (w in stop_words or not w.isalpha() or len(w) < 3) else 0,
        })

    # Filtrer df_valid til valgt vokabular og sorter etter (new_doc_idx, word_id) for CSR-konstruksjon
    sub = df_valid[df_valid["word"].isin(word_to_id)].copy()
    sub["d_id"] = sub["doc_idx"].map(old_to_new_doc)
    sub["w_id"] = sub["word"].map(word_to_id)
    sub = sub.sort_values(["d_id", "w_id"])

    # Kompakte flate tabeller for CSR i JS:
    # row_ptr (lengde n_books + 1), col_idx, freq, fano_x100 (int), dfratio_x100 (int)
    n_books = len(books)
    counts_per_doc = sub.groupby("d_id").size().reindex(range(n_books), fill_value=0).values
    row_ptr = [0] + np.cumsum(counts_per_doc).astype(int).tolist()

    col_idx = sub["w_id"].astype(int).tolist()
    freqs = sub["freq"].astype(int).tolist()
    fano_x100 = np.round(sub["fano"].values * 100).astype(int).tolist()
    dfratio_x100 = np.round(sub["df_ratio"].values * 100).astype(int).tolist()

    bundle = {
        "meta": {
            "corpus_name": "Terje Tvedt – Historieverk (vann.csv)",
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

    out_path = BASE_DIR / "tvedt_bundle.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Skrev {out_path} ({size_mb:.2f} MB): {n_books} bøker, {len(words_meta):,} ord, {len(col_idx):,} ikke-null celler.")


if __name__ == "__main__":
    main()
