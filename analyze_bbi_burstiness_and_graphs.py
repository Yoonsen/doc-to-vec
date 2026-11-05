import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from dual_graph_clustering import (
    sparse_to_igraph,
    cluster_leiden,
    build_weighted_line_graph,
    match_clusters_with_overlap,
    prune_hub_clusters_by_entropy,
    get_node_to_clusters,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BBI_DIR = Path("/Users/larsj/Github/bbi_diversity")


def classify_macro(r):
    k = r["kategori"]
    dom = r["domain"]
    if dom == "funksjon":
        if k == "faktisk_funksjonsnedsettelse":
            return "1_lived_practice_identity"
        elif k in {"historisk_tematisk", "markert_annerledeshet"}:
            return "2_thematic_discourse"
        elif k == "objekt_miljø":
            return "3_spatial_backdrop"
        else:
            return "4_metaphor_idiom_homonym"
    elif dom == "religion":
        if k == "levd_praksis":
            return "1_lived_practice_identity"
        elif k in {"identitetsmarkør", "historisk_tematisk", "markert_annerledeshet"}:
            return "2_thematic_discourse"
        elif k in {"romlig_kulisse", "objekt_miljø", "sekulær_høytid_utrop"}:
            return "3_spatial_backdrop"
        else:
            return "4_metaphor_idiom_homonym"
    elif dom == "kjonn_seksualitet":
        if k == "levd_identitet_relasjon":
            return "1_lived_practice_identity"
        elif k in {"tematisk_diskurs", "skjellsord_negativ_sjargong", "stereotyp"}:
            return "2_thematic_discourse"
        else:
            return "4_metaphor_idiom_homonym"
    elif dom == "nasjonale_minoriteter":
        if k == "levd_kultur_identitet":
            return "1_lived_practice_identity"
        elif k in {"historisk_forfølgelse_konflikt", "stereotyp", "tematisk_diskurs"}:
            return "2_thematic_discourse"
        elif k in {"eksotisk_kulisse_metafor", "bakgrunn"}:
            return "3_spatial_backdrop"
        else:
            return "4_metaphor_idiom_homonym"
    elif dom == "flerkulturell":
        if k == "levd_migrasjon_minoritet":
            return "1_lived_practice_identity"
        elif k in {
            "rasisme_diskriminering_kritikk",
            "historisk_forfølgelse_konflikt",
            "tematisk_diskurs",
            "stereotyp",
        }:
            return "2_thematic_discourse"
        elif k == "utseende_beskrivelse":
            return "3_spatial_backdrop"
        else:
            return "4_metaphor_idiom_homonym"
    return "4_metaphor_idiom_homonym"


def build_chunk_cooccurrence_graph(
    post_df: pd.DataFrame, target_words: list[str], top_k_edges_per_word: int = 10
):
    word_set = set(target_words)
    df = post_df[post_df["word"].isin(word_set)].copy()

    df["global_chunk"] = (
        df["doc_idx"].astype(np.int64) * 100_000 + df["chunk_idx"].astype(np.int64)
    )
    chunk_codes, _ = pd.factorize(df["global_chunk"])

    vocab = sorted(word_set)
    word2id = {w: i for i, w in enumerate(vocab)}
    col_ids = df["word"].map(word2id).to_numpy()
    weights = np.log2(1.0 + df["count"].to_numpy(dtype=np.float32))

    C = sparse.csr_matrix(
        (weights, (chunk_codes, col_ids)),
        shape=(int(chunk_codes.max()) + 1, len(vocab)),
        dtype=np.float32,
    )

    col_norms = np.sqrt(np.asarray(C.power(2).sum(axis=0)).ravel())
    col_norms[col_norms == 0] = 1.0
    C_norm = C @ sparse.diags(1.0 / col_norms)

    W = (C_norm.T @ C_norm).toarray()
    np.fill_diagonal(W, 0.0)

    n = len(vocab)
    rows, cols, vals = [], [], []
    for i in range(n):
        row = W[i]
        pos_idx = np.where(row > 0.02)[0]
        if len(pos_idx) == 0:
            continue
        if len(pos_idx) > top_k_edges_per_word:
            top_idx = pos_idx[np.argsort(row[pos_idx])[::-1][:top_k_edges_per_word]]
        else:
            top_idx = pos_idx
        for j in top_idx:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(row[j]))

    sym_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    sym_mat = sym_mat.maximum(sym_mat.T)
    return sym_mat, vocab, W


def main():
    burst_df = pd.read_parquet(DATA_DIR / "bbi_book_word_burstiness.parquet")
    post_df = pd.read_parquet(DATA_DIR / "bbi_chunk_postings_f3.parquet")
    print(f"Loaded burst_df: {len(burst_df):,} rows across {burst_df['urn'].nunique()} books")
    print(f"Loaded post_df:  {len(post_df):,} rows")

    bw_lookup = {}
    for row in burst_df.itertuples(index=False):
        bw_lookup[(row.urn, row.word)] = row

    r1 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_annotations.jsonl", encoding="utf-8")]
    r2 = [json.loads(line) for line in open(BBI_DIR / "outputs/luminon_round2_annotations.jsonl", encoding="utf-8")]
    for r in r1:
        r["domain"] = (
            "funksjon"
            if r["subgroup"].startswith(("syn_", "horsel_", "bevegelse_", "kognitiv_"))
            else "religion"
        )
    all_recs = r1 + r2

    subgroup_words = defaultdict(set)
    for r in all_recs:
        sub = r["subgroup"]
        for k in ("matched_word", "target_token"):
            if r.get(k):
                subgroup_words[sub].add(str(r[k]).strip().lower())

    urn_sub_tf = defaultdict(int)
    urn_sub_max_fano = defaultdict(lambda: 1.0)
    urn_sub_min_dfratio = defaultdict(lambda: 1.0)
    for row in burst_df.itertuples(index=False):
        for sub, wset in subgroup_words.items():
            if row.word in wset:
                urn_sub_tf[(row.urn, sub)] += row.tf
                if row.fano > urn_sub_max_fano[(row.urn, sub)]:
                    urn_sub_max_fano[(row.urn, sub)] = float(row.fano)
                if row.df_over_exp < urn_sub_min_dfratio[(row.urn, sub)]:
                    urn_sub_min_dfratio[(row.urn, sub)] = float(row.df_over_exp)

    ann_rows = []
    for r in all_recs:
        urn = r["urn"]
        sub = r["subgroup"]
        w = str(r.get("target_token") or r.get("matched_word") or "").strip().lower()
        w_alt = str(r.get("matched_word") or "").strip().lower()
        bw = bw_lookup.get((urn, w)) or bw_lookup.get((urn, w_alt))
        if bw is None:
            continue
        ann_rows.append(
            {
                "urn": urn,
                "domain": r["domain"],
                "subgroup": sub,
                "kategori": r["kategori"],
                "macro_class": classify_macro(r),
                "word": bw.word,
                "word_tf": int(bw.tf),
                "chunk_df": int(bw.chunk_df),
                "num_chunks": int(bw.num_chunks),
                "fano": float(bw.fano),
                "df_over_exp": float(bw.df_over_exp),
                "pmi_intern": float(bw.pmi_intern),
                "sub_tf": int(urn_sub_tf[(urn, sub)]),
                "sub_max_fano": float(urn_sub_max_fano[(urn, sub)]),
                "sub_min_dfratio": float(urn_sub_min_dfratio[(urn, sub)]),
            }
        )

    ann_df = pd.DataFrame(ann_rows)
    print(f"\nMatched {len(ann_df):,} / {len(all_recs):,} Luminon annotations to book-chunk profiles.")

    print("\n" + "=" * 100)
    print("1. HYPOTHESIS 1A: MACRO SEMANTIC CLASS VS. IN-BOOK CHUNK BURSTINESS (ALL 5 DOMAINS)")
    print("=" * 100)
    g_macro = (
        ann_df.groupby("macro_class")
        .agg(
            n=("urn", "count"),
            mean_word_tf=("word_tf", "mean"),
            mean_sub_tf=("sub_tf", "mean"),
            mean_fano=("fano", "mean"),
            mean_sub_fano=("sub_max_fano", "mean"),
            pct_fano_gt_1_15=("sub_max_fano", lambda x: 100.0 * np.mean(x > 1.15)),
            pct_multi_hit=("sub_tf", lambda x: 100.0 * np.mean(x >= 3)),
            mean_df_ratio=("sub_min_dfratio", "mean"),
        )
        .reset_index()
    )
    print(g_macro.to_string(index=False))

    print("\n" + "=" * 100)
    print("2. HYPOTHESIS 1B: KEY CONTRASTS WITHIN SPECIFIC SUBGROUPS (LIVED VS. BACKDROP / IDIOM)")
    print("=" * 100)
    key_subs = [
        "moske_bygg",
        "kirke_bygg",
        "synagoge_bygg",
        "bedehus_bygg",
        "islam_markor",
        "syn_blind",
        "horsel_dov",
        "bevegelse_lam",
        "bevegelse_rullestol",
        "horsel_hjelpemiddel",
        "bevegelse_krykker",
        "homo_kortform",
        "homo_lesbisk_bifil",
        "skeiv_ord",
        "samisk_identitet",
        "samisk_kultur_markor",
        "kven_skogfinne",
        "flyktning_asyl",
        "flukt_generell",
    ]
    sub_df = ann_df[ann_df["subgroup"].isin(key_subs)]
    g_sub = (
        sub_df.groupby(["subgroup", "kategori"])
        .agg(
            n=("urn", "count"),
            books=("urn", "nunique"),
            mean_word_tf=("word_tf", "mean"),
            mean_sub_tf=("sub_tf", "mean"),
            mean_fano=("fano", "mean"),
            mean_sub_fano=("sub_max_fano", "mean"),
            pct_bursty=("sub_max_fano", lambda x: 100.0 * np.mean(x > 1.15)),
            mean_df_ratio=("sub_min_dfratio", "mean"),
        )
        .reset_index()
    )
    g_sub = g_sub[g_sub["n"] >= 8].sort_values(["subgroup", "n"], ascending=[True, False])
    print(g_sub.to_string(index=False))

    # 3. 2x2 DOUBLE BURSTINESS MATRIX (MICRO IN-BOOK VS MACRO ACROSS-BOOKS)
    print("\n" + "=" * 100)
    print("3. 2x2 DOUBLE BURSTINESS TAXONOMY ACROSS 1,794 BBI BOOKS (TF>=3 IN BOOK)")
    print("=" * 100)
    N_books = burst_df["urn"].nunique()
    bf3 = burst_df[(burst_df["tf"] >= 3) & (burst_df["word"].str.isalpha()) & (burst_df["word"].str.len() >= 3)].copy()
    w_stats = (
        bf3.groupby("word")
        .agg(
            n_books=("urn", "nunique"),
            total_tf=("tf", "sum"),
            median_disp=("chunk_disp", "median"),
            median_fano=("fano", "median"),
            mean_fano=("fano", "mean"),
            median_df_ratio=("df_over_exp", "median"),
        )
        .reset_index()
    )
    w_stats["book_share"] = w_stats["n_books"] / N_books

    q1 = w_stats[(w_stats["book_share"] >= 0.35) & (w_stats["median_fano"] <= 1.08)].sort_values("n_books", ascending=False)
    q3 = w_stats[(w_stats["book_share"] >= 0.08) & (w_stats["book_share"] <= 0.45) & (w_stats["median_fano"] >= 1.55) & (w_stats["median_disp"] <= 0.08)].sort_values("median_fano", ascending=False)
    q4 = w_stats[(w_stats["n_books"] >= 8) & (w_stats["book_share"] < 0.08) & (w_stats["median_fano"] >= 1.75) & (w_stats["median_disp"] <= 0.08)].sort_values("total_tf", ascending=False)

    print(f"Quadrant 1 (High Macro Book Spread >=35%, Low Micro Fano <=1.08 — Narrative & Grammatical Glue, n={len(q1)}):")
    print("  Top 25:", ", ".join(f"{r.word} (Fano={r.median_fano:.2f}, books={r.book_share:.0%})" for r in q1.head(25).itertuples()))

    print(f"\nQuadrant 3 (High Macro Book Spread 8–45%, High Micro Fano >=1.55 — Universal Episodic Scenes in YA/Children's Lit, n={len(q3)}):")
    print("  Top 30:", ", ".join(f"{r.word} (Fano={r.median_fano:.2f}, books={r.n_books})" for r in q3.head(30).itertuples()))

    print(f"\nQuadrant 4 (Low Macro Book Spread <8% [8..140 books], High Micro Fano >=1.75 — Specific Thematic Worlds & Minority Motifs, n={len(q4)}):")
    print("  Top 40:", ", ".join(f"{r.word} (Fano={r.median_fano:.2f}, books={r.n_books})" for r in q4.head(40).itertuples()))

    # 4. DUAL-GRAPH CLUSTERING ON DIVERSITY + BURSTY SCENE VOCABULARY
    print("\n" + "=" * 100)
    print("4. DUAL-GRAPH L(G) OVERLAPPING CLUSTERING OF BBI DIVERSITY & SCENE WORDS")
    print("=" * 100)
    div_candidates = w_stats[
        (w_stats["n_books"] >= 5)
        & (w_stats["median_disp"] <= 0.08)
        & (w_stats["median_fano"] >= 1.45)
        & (w_stats["median_df_ratio"] <= 0.84)
    ].sort_values("total_tf", ascending=False)

    # Combine top 350 bursty thematic words + all BBI diversity words that appear in >= 4 books with TF>=3
    bbi_div_words = set()
    for s in subgroup_words.values():
        bbi_div_words.update(s)
    valid_div_words = set(w_stats[w_stats["n_books"] >= 4]["word"]).intersection(bbi_div_words)
    selected_words = sorted(set(div_candidates.head(350)["word"]).union(valid_div_words))
    print(f"Selected {len(selected_words)} words ({len(valid_div_words)} BBI diversity markers + top bursty thematic words) for chunk co-occurrence graph.")

    cooc_sparse, graph_vocab, _ = build_chunk_cooccurrence_graph(post_df, selected_words, top_k_edges_per_word=10)
    idx2word = {i: w for i, w in enumerate(graph_vocab)}

    g = sparse_to_igraph(cooc_sparse, threshold=0.0)
    comm_orig = cluster_leiden(g, resolution=1.3, objective="modularity")
    g_dual, orig_edges = build_weighted_line_graph(g, method="adamic_adar")
    comm_dual = cluster_leiden(g_dual, resolution=1.3, objective="modularity")

    overlapping = match_clusters_with_overlap(
        comm_orig,
        comm_dual,
        orig_edges,
        strategy="shielded",
        min_internal_support=2,
        dedup_jaccard_threshold=0.80,
    )
    adj = {i: set(g.neighbors(i)) for i in range(g.vcount())}
    overlapping, pruned = prune_hub_clusters_by_entropy(overlapping, adj=adj, tau_H=0.78, tau_I=0.10)
    node_to_clusters = get_node_to_clusters(overlapping)

    degrees = g.strength(weights="weight")
    sorted_clusters = sorted(overlapping.items(), key=lambda kv: len(kv[1]), reverse=True)
    print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges | Overlapping clusters: {len(overlapping)} (pruned hubs: {len(pruned)})")
    for cid, node_set in sorted_clusters:
        if len(node_set) < 4:
            continue
        ranked_nodes = sorted(node_set, key=lambda n: degrees[n], reverse=True)
        top2_name = f"{idx2word[ranked_nodes[0]]}-{idx2word[ranked_nodes[1]]}"
        words_preview = [idx2word[n] for n in ranked_nodes[:18]]
        div_in_cluster = [idx2word[n] for n in ranked_nodes if idx2word[n] in valid_div_words]
        div_tag = f" [BBI-ord: {', '.join(div_in_cluster[:8])}]" if div_in_cluster else ""
        print(f"  • [{top2_name:<25}] ({len(node_set):>2}w): {', '.join(words_preview)}{div_tag}")


if __name__ == "__main__":
    main()
