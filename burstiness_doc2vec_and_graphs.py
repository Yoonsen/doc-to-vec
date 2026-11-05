from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
import igraph as ig

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


def load_burstiness_data():
    corpus = pd.read_csv(BASE_DIR / "vann.csv")
    df_burst = pd.read_parquet(DATA_DIR / "book_word_burstiness.parquet")
    return corpus, df_burst


def build_burst_doc_word_matrix(df_burst: pd.DataFrame, stop_words: set[str], min_books: int = 2):
    """
    Bygger en (bøker x ord) matrise basert på intern chunk-burstiness i hver bok,
    med luking av de mest høyfrekvente funksjons-/pronomenordene og bok-paratekst (ff, flg, fg).
    """
    # Filtrer til alfabetiske ord (minst 3 tegn) utenom stoppord/paratekst
    valid_mask = (
        df_burst["word"].str.isalpha()
        & (df_burst["word"].str.len() >= 3)
        & (~df_burst["word"].isin(stop_words))
    )
    df = df_burst[valid_mask].copy()

    # Filtrer til ord som forekommer i minst min_books bøker
    word_doc_counts = df.groupby("word")["doc_idx"].nunique()
    keep_words = set(word_doc_counts[word_doc_counts >= min_books].index)
    df = df[df["word"].isin(keep_words)].copy()

    # Intern burst-vekt:
    # 1. max(0, fano - 1.15): måler overdispersjon utover tilfeldig Poisson-prosess
    # 2. max(0, 1.0 - df_ratio): belønner ord som konsentreres i færre chunks enn forventet ved tilfeldig spredning
    # 3. log2(1 + freq): skalerer med hvor sentralt ordet er i boka
    excess_fano = np.maximum(0.0, df["fano"].values - 1.15)
    concentration = np.maximum(0.0, 1.0 - df["df_ratio"].values)
    df["burst_weight"] = np.log2(1.0 + df["freq"].values) * excess_fano * concentration

    pos = df[df["burst_weight"] > 0.05].copy()

    doc_ids = np.sort(df["doc_idx"].unique())
    vocab = np.sort(pos["word"].unique())

    doc_to_row = {d: i for i, d in enumerate(doc_ids)}
    word_to_col = {w: j for j, w in enumerate(vocab)}

    rows = pos["doc_idx"].map(doc_to_row).values
    cols = pos["word"].map(word_to_col).values
    vals = pos["burst_weight"].values.astype(np.float32)

    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(doc_ids), len(vocab)))
    return mat, doc_ids, vocab, df


def l2_normalize_rows(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = sparse.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ mat


def l2_normalize_cols(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = sparse.linalg.norm(mat, axis=0)
    norms[norms == 0] = 1.0
    return mat @ sparse.diags(1.0 / norms)


def build_chunk_cooccurrence_graph(
    selected_words: list[str],
    top_k_edges_per_word: int = 12,
    min_pmi: float = 1.5,
):
    """
    Bygger en ord-mot-ord-graf ved å ta dot-produktet på CHUNK-nivå (300-ords avsnitt)
    på tvers av alle bøker, for ordene valgt ut etter burstiness-kriteriet.
    """
    postings = pd.read_parquet(DATA_DIR / "chunk_postings_f5.parquet")
    word_set = set(selected_words)
    sub = postings[postings["word"].isin(word_set)].copy()
    sub["word"] = sub["word"].astype(str)

    # Lag en unik global chunk-ID: (doc_idx, chunk_idx)
    sub["global_chunk"] = sub["doc_idx"].astype(np.int64) * 100_000 + sub["chunk_idx"].astype(np.int64)

    unique_chunks = sub["global_chunk"].unique()
    chunk_to_row = {c: i for i, c in enumerate(unique_chunks)}
    vocab = sorted(selected_words)
    word_to_col = {w: j for j, w in enumerate(vocab)}

    rows = sub["global_chunk"].map(chunk_to_row).values
    cols = sub["word"].map(word_to_col).values
    # Log-frekvens inne i chunken (eller binær / sublineær)
    vals = np.log2(1.0 + sub["count"].values.astype(np.float32))

    C = sparse.csr_matrix((vals, (rows, cols)), shape=(len(unique_chunks), len(vocab)))

    # L2-normaliser kolonnene (ordvektorene over alle chunks) og ta dot-produktet C^T @ C (cosinus-likhet i chunks)
    C_col_norm = l2_normalize_cols(C)
    W_sim = (C_col_norm.T @ C_col_norm).toarray()
    np.fill_diagonal(W_sim, 0.0)

    # Behold topp-k sterkeste naboer per ord (k-NN sparsifisering som i galakser)
    n = len(vocab)
    r_idx, c_idx, data = [], [], []
    for i in range(n):
        row_sims = W_sim[i]
        top_j = np.argsort(row_sims)[::-1][:top_k_edges_per_word]
        for j in top_j:
            if row_sims[j] >= 0.06:
                r_idx.extend([i, j])
                c_idx.extend([j, i])
                data.extend([float(row_sims[j]), float(row_sims[j])])

    cooc_sparse = sparse.csr_matrix((data, (r_idx, c_idx)), shape=(n, n))
    # Dedupliser symmetriske kanter ved å ta gjennomsnitt/maks
    cooc_sparse = cooc_sparse.maximum(cooc_sparse.T)
    return cooc_sparse, vocab, W_sim


def main():
    corpus, df_burst = load_burstiness_data()
    print(f"Lastet burstiness-data: {len(df_burst):,} (bok, ord)-par over {df_burst['doc_idx'].nunique()} bøker.")

    # 1. Undersøk hvordan Norge vs. andre land oppfører seg på burstiness (Fano og DF/E[DF])
    print("\n=== 1. NORGE VS. ANDRE LAND: BURSTINESS OVER ALLE 352 BØKER ===")
    target_places = [
        "norge", "norges", "norsk", "norske",
        "danmark", "sverige", "england", "storbritannia",
        "island", "europa", "russland", "tyskland", "frankrike", "amerika", "usa", "afrika", "asia"
    ]
    sub_places = df_burst[df_burst["word"].isin(target_places)]
    place_summary = (
        sub_places.groupby("word")
        .agg(
            n_books=("doc_idx", "nunique"),
            total_freq=("freq", "sum"),
            median_disp=("disp", "median"),
            median_df_ratio=("df_ratio", "median"),
            median_fano=("fano", "median"),
            mean_fano=("fano", "mean"),
        )
        .loc[[w for w in target_places if w in sub_places["word"].values]]
    )
    print(place_summary.round(3).to_string())

    # 2. Bygg Bok x Ord matrise KUN fra intern burstiness (uten nb.totals!) og kjør SVD
    # Vi identifiserer automatisk globale funksjonsord fra chunk-statistikken (høy median_disp eller høy df_ratio)
    # pluss typisk bok-paratekst (registerforkortelser fgg/flg/side/scanpix) og dialog-pronomen (jeg/du/han/hun/mig).
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
        "norge", "norges", "noreg", "noregs", "norsk", "norske", "historie", "historia",
    }
    stop_words = auto_func_words | paratext_and_pronouns

    mat_burst, doc_ids, vocab, df_with_weights = build_burst_doc_word_matrix(
        df_burst, stop_words=stop_words, min_books=2
    )
    X_norm = l2_normalize_rows(mat_burst)

    u, s, vt = svds(X_norm.astype(np.float64), k=15)
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order, :]
    if u[:, 0].sum() < 0:
        u[:, 0] = -u[:, 0]
        vt[0, :] = -vt[0, :]

    doc_meta = {}
    for idx, row in corpus.iterrows():
        yr = int(row["year"]) if pd.notna(row["year"]) else "?"
        auth = str(row["authors"]).split("/")[0].strip() if pd.notna(row["authors"]) else "?"
        doc_meta[idx] = f"[{yr}] {str(row['title'])[:48]} ({auth})"

    print("\n=== 2. DE FØRSTE 6 SVD-AKSENE FRA REN CHUNK-BURSTINESS (UTEN NB.TOTALS) ===")
    for dim in range(6):
        w_vec, d_vec = vt[dim, :], u[:, dim]
        pos_w = [vocab[i] for i in np.argsort(w_vec)[::-1][:10]]
        neg_w = [vocab[i] for i in np.argsort(w_vec)[:10]]
        pos_d = [doc_meta.get(doc_ids[i], str(doc_ids[i])) for i in np.argsort(d_vec)[::-1][:2]]
        neg_d = [doc_meta.get(doc_ids[i], str(doc_ids[i])) for i in np.argsort(d_vec)[:2]]
        print(f"\n--- Burst-SVD Akse {dim + 1} (σ = {s[dim]:.2f}) ---")
        if dim == 0:
            print(f"  [Tyngdepunkt] Sterkeste burst-ord: {', '.join(pos_w)}")
            print(f"  Topp bøker:   {' | '.join(pos_d)}")
        else:
            print(f"  (+) Ord:   {', '.join(pos_w)}")
            print(f"  (+) Bøker: {' | '.join(pos_d)}")
            print(f"  (-) Ord:   {', '.join(neg_w)}")
            print(f"  (-) Bøker: {' | '.join(neg_d)}")

    # 3. Velg ut de mest 'bursty' innholdsordene i korpuset og bygg ord-mot-ord graf fra chunk-dot-produkt
    print("\n=== 3. ORD-MOT-ORD DOT-PRODUKT (I CHUNKS) & DUAL-GRAPH CLUSTRING ===")
    word_stats = (
        df_with_weights.groupby("word")
        .agg(
            n_books=("doc_idx", "nunique"),
            total_freq=("freq", "sum"),
            median_disp=("disp", "median"),
            median_fano=("fano", "median"),
            mean_fano=("fano", "mean"),
            median_df_ratio=("df_ratio", "median"),
            sum_burst_weight=("burst_weight", "sum"),
        )
        .reset_index()
    )
    # Kriterium:
    # - forekommer i minst 10 bøker
    # - median_disp <= 0.12 (slik at allmenne gjennomgangsord som 'kongen' ikke blir super-hubs)
    # - tydelig bursty: median Fano >= 1.65 og median df_ratio <= 0.78
    cand_words = word_stats[
        (word_stats["n_books"] >= 10)
        & (word_stats["median_disp"] <= 0.12)
        & (word_stats["median_fano"] >= 1.65)
        & (word_stats["median_df_ratio"] <= 0.78)
    ].sort_values("sum_burst_weight", ascending=False).head(400)

    selected_words = cand_words["word"].tolist()
    print(f"Valgte ut {len(selected_words)} høy-burst innholdsord for ord-mot-ord chunk-graf.")

    cooc_sparse, graph_vocab, W_sim = build_chunk_cooccurrence_graph(
        selected_words, top_k_edges_per_word=10
    )
    idx2word = {i: w for i, w in enumerate(graph_vocab)}

    g = sparse_to_igraph(cooc_sparse, threshold=0.0)
    comm_orig = cluster_leiden(g, resolution=1.2, objective="modularity")

    g_dual, orig_edges = build_weighted_line_graph(g, method="adamic_adar")
    comm_dual = cluster_leiden(g_dual, resolution=1.2, objective="modularity")

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

    print(
        f"Graf: {g.vcount()} noder, {g.ecount()} kanter | "
        f"Antall overlapps-klynger: {len(overlapping)} (sanerte hub-klynger: {len(pruned)})"
    )

    # Skriv ut klyngene sortert etter størrelse, navngitt etter de 2 mest sentrale ordene i klyngen
    degrees = g.strength(weights="weight")
    sorted_clusters = sorted(overlapping.items(), key=lambda kv: len(kv[1]), reverse=True)
    print("\nTematiske ordgalakser / klynger (fra chunk-dot-produkt på bursty ord):")
    for cid, node_set in sorted_clusters:
        if len(node_set) < 4:
            continue
        ranked_nodes = sorted(node_set, key=lambda n: degrees[n], reverse=True)
        top2_name = f"{idx2word[ranked_nodes[0]]}-{idx2word[ranked_nodes[1]]}"
        words_preview = [idx2word[n] for n in ranked_nodes[:18]]
        print(f"  • Klynge [{top2_name:<25}] ({len(node_set):>2} ord): {', '.join(words_preview)}")

    # Vis bro-ord (ord som tilhører >= 2 klynger etter linjegraf-utvidelsen)
    bridges = [
        (idx2word[n], len(c_list), c_list)
        for n, c_list in node_to_clusters.items()
        if len(c_list) >= 2
    ]
    bridges.sort(key=lambda x: (x[1], degrees[graph_vocab.index(x[0])]), reverse=True)
    print(f"\nAntall bro-ord (i >= 2 tematiske klynger): {len(bridges)} av {len(graph_vocab)} ({len(bridges)/len(graph_vocab):.1%})")
    if bridges:
        print("Topp 15 bro-ord mellom klyngene:")
        print("  " + ", ".join(f"{w} ({cnt})" for w, cnt, _ in bridges[:15]))


if __name__ == "__main__":
    main()
