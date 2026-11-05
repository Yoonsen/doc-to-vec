from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def load_corpus_and_data(cutoff: int = 5):
    corpus = pd.read_csv(BASE_DIR / "vann.csv")
    corpus["dhlabid_int"] = corpus["dhlabid"].dropna().astype(int)

    # Bygg dhlabid -> urn mapping (både direkte fra vann.csv og fra identifier_mapping)
    dhlabid_to_urn = dict(zip(corpus["dhlabid_int"], corpus["urn"]))
    map_path = DATA_DIR / "historieverk_identifier_mapping.csv"
    if map_path.exists():
        id_map = pd.read_csv(map_path)
        for _, row in id_map.iterrows():
            dhlabid_to_urn.setdefault(int(row["dhlabid"]), row["urn"])

    totals = pd.read_parquet(DATA_DIR / "totals_500k.parquet")
    freqs = pd.read_parquet(DATA_DIR / f"book_frequencies_cutoff{cutoff}.parquet")

    return corpus, dhlabid_to_urn, totals, freqs


def build_ppmi_matrix(
    freqs: pd.DataFrame,
    totals: pd.DataFrame,
    alpha: float = 1.0,
    sublinear_tf: bool = False,
    lowercase: bool = True,
):
    """
    Bygger en sparse (dokumenter x ord) PPMI-matrise.
    - alpha: eksponent på bakgrunnsfrekvens P(w)^alpha (1.0 = standard PMI)
    - sublinear_tf: hvis True, multipliseres PPMI med log2(1 + c(w, d))
    - lowercase: hvis True, slås store/små bokstaver sammen i både bokfrekvenser og totals før PMI
    """
    totals = totals.dropna(subset=["word", "total_freq"]).copy()
    totals = totals[totals["total_freq"] > 0]

    if lowercase:
        totals["word"] = totals["word"].astype(str).str.lower()
        totals = totals.groupby("word", as_index=False)["total_freq"].sum()

        freqs = freqs.copy()
        freqs["word"] = freqs["word"].astype(str).str.lower()
        freqs = freqs.groupby(["dhlabid", "word"], as_index=False)["freq"].sum()
    else:
        totals = totals.drop_duplicates(subset=["word"])

    # Beregn dokumentlengder (sum av ord med freq >= cutoff i boka) før eller etter filtrering
    doc_totals = freqs.groupby("dhlabid")["freq"].sum().rename("doc_len")

    # Filtrer til topp 500 000 fra totals
    merged = freqs.merge(totals[["word", "total_freq"]], on="word", how="inner")
    merged = merged.merge(doc_totals, on="dhlabid", how="left")

    # Sannsynligheter
    p_w_given_d = merged["freq"].values / merged["doc_len"].values

    if alpha == 1.0:
        total_bg = totals["total_freq"].sum()
        p_w = merged["total_freq"].values / total_bg
    else:
        bg_powered = np.power(totals["total_freq"].values.astype(np.float64), alpha)
        total_bg = bg_powered.sum()
        p_w = np.power(merged["total_freq"].values.astype(np.float64), alpha) / total_bg

    pmi = np.log2(p_w_given_d / p_w)
    ppmi = np.maximum(0.0, pmi)

    if sublinear_tf:
        weights = ppmi * np.log2(1.0 + merged["freq"].values)
    else:
        weights = ppmi

    merged["pmi"] = pmi
    merged["weight"] = weights

    # Beholder kun positive vekter i sparse-matrisen
    pos = merged[merged["weight"] > 0].copy()

    doc_ids = np.sort(freqs["dhlabid"].unique())
    vocab = np.sort(pos["word"].unique())

    doc_to_idx = {d: i for i, d in enumerate(doc_ids)}
    word_to_idx = {w: j for j, w in enumerate(vocab)}

    row_ind = pos["dhlabid"].map(doc_to_idx).values
    col_ind = pos["word"].map(word_to_idx).values
    data = pos["weight"].values.astype(np.float32)

    mat = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(doc_ids), len(vocab)))
    return mat, doc_ids, vocab, merged


def l2_normalize_rows(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = sparse.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    inv_norms = sparse.diags(1.0 / norms)
    return inv_norms @ mat


def compute_svd_embeddings(mat_norm: sparse.csr_matrix, n_components: int = 100) -> np.ndarray:
    k = min(n_components, min(mat_norm.shape) - 1)
    u, s, _ = svds(mat_norm.astype(np.float64), k=k)
    # svds returnerer egenverdier i stigende rekkefølge; snu til synkende
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    # Vektet med rot av singulærverdier (klassisk LSA/PPMI-faktorisering: U * S^0.5)
    emb = u * np.sqrt(s)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (emb / norms).astype(np.float32)


if __name__ == "__main__":
    corpus, dhlabid_to_urn, totals, freqs = load_corpus_and_data(cutoff=5)

    # Koble dhlabid -> tittel, forfatter, år
    urn_meta = corpus.set_index("urn")[["title", "authors", "year"]].to_dict("index")
    doc_meta = {}
    for d in freqs["dhlabid"].unique():
        u = dhlabid_to_urn.get(int(d))
        if u and u in urn_meta:
            m = urn_meta[u]
            doc_meta[int(d)] = f"{m['title']} ({int(m['year']) if pd.notna(m['year']) else '?'}) – {m['authors']}"
        else:
            doc_meta[int(d)] = f"dhlabid={d}"

    raw_types = freqs["word"].nunique()
    raw_pairs = len(freqs)
    raw_tokens = freqs["freq"].sum()

    print("=== 1. DATA & FILTRERING (cutoff=5 + totals 500k) ===")
    mat_raw_ppmi, doc_ids, vocab, merged = build_ppmi_matrix(
        freqs, totals, alpha=1.0, sublinear_tf=False
    )
    kept_types = merged["word"].nunique()
    kept_pairs = len(merged)
    kept_tokens = merged["freq"].sum()

    print(f"Antall bøker med data:       {len(doc_ids)} (av {len(corpus)} i vann.csv)")
    print(f"Unike ordtyper (cutoff>=5):  {raw_types:,} -> etter topp 500k filter: {kept_types:,} ({kept_types/raw_types:.1%})")
    print(f"(Bok, ord)-par (cutoff>=5):  {raw_pairs:,} -> etter topp 500k filter: {kept_pairs:,} ({kept_pairs/raw_pairs:.1%})")
    print(f"Sum ordforekomster (tokens): {raw_tokens:,} -> etter topp 500k filter: {kept_tokens:,} ({kept_tokens/raw_tokens:.1%})")
    print(f"PPMI > 0 matriseform:        {mat_raw_ppmi.shape[0]} bøker x {mat_raw_ppmi.shape[1]:,} ord ({mat_raw_ppmi.nnz:,} ikke-null celler)")

    # Bygg også log-TF vektet PPMI (ofte mer balansert mellom sjeldne og sentrale fagord)
    mat_tf_ppmi, _, vocab_tf, merged_tf = build_ppmi_matrix(
        freqs, totals, alpha=0.85, sublinear_tf=True
    )

    # Eksempel på toppord i én bok med ren PPMI vs log-tf * PPMI(alpha=0.85)
    sample_doc = doc_ids[4]  # En bok fra korpuset
    print(f"\n=== 2. EKSEMPEL PÅ TOPPORD I BOK: {doc_meta[sample_doc]} ===")
    sub1 = merged[merged["dhlabid"] == sample_doc].sort_values("weight", ascending=False).head(12)
    print("\nRen PPMI (topp 12 ord):")
    for _, r in sub1.iterrows():
        print(f"  {r['word']:<22} freq={int(r['freq']):>4}  total={int(r['total_freq']):>10,}  PPMI={r['weight']:.2f}")

    sub2 = merged_tf[merged_tf["dhlabid"] == sample_doc].sort_values("weight", ascending=False).head(12)
    print("\nLog-TF * PPMI (alpha=0.85) (topp 12 ord):")
    for _, r in sub2.iterrows():
        print(f"  {r['word']:<22} freq={int(r['freq']):>4}  total={int(r['total_freq']):>10,}  vekt={r['weight']:.2f}")

    # Beregn dokumentlikhet og SVD-vektorer (100 dim)
    X_sparse_norm = l2_normalize_rows(mat_tf_ppmi)
    sim_sparse = (X_sparse_norm @ X_sparse_norm.T).toarray()

    emb_100 = compute_svd_embeddings(X_sparse_norm, n_components=100)
    sim_svd = emb_100 @ emb_100.T

    # Lagre matrise og SVD-vektorer
    sparse.save_npz(DATA_DIR / "ppmi_tf_matrix.npz", X_sparse_norm)
    np.save(DATA_DIR / "doc2vec_svd100.npy", emb_100)
    pd.DataFrame({
        "dhlabid": doc_ids,
        "urn": [dhlabid_to_urn.get(int(d), "") for d in doc_ids],
        "label": [doc_meta[int(d)] for d in doc_ids],
    }).to_csv(DATA_DIR / "doc_index.csv", index=False)

    print("\n=== 3. NÆRMESTE NABOER (DOC2VEC / PPMI) FOR UTVALGTE BØKER ===")
    for test_idx in [0, 4, 18, 50, 120]:
        if test_idx >= len(doc_ids):
            continue
        print(f"\nOppslagsbok [{test_idx}]: {doc_meta[doc_ids[test_idx]]}")
        # Topp 4 naboer i SVD-100
        scores = sim_svd[test_idx]
        top_k = np.argsort(scores)[::-1][1:5]
        for rank, nb_idx in enumerate(top_k, 1):
            print(
                f"  {rank}. (svd={scores[nb_idx]:.3f}, sparse={sim_sparse[test_idx, nb_idx]:.3f}) "
                f"{doc_meta[doc_ids[nb_idx]]}"
            )

