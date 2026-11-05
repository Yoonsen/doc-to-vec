import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from pmi_doc2vec import load_corpus_and_data, build_ppmi_matrix, l2_normalize_rows

if __name__ == "__main__":
    corpus, dhlabid_to_urn, totals, freqs = load_corpus_and_data(cutoff=5)

    urn_meta = corpus.set_index("urn")[["title", "authors", "year"]].to_dict("index")
    doc_meta = {}
    for d in freqs["dhlabid"].unique():
        u = dhlabid_to_urn.get(int(d))
        if u and u in urn_meta:
            m = urn_meta[u]
            yr = int(m["year"]) if pd.notna(m["year"]) else "?"
            auth = str(m["authors"]).split("/")[0].strip() if pd.notna(m["authors"]) else "?"
            doc_meta[int(d)] = f"[{yr}] {m['title'][:55]} ({auth})"
        else:
            doc_meta[int(d)] = f"dhlabid={d}"

    mat_tf_ppmi, doc_ids, vocab, merged = build_ppmi_matrix(
        freqs, totals, alpha=0.85, sublinear_tf=True, lowercase=True
    )
    X_norm = l2_normalize_rows(mat_tf_ppmi)

    u, s, vt = svds(X_norm.astype(np.float64), k=15)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    vt = vt[idx, :]

    # For akse 1 (Perron-Frobenius / tyngdepunkt i en ikke-negativ matrise), sørg for positivt fortegn
    if u[:, 0].sum() < 0:
        u[:, 0] = -u[:, 0]
        vt[0, :] = -vt[0, :]

    print("=== DE FØRSTE 10 SVD-AKSENE (PPMI, cutoff=5) ===")
    for dim in range(10):
        w_vec = vt[dim, :]
        d_vec = u[:, dim]

        pos_w = [f"{vocab[i]} ({w_vec[i]:+.3f})" for i in np.argsort(w_vec)[::-1][:8]]
        neg_w = [f"{vocab[i]} ({w_vec[i]:+.3f})" for i in np.argsort(w_vec)[:8]]

        pos_d = [doc_meta[doc_ids[i]] for i in np.argsort(d_vec)[::-1][:3]]
        neg_d = [doc_meta[doc_ids[i]] for i in np.argsort(d_vec)[:3]]

        print(f"\n--- Akse {dim + 1} (singulærverdi σ = {s[dim]:.2f}) ---")
        if dim == 0:
            print("  [Tyngdepunkt / fellesnevner for hele korpuset — alle verdier >= 0]")
            print(f"  Sterkeste ord:   {', '.join(pos_w)}")
            print(f"  Mest typiske:    {' | '.join(pos_d)}")
            print(f"  Minst typiske:   {' | '.join(neg_d)}")
        else:
            print(f"  (+) Ord:   {', '.join(pos_w)}")
            print(f"  (+) Bøker: {' | '.join(pos_d)}")
            print(f"  (-) Ord:   {', '.join(neg_w)}")
            print(f"  (-) Bøker: {' | '.join(neg_d)}")
