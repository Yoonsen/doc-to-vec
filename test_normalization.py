import re
import numpy as np
import pandas as pd
from pathlib import Path
from pmi_doc2vec import load_corpus_and_data, build_ppmi_matrix, l2_normalize_rows, compute_svd_embeddings

# Høyfrekvente, entydige dansk-norske -> moderne bokmålsformer i historietekster
EXACT_HIST_MAP = {
    "aar": "år", "aaret": "året", "aarene": "årene", "aarhundrede": "århundre", "aarhundreder": "århundreder",
    "efter": "etter", "efterat": "etterat", "bagefter": "baketter", "derefter": "deretter", "herefter": "heretter",
    "blev": "ble", "bleve": "ble", "bleven": "blitt", "blevet": "blitt",
    "mod": "mot", "imod": "imot",
    "ogsaa": "også", "saa": "så", "saaledes": "således", "saadan": "sådan", "saadanne": "sådanne",
    "paa": "på", "oppe": "oppe", "op": "opp",
    "under": "under", "uden": "uten", "udenfor": "utenfor", "ud": "ut", "ude": "ute", "udvikling": "utvikling", "udviklingen": "utviklingen",
    "ind": "inn", "inde": "inne", "inden": "innen", "indenfor": "innenfor",
    "igjennem": "igjennom", "gjennem": "gjennom", "imellem": "mellom", "mellem": "mellom",
    "mand": "mann", "manden": "mannen", "mænd": "menn", "mændene": "mennene",
    "kvinde": "kvinne", "kvinden": "kvinnen", "kvinder": "kvinner", "kvinderne": "kvinnene",
    "rige": "rike", "riget": "riket", "rigets": "rikets", "riger": "riker",
    "krig": "krig", "krigen": "krigen",
    "skib": "skip", "skibe": "skip", "skibene": "skipene",
    " kjøbmand": "kjøpmann", "kjøbmand": "kjøpmann", "kjøbmænd": "kjøpmenn",
    "embedsmand": "embetsmann", "embedsmænd": "embetsmenn", "embedsmændene": "embetsmennene", "embede": "embete",
    "bønderne": "bøndene", "kongerne": "kongene", "byerne": "byene", "lovene": "lovene",
    "tidsrum": "tidsrom", "tidsrummet": "tidsrommet",
    "sproget": "språket", "sprog": "språk",
    "gaa": "gå", "gaar": "går", "gaaet": "gått", "faar": "får", "faa": "få", "faaet": "fått",
    "staa": "stå", "staar": "står", "staaet": "stått", "naa": "nå", "naaede": "nådde",
    "gaard": "gård", "gaarden": "gården", "gaarde": "gårder", "gaardene": "gårdene",
    "baade": "både", "maatte": "måtte", "maa": "må", "maade": "måte", "maaden": "måten",
    "raad": "råd", "raadet": "rådet", " rigsraad": "riksråd", "rigsraad": "riksråd", "rigsraadet": "riksrådet",
    "haakon": "håkon", "olafr": "olav", "olaf": "olav",
}


def make_normalizer(modern_vocab: set[str]):
    """
    Lager en forsiktig normaliseringsfunksjon:
    1. lowercasing
    2. eksakt høyfrekvent historisk tabell
    3. konservative suffiks-/prefiks- og aa->å-regler KUN hvis målformen finnes blant kjente norske ord.
    """
    cache = {}

    def normalize_word(w: str) -> str:
        if not isinstance(w, str):
            return ""
        wl = w.lower()
        if wl in cache:
            return cache[wl]

        if wl in EXACT_HIST_MAP:
            cache[wl] = EXACT_HIST_MAP[wl]
            return cache[wl]

        cand = wl
        # 1. aa -> å (hvis ordet har minst 3 tegn og ikke bare er 'aa')
        if "aa" in cand and len(cand) >= 3:
            c2 = cand.replace("aa", "å")
            if c2 in modern_vocab:
                cand = c2

        # 2. Vanlige dansk-norske suffikser -> moderne bokmål, kun hvis målordet finnes i modern_vocab
        for old_suf, new_suf in [
            ("ighed", "ighet"),
            ("igheden", "igheten"),
            ("igheder", "igheter"),
            ("skab", "skap"),
            ("skabet", "skapet"),
            ("skaber", "skaper"),
            ("mænd", "menn"),
            ("mand", "mann"),
            ("kvinde", "kvinne"),
            ("kvinder", "kvinner"),
            ("rige", "rike"),
            ("riget", "riket"),
            ("tidsrum", "tidsrom"),
        ]:
            if cand.endswith(old_suf) and len(cand) > len(old_suf) + 1:
                c2 = cand[: -len(old_suf)] + new_suf
                if c2 in modern_vocab:
                    cand = c2
                    break

        cache[wl] = cand
        return cand

    return normalize_word


def evaluate_cross_period(sim_mat: np.ndarray, years: np.ndarray, k: int = 5) -> float:
    """Måler hvor stor andel av topp-k naboer som krysser 1917-grensen (før 1917 vs >= 1917)."""
    is_modern = years >= 1917
    cross_hits = 0
    total = 0
    for i in range(len(years)):
        if np.isnan(years[i]):
            continue
        top_k = np.argsort(sim_mat[i])[::-1][1 : k + 1]
        for j in top_k:
            if np.isnan(years[j]):
                continue
            if is_modern[i] != is_modern[j]:
                cross_hits += 1
            total += 1
    return cross_hits / total if total else 0.0


if __name__ == "__main__":
    corpus, dhlabid_to_urn, totals, freqs = load_corpus_and_data(cutoff=5)
    urn_year = corpus.set_index("urn")["year"].to_dict()

    # 1. Baseline (case-sensitiv)
    mat_raw, doc_ids, _, _ = build_ppmi_matrix(freqs, totals, alpha=0.85, sublinear_tf=True)
    years = np.array([urn_year.get(dhlabid_to_urn.get(int(d), ""), np.nan) for d in doc_ids])
    X_raw = l2_normalize_rows(mat_raw)
    emb_raw = compute_svd_embeddings(X_raw, 100)

    # 2. Kun lowercasing
    totals_lc = totals.dropna(subset=["word"]).copy()
    totals_lc["word"] = totals_lc["word"].astype(str).str.lower()
    totals_lc = totals_lc.groupby("word", as_index=False)["total_freq"].sum()

    freqs_lc = freqs.copy()
    freqs_lc["word"] = freqs_lc["word"].astype(str).str.lower()
    freqs_lc = freqs_lc.groupby(["dhlabid", "word"], as_index=False)["freq"].sum()

    mat_lc, _, _, _ = build_ppmi_matrix(freqs_lc, totals_lc, alpha=0.85, sublinear_tf=True)
    X_lc = l2_normalize_rows(mat_lc)
    emb_lc = compute_svd_embeddings(X_lc, 100)

    # 3. Lowercasing + forsiktig ordboksverifisert normalisering
    # Vi bruker de 250 000 mest frekvente ordene (uten "aa") som fasit på gyldige moderne målformer
    modern_vocab = set(totals_lc[totals_lc["total_freq"] >= 500]["word"])
    norm_fn = make_normalizer(modern_vocab)

    totals_norm = totals_lc.copy()
    totals_norm["word"] = totals_norm["word"].map(norm_fn)
    totals_norm = totals_norm.groupby("word", as_index=False)["total_freq"].sum()

    freqs_norm = freqs_lc.copy()
    freqs_norm["word"] = freqs_norm["word"].map(norm_fn)
    freqs_norm = freqs_norm.groupby(["dhlabid", "word"], as_index=False)["freq"].sum()

    mat_norm, _, _, _ = build_ppmi_matrix(freqs_norm, totals_norm, alpha=0.85, sublinear_tf=True)
    X_norm = l2_normalize_rows(mat_norm)
    emb_norm = compute_svd_embeddings(X_norm, 100)

    for label, X_sp, emb in [
        ("A. Rå (case-sensitiv)", X_raw, emb_raw),
        ("B. Lowercase", X_lc, emb_lc),
        ("C. Lowercase + konservativ normalisering", X_norm, emb_norm),
    ]:
        sp_sim = (X_sp @ X_sp.T).toarray()
        svd_sim = emb @ emb.T
        print(
            f"{label:<42} | Vocab: {X_sp.shape[1]:>6,} | "
            f"Kryss-1917 (sparse): {evaluate_cross_period(sp_sim, years):.1%} | "
            f"Kryss-1917 (SVD-100): {evaluate_cross_period(svd_sim, years):.1%}"
        )
