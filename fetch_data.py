import time
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://api.nb.no/dhlab"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_totals(top_words: int = 500_000) -> pd.DataFrame:
    out_path = DATA_DIR / f"totals_{top_words // 1000}k.parquet"
    if out_path.exists():
        print(f"Leser eksisterende {out_path}...")
        return pd.read_parquet(out_path)

    print(f"Henter totals({top_words}) fra {BASE_URL}/totals/{top_words} ...")
    t0 = time.time()
    resp = requests.get(f"{BASE_URL}/totals/{top_words}", timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # data kan være dict eller liste av par
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=["word", "total_freq"])
    else:
        df = pd.DataFrame(data, columns=["word", "total_freq"])
    df.to_parquet(out_path, index=False)
    print(f"Lagret {len(df):,} ord til {out_path} på {time.time() - t0:.1f}s")
    return df


def fetch_book_frequencies(urns: list[str], cutoff: int = 5, batch_size: int = 50) -> pd.DataFrame:
    out_path = DATA_DIR / f"book_frequencies_cutoff{cutoff}.parquet"
    if out_path.exists():
        print(f"Leser eksisterende {out_path}...")
        return pd.read_parquet(out_path)

    rows = []
    t0 = time.time()
    for i in range(0, len(urns), batch_size):
        batch = urns[i : i + batch_size]
        print(f"Henter frekvenser (cutoff={cutoff}) for bok {i+1}–{min(i+batch_size, len(urns))} av {len(urns)}...")
        resp = requests.post(
            f"{BASE_URL}/frequencies",
            json={"urns": batch, "cutoff": cutoff, "words": None},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        for doc_entries in result:
            if not doc_entries:
                continue
            for entry in doc_entries:
                # [dhlabid, word, count]
                rows.append((int(entry[0]), str(entry[1]), int(entry[2])))

    df = pd.DataFrame(rows, columns=["dhlabid", "word", "freq"])
    df.to_parquet(out_path, index=False)
    print(f"Lagret {len(df):,} (bok, ord)-par for {df['dhlabid'].nunique()} dokumenter til {out_path} på {time.time() - t0:.1f}s")
    return df


if __name__ == "__main__":
    corpus = pd.read_csv(Path(__file__).parent / "vann.csv")
    urns = corpus["urn"].dropna().unique().tolist()
    print(f"Korpus lastet fra vann.csv: {len(corpus)} rader, {len(urns)} unike URN-er.")

    fetch_totals(500_000)
    fetch_book_frequencies(urns, cutoff=5, batch_size=50)
