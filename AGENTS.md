# AGENTS.md — AI Technical Handover & Architecture Contract (`doc-to-vec`)

Dette dokumentet er skrevet for AI-agenter (Antigravity, Claude, Cursor, m.fl.) som enten jobber direkte i dette repoet eller leser dette repoet som kilde når funksjonalitet skal splittes ut til nye prosjekter (f.eks. DDK/emneord-prediksjon, BBI/Luminon-analyse eller anbefalingssystemer).

---

## 1. Hva dette repoet inneholder (3 logiske deler som kan splittes)

Repoet rommer for øyeblikket tre delprosjekter som deler samme matematiske kjerne:

1. **Kjernebibliotek & CLI for 4-kanals dokument- og ordvektorisering (Python):**
   - [build_bundle_from_urns.py](file:///Users/larsj/Github/doc-to-vec/build_bundle_from_urns.py): Tar en hvilken som helst liste/fil med URN-er, henter metadata og 250-ords tekstblokker fra NB DHlab API med 12 tråder, og bygger en CSR `.json`-pakke.
   - [dual_graph_clustering.py](file:///Users/larsj/Github/doc-to-vec/dual_graph_clustering.py): Overlappende ord-klustring via Leiden på primærgrafen $G$ + Adamic–Adar-vektet linjegraf $L(G)$ med entropibasert beskjæring av bro-ord.
   - [pmi_doc2vec.py](file:///Users/larsj/Github/doc-to-vec/pmi_doc2vec.py) og [burstiness_doc2vec_and_graphs.py](file:///Users/larsj/Github/doc-to-vec/burstiness_doc2vec_and_graphs.py): Referanseimplementasjoner i Python/SciPy for Shifted PPMI, innen-bok chunk-burstiness og Truncated SVD.
   - `data/totals_500k.parquet`: Referansefrekvenser for de 500 000 vanligste ordene i Nasjonalbiblioteket (`nb.totals(500_000)`).

2. **Frittstående nettleser-applikasjon (Vanilla JS / GitHub Pages):**
   - [index.html](file:///Users/larsj/Github/doc-to-vec/index.html), [app.js](file:///Users/larsj/Github/doc-to-vec/app.js), [style.css](file:///Users/larsj/Github/doc-to-vec/style.css): 100 % klientbasert utforsker og vektoreksportør. Ingen bundler/build-step; deployes direkte via `.github/workflows/deploy.yml`.
   - Offentlige demokorpus i git: [mat_bundle.json](file:///Users/larsj/Github/doc-to-vec/mat_bundle.json) (DDK 641 kokebøker, 338 bøker) og [kane_bundle.json](file:///Users/larsj/Github/doc-to-vec/kane_bundle.json) (Morgan Kane, 255 bøker).

3. **Private forskningskorpus & DH2027-artikkel (KUN LOKALT — `.gitignore`):**
   - `tvedt_bundle.json`, `vann.csv` (Terje Tvedt-korpuset, 352 historieverk).
   - `bbi_bundle.json`, `fetch_bbi_burstiness.py`, `analyze_bbi_burstiness_and_graphs.py`, `export_bbi_bundle.py`, `data/bbi_*.parquet` (BBI Diversity-korpuset, 1 794 bøker kryssvalidert mot 6 137 LLM *luminon*-annoteringer).
   - `dh2027_abstract_draft.md` (utkast til DH2027 Long Paper).
   - **VIKTIG REGEL:** Disse filene er listet i `.gitignore` og skal **aldri** committes til det offentlige `doc-to-vec`-repoet.

---

## 2. Matematisk kontrakt: De 4 vektorkanalene (Tensor-representasjon)

For en bok $d$ delt inn i $K_d$ tekstblokker (chunks à 250 eller 300 ord) og et ord $w$ med blokk-tellinger $(c_1, \dots, c_{K_d})$:

1. **In-book Chunk Burstiness:**
   - Gjennomsnitt per blokk: $\mu = \frac{1}{K_d}\sum_{j=1}^{K_d} c_j = \frac{\text{TF}_{d,w}}{K_d}$
   - **Fano-faktor (overdispersjon):** $\text{Fano}_{d,w} = \frac{\sigma^2}{\mu}$
   - **Chunk-spredning vs. Poisson-forventning:** $\text{DF\_ratio}_{d,w} = \frac{\text{DF}_{\text{chunk}}(w)}{K_d(1 - e^{-\mu})}$
   - *Tolkning:* Funksjonsord, metaforer og idiomer følger en homogen Poisson-prosess ($\text{Fano} \approx 1.0$, $\text{DF\_ratio} \approx 1.0$). Reelle tematiske drøftinger og lokaliserte episoder klumper seg i bestemte kapitler ($\text{Fano} \gg 1.2$, $\text{DF\_ratio} < 0.85$).

2. **Makro-spesifisitet (Shifted PPMI mot `nb.totals`):**
   $$\text{PPMI}_{d,w} = \max\!\left(0,\; \log_2 \frac{P(w \mid d)}{P_{\text{bg}}(w)^\alpha} - \log_2 k\right)$$
   der $\alpha \in [0.75, 1.0]$ er kontekstutjevning og $k$ er shift-parameteren.

3. **De 4 eksplisitte bokvektorene (partielle funksjoner $f_d : V_d \to \mathbb{R}_{\ge 0}$):**
   - **$\mathbf{q}_{\text{burst}}$ (`burst`):** $\ln(1 + \text{TF}_{d,w}) \cdot \max(0, \text{Fano}_{d,w} - 0.85) \cdot \max(0, 1.05 - \text{DF\_ratio}_{d,w})$. Isolerer ord som drøftes intensivt i avgrensede deler av boka. Ideell for **spesifikk emneordsprediksjon (Subject Headings)**.
   - **$\mathbf{q}_{\text{frame}}$ (`frame`):** $\ln(1 + \text{TF}_{d,w}) \cdot \min(1.2, \text{DF\_ratio}_{d,w})^2 \cdot \max(0, \text{PPMI}(\alpha=0.95, k=1))$. Isolerer gjennomgående vokabular/kulisse som preger hele boka. Ideell for **sjanger og bred DDK-klasse**.
   - **$\mathbf{q}_{\text{PPMI}}$ (`ppmi`):** Klassisk Shifted PPMI ($\alpha=0.85, k=1$).
   - **$\mathbf{q}_{\text{hybrid}}$ (`hybrid`):** Produktet $\text{PPMI}_{d,w} \cdot \left(\max(0, \text{Fano}_{d,w} - 0.7) \cdot \max(0.1, 1.15 - \text{DF\_ratio}_{d,w})\right)^\beta$.

4. **Glissen SVD & Fold-in Projeksjon:**
   - Beregn **aldri** en tett $(N_{\text{bøker}} \times N_{\text{bøker}})$ Gram-matrise når $N_{\text{bøker}}$ kan være stor. Bruk direkte glissen matrise-vektor-multiplikasjon $u \leftarrow M(M^T u)$ i potensiterasjonen ($O(\text{nnz})$ per iterasjon, se [`computeWeightsAndSVD`](file:///Users/larsj/Github/doc-to-vec/app.js#L254-L420)).
   - En ny bok med glissen vektvektor $\mathbf{q}$ projiseres («foldes inn») i det $k$-dimensjonale SVD-rommet via:
     $$\mathbf{q}_{\text{SVD}} = \mathbf{q} \cdot \left(V_k \Sigma_k^{-1}\right)$$
     der radene i $V_k \Sigma_k^{-1}$ tilsvarer ord-basisen (`word_basis` i eksportformatet).

---

## 3. Dataskjema for `*_bundle.json` (CSR-format)

Alle korpuspakker (`mat_bundle.json`, `kane_bundle.json`, `tvedt_bundle.json`, `bbi_bundle.json`) følger dette eksakte JSON-skjemaet:

```json
{
  "meta": {
    "corpus_name": "Visningsnavn",
    "n_books": 338,
    "n_words": 5588,
    "nnz": 238645,
    "total_bg_sum": 15432198765.0
  },
  "books": [
    {
      "id": 0,
      "urn": "URN:NBN:no-nb_digibok_...",
      "title": "Tittel",
      "authors": "Forfatter",
      "year": 1995,
      "n_chunks": 142,
      "doc_tokens": 28450
    }
  ],
  "words": [
    {
      "w": "ord",
      "tf_bg": 12450,
      "nb": 42,
      "cf": 310,
      "md": 0.084,
      "mr": 0.712,
      "mf": 1.64,
      "stop": 0
    }
  ],
  "csr": {
    "row_ptr": [0, 612, 1180, "... (lengde n_books + 1)"],
    "col_idx": [0, 4, 9, "... (lengde nnz, indeks inn i words[])"],
    "freq": [12, 5, 8, "... (lengde nnz, rå TF i boka)"],
    "fano_x100": [185, 102, 340, "... (lengde nnz, round(Fano * 100))"],
    "dfratio_x100": [68, 98, 42, "... (lengde nnz, round((DF_chunk / E[DF]) * 100))"]
  }
}
```

---

## 4. Viktige NB DHlab API-detaljer («Gotchas»)

Når du skriver nye skript mot `https://api.nb.no/dhlab`, unngå disse fellene:

1. **`/dhlab/chunks` krever `GET` med query-parametre (IKKE `POST`):**
   ```python
   resp = requests.get("https://api.nb.no/dhlab/chunks", params={"urn": urn, "chunk_size": 250}, timeout=60)
   ```
   En `POST` mot `/dhlab/chunks` returnerer tom body (`JSONDecodeError`).
2. **`/dhlab/build_corpus` bruker nøkkelen `"ddk"` (IKKE `"ddc"`):**
   ```python
   resp = requests.post("https://api.nb.no/dhlab/build_corpus", json={"doctype": "digibok", "ddk": "641*", "from_year": 1950, "to_year": 2020, "limit": 200})
   ```
3. **`/dhlab/get_metadata` tar en liste med URN-er via `POST`:**
   ```python
   resp = requests.post("https://api.nb.no/dhlab/get_metadata", json={"urns": ["URN:NBN:no-nb_digibok_..."]})
   ```
   Returnerer en kolonne-orientert dict (`{"urn": {"0": ...}, "title": {"0": ...}, "authors": {"0": ...}, "year": {"0": ...}, "ddc": {"0": ...}, "subjects": {"0": ...}}`). Merk at `"ddc"` og `"subjects"` herfra kan brukes direkte som fasit-etiketter ved trening av DDK- og emneordsklassifikatorer!
4. **Kolonnenavn i `data/totals_500k.parquet`:**
   Kolonnene heter `["word", "total_freq"]` (ikke `"freq"`). Husk alltid å lowercase `word` og summere duplikater (`totals.groupby("word", as_index=False)["total_freq"].sum()`).

---

## 5. Slik gjenbruker du koden i et nytt repo

Hvis brukeren ber deg opprette et nytt repo (f.eks. for **DDK/emneord-prediksjon** eller **anbefalingsklustring**):
1. Kopier over `build_bundle_from_urns.py`, `dual_graph_clustering.py` og `data/totals_500k.parquet`.
2. For å hente treningsdata med fasit for DDK og emneord (`subjects`), kall `POST https://api.nb.no/dhlab/get_metadata` på URN-listen — den returnerer både `"ddc"` og `"subjects"` (semikolon-/skråstrek-separerte emneord) for hver bok.
3. Bygg de 4 partielle funksjonskanalene (`burst`, `frame`, `ppmi`, `hybrid`) + eventuelt Dual-Graph-galaksevektoren per bok slik at klassifikatoren kan vekte gjennomgående ramme ($\mathbf{q}_{\text{frame}}$) mot episodiske kapitteldrøftinger ($\mathbf{q}_{\text{burst}}$).
