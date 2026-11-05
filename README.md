# Doc2Vec · PPMI, Chunk-Burstiness & Dual-Graph Ordgalakser

Eksperimentelt rammeverk og interaktivt nettleser-dashboard for dokument- og begrepsvektorisering (statisk `doc2vec` via eksplisitt PPMI, intern chunk-burstiness og Truncated SVD), kombinert med ord-mot-ord dot-produkter og overlappende linjegraf-klustring ($L(G)$).

## Hovedidéer

1. **Eksplisitt PPMI + SVD (Levy & Goldberg):**
   Henter ordtellinger per bok med `cutoff >= 5` fra DHlab-API-et (`https://api.nb.no/dhlab/frequencies`) og vaskes mot topp 500 000 ord i `nb.totals`. Ved å faktorisere den glisne PPMI-matrisen $M \approx U_k \Sigma_k V_k^T$ får vi tette, globalt optimale dokumentvektorer ($U_k \Sigma_k^{1/2}$) og ordvektorer ($V_k \Sigma_k^{1/2}$) i samme rom — med full sporbarhet.
2. **Intern Chunk-Burstiness (uten `nb.totals`):**
   Ved å dele hver bok i chunks à 300 ord (`/chunks?chunk_size=300`) blir hver bok sitt eget referansekorpus. Funksjonsord (uansett om boka er på 1780-talls dansk, 1920-talls landsmål eller moderne bokmål) får lav intern PMI og Fano-faktor $\sigma^2/\mu \approx 1$, mens reelle innholdsord og drøftede temaer slår ut med høy burstiness ($\text{Fano} \gg 1$, lav $\text{DF}/\mathbb{E}[\text{DF}]$).
3. **Sanntids-SVD i nettleseren ($G = M M^T$):**
   Fordi $N_{\text{bøker}} \ll N_{\text{ord}}$, beregnes egenvektorene $U_k$ direkte på $(N_{\text{bøker}} \times N_{\text{bøker}})$-Gram-matrisen $G = M M^T$ i JavaScript (`Float32Array`/`Float64Array`) på ~30 ms, og ordvektorene projiseres som $V_k = M^T U_k \Sigma_k^{-1}$.
4. **Ord-mot-ord Galakse & Dual-Graph $L(G)$:**
   Bygger ord-mot-ord-graf fra kolonne-dot-produktet ($M^T M$ eller chunk-matrisen $C^T C$) og finner overlappende tematiske klynger og bro-ord via Louvain/Leiden + Adamic–Adar-vektet linjegraf $L(G)$.

## Filer i repoet

* `index.html`, `app.js`, `style.css` – Frittstående nettleser-dashboard (fungerer også direkte på GitHub Pages med forhåndslastet `tvedt_bundle.json`, eller mot DHlab-API-et via CORS).
* `tvedt_bundle.json` – Ferdigpakket CSR-datasett for Terje Tvedt-korpuset (`vann.csv`, 352 historieverk).
* `fetch_data.py` – Henter `totals(500_000)` og bokfrekvenser (`cutoff=5`) fra DHlab-API-et.
* `pmi_doc2vec.py` – Bygger PPMI-matrisen (med lowercasing og $\alpha$-utjevning) og beregner SVD-vektorer.
* `inspect_axes.py` – Inspiserer $(+)$- og $(-)$-polene på SVD-aksene.
* `fetch_chunks_and_burstiness.py` – Henter 300-ords chunks for hele korpuset og beregner entropi, intern PMI, $\text{DF}/\mathbb{E}[\text{DF}]$ og Fano-faktor per ord i hver bok.
* `burstiness_doc2vec_and_graphs.py` – Kjører burstiness-SVD, «Norge vs. verden»-analyse og ord-mot-ord dual-graph-klustring i Python.
* `export_web_bundle.py` – Eksporterer Parquet-dataene til `tvedt_bundle.json` for web-appen.
