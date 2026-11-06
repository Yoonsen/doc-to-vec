# Dokumentlikhet: sparse CPU vs. GPU (sparse, tett, projisert, hybrid)

Spørsmål: hjelper GPU-en på dokumentlikhet i doc-to-vec-stil (bok × ord-frekvensvektorer), og hva koster det i *ettersporbarhet*, altså muligheten til å dekomponere en likhet eller divergens i eksakte bidrag per ord (`explain_dot`, `explain_jsd_divergence` i `partial-functions`)?

Kort svar: **ja, GPU-en hjelper mye, men bare når den regner på den glisne (sparse) matrisen eller brukes som kandidatfilter.** Per spørring er eksakt `torch.sparse` på GPU 10–20× raskere enn Numba på 40 CPU-tråder for cosinus og Hellinger, ≈ 3× raskere for JSD, og 400–600× raskere enn `PartialFunc`-løkka i Python. Tette vektorer over et avkortet vokabular og projeksjoner til 256–1 024 dimensjoner er enda raskere, men mister mange av de riktige naboene. Den beste kombinasjonen er hybriden: projisert GPU-søk finner kandidater, og eksakt `PartialFunc` rangerer dem på nytt og forklarer resultatet.

Skript: [bench_similarity_gpu.py](bench_similarity_gpu.py), [bench_pairwise_dot.py](bench_pairwise_dot.py) og [extract_shard_counts.py](extract_shard_counts.py). Rådata (JSONL) og fullstendige tabeller ligger i [bench_results/](bench_results/): [tables_similarity.md](bench_results/tables_similarity.md) og [tables_pairwise.md](bench_results/tables_pairwise.md).

---

## 1. Oppsett

**Maskinvare:** én arbeidsstasjon med NVIDIA RTX A6000 (48 GB), 2× Intel Xeon Silver 4210R (40 tråder) og 125 GB RAM, der ≈ 47 GB var ledig under kjøringen. Programvare: Python 3.10, torch 2.14 (CUDA 12.6), numba 0.68, scipy 1.15, numpy 2.2 og pyroaring 1.2.

**Data** (bare tellinger og heltalls-ID-er, ingen tekst):

| datasett | bøker | vokabular | nnz | distinkte ord/bok |
|---|---|---|---|---|
| `mat` (DDK 641, `mat_bundle.json`) | 338 | 5 588 | 0,24 M | 706 |
| `synthetic_1000` | 1 000 | 500 000 | 15,5 M | 15 527 |
| `synthetic_10000` | 10 000 | 500 000 | 155 M | 15 474 |
| `shard_8969` (ekte tellinger, shard 0) | 8 969 | 11 812 683 | 77,6 M | 8 650 |
| `synthetic_100000` | 100 000 | 500 000 | 1 548 M | 15 483 |

Syntetiske bøker trekkes som en blanding av en bakgrunnsfordeling (de faktiske frekvensene i `totals_500k.parquet`) og ett primærtema pluss ett nabotema. Det er ett tema per ≈ 50 bøker, så hver bok har flere sanne naboer enn k = 10. Bøkene har lognormal lengde med median 60k tokens. Shard 0 gir ekte tellinger fra `unigrams(book_id, cf_id, tf)`, med ord-ID-er omnummerert etter fallende frekvens. Vokabularet på 11,8 M består for det meste av OCR-støy.

**Mål:**
* `cos_tf`: cosinus på rå tellinger.
* `cos_ppmi`: cosinus på Shifted PPMI med α = 0,85 og k = 1 mot korpusets bakgrunnsfordeling, slik `ppmi`-kanalen i doc-to-vec er definert.
* `hellinger`: $H = \sqrt{1 - \sum_{I}\sqrt{pq}}$. Dette er et prikkprodukt på $\sqrt{p}$, så det kan regnes som GEMM.
* `jsd`: JSD i bits over snittet $I$. Dette er ikke et prikkprodukt og kan ikke regnes som GEMM.

**Kvalitet:** recall@10 for 100 tilfeldige spørrebøker, målt mot eksakt topp-10. Fasiten regnes over hele vokabularet med fp64-akkumulering i Numba, og spørreboka selv er utelatt. `PartialFunc` og `torch.sparse` gir samme topp-10 som fasiten. Avviket i skårer mellom `PartialFunc` og Numba er ≤ 1e-15.

**Tider merket «(est)»** er ekstrapolert lineært fra et tidsbudsjett på 30–60 s per måling, fordi full kjøring ville tatt timer eller dager. Alle-mot-alle betyr topp-10 for hver bok, altså N² sammenligninger.

---

## 2. Hovedtall

### 2.1 Én-mot-alle, per spørring (eksakt)

| backend | 10k cos_ppmi | 10k jsd | shard 0 cos_ppmi | shard 0 jsd | 100k cos_ppmi | 100k jsd |
|---|---|---|---|---|---|---|
| `PartialFunc` + `SparseWorkspace` (1 tråd, Python-løkke) | 633 ms | 2 133 ms | 668 ms | 692 ms | 3,9 s | 19,7 s |
| scipy CSR (1 tråd; JSD numpy-vektorisert) | 152 ms | 5 242 ms | 223 ms | 1 789 ms | 1,5 s | – |
| Numba CSR + scatter (40 tråder) | 14 ms | 88 ms | 35 ms | 52 ms | 126 ms | 827 ms |
| **`torch.sparse` CSR, GPU** | **1,1 ms** | **33 ms** | **1,8 ms** | **18 ms** | **8,9 ms** | **332 ms** |
| tett 50k fp16, GPU (ikke eksakt) | 1,5 ms | 79 ms | 1,3 ms | 70 ms | 16 ms | 787 ms |

### 2.2 Alle-mot-alle topp-10 og kvalitet (100k bøker)

| metode | cos_ppmi tid | recall@10 | jsd tid | recall@10 | ettersporbar? |
|---|---|---|---|---|---|
| `PartialFunc`, 1 tråd | 109 h (est) | 1,000 | 549 h (est) | 1,000 | ja, fullt |
| Numba CSR, 40 tråder | 210 min (est) | 1,000 | 23 h (est) | 1,000 | ja, data er de samme |
| `torch.sparse`, GPU | 5,2 min (est) | 1,000 | 9,2 h (est) | 1,000 | ja, eksakt |
| tett 50k fp32 / fp16 | 58 s / 8,7 s | 0,69 | 20,6 h / 21,9 h (est) | 0,76 | nei, halen av vokabularet mangler |
| projeksjon rp1024 (fp16) | 0,36 s | 0,80 | 0,36 s (Hellinger-rom) | 0,85 | nei |
| projeksjon svd256 | 0,24 s | 0,66 | 0,24 s | 0,49 | nei |
| **hybrid svd256 → `PartialFunc`, C = 50** | **3,6 min**¹ | **0,982** | 20 min¹ | 0,826 | **ja, på topp-k** |
| **hybrid rp256 → `PartialFunc`, C = 50** | 3,6 min¹ | 0,929 | **22 min**¹ | **0,959** | **ja, på topp-k** |
| hybrid rp256, C = 200 | 14 min¹ | 0,980 | 95 min¹ | 0,983 | ja, på topp-k |

¹ Tiden gjelder GPU-kandidater pluss omrangering, med `PartialFunc`-objektene allerede bygget. Bygger man dem fra CSR for hver spørring, kommer det 20–160 ms i tillegg per spørring. Da blir tidene 35–350 min (se de fulle tabellene). Omrangeringen kjører i én Python-tråd.

### 2.3 Kvalitet på ekte data (shard 0, 8 969 bøker)

| metode | cos_tf | cos_ppmi | hellinger | jsd |
|---|---|---|---|---|
| tett 50k fp32 | 0,977 | 0,632 | 0,822 | 0,822 |
| tett 50k fp16 | 0,968 | 0,631 | 0,819 | 0,822 |
| rp256 / rp1024 | – | 0,10 / 0,23 | 0,55 / 0,75 | 0,55 / 0,74 |
| svd256 / svd1024 | – | 0,50 / 0,54 | 0,54 / 0,64 | 0,57 / 0,66 |
| hybrid svd256, C = 50 / 200 | – | 0,817 / **0,949** | 0,867 / **0,978** | 0,890 / **0,981** |
| hybrid rp256, C = 50 / 200 | – | 0,161 / 0,251 | 0,843 / 0,943 | 0,842 / 0,938 |

Én hybridspørring tar 6–30 ms. Forklaringen av topp-10 med `explain_dot` eller `explain_jsd_divergence` og `top_k(10)` tar 2–19 ms per spørring.

### 2.4 Parvise prikkprodukter: Roaring-snitt vs. matrisemultiplikasjon

[bench_pairwise_dot.py](bench_pairwise_dot.py) måler $\langle f, g\rangle$ via $I = \text{dom}(f) \cap \text{dom}(g)$ og $f[I] \cdot g[I]$, på alle måtene `partial_func.py` gjør det, og sammenligner med `A @ A.T`. Verdiene er L2-normaliserte tellinger. Alle metodene gir samme svar som float64-referansen: avviket er ≤ 2e-13 for CPU-metodene, ≤ 1e-5 for GPU fp32 og ≤ 6e-3 for GPU fp16.

| metode (100k syntetisk, \|dom\| ≈ 15,6k, \|I\| ≈ 4,7k) | tid per par | alle-mot-alle (N²) |
|---|---|---|
| `align_intersection(method="rank")`, `dom.rank(w)-1` per $w \in I$ | 6 760 µs | 783 døgn (est) |
| `PartialFunc.dot` (Roaring `&` + gallop) | 258 µs | 30 døgn (est) |
| `align_intersection` gallop / twopointer | 250 / 313 µs | 29 / 36 døgn (est) |
| `_bitblock_intersect_dot_and_jsd` (POPCNT), per par | 217 µs | 25 døgn (est) |
| `SparseWorkspace.dot_with_active` / `dot_and_jsd_with_active` | 166 / 197 µs | 19 / 23 døgn (est) |
| Numba batch, flettesnitt (40 tråder) | 6,5 µs | 18 h (est) |
| Numba batch, bitblokk + POPCNT (40 tråder) | 2,2 µs | 6,2 h (est) |
| Numba batch, workspace/Gustavson (40 tråder) | 2,4 µs | 6,6 h (est) |
| scipy `A[blokk] @ A.T` (1 tråd, 10k) | 9,8 µs | – |
| **`torch.sparse` SpMM (cuSPARSE)** | **0,073 µs** | **12 min (est)** |
| tett GEMM fp32 / fp16 (fullt vokabular, 10k bøker) | 0,064 / 0,025 µs | 6,4 s / 2,5 s (10k) |

Funn:
* **`rank` er 25× tregere enn de andre snittmetodene.** Det koster å slå opp `rank(w)` i Python for hvert element i $I$. Metoden bør bare brukes for enkeltoppslag.
* Med 15k ord per bok er det ikke Python-overheaden som dominerer. Arbeidet per par er O(|f| + |g|) eller O(|g|), som er ~100–200 µs effektiv CPU-tid uansett metode. Batchet Numba med 40 tråder gir ≈ 2,2 µs per par, eller ≈ 90 µs CPU-tid per par. Det er rent minnebåndbredde. På DDK 641, med 706 ord per bok, er Python-overheaden en større andel (`ws_dot` tar 5,7 µs per par mot 0,28 µs i Numba-batch).
* **Bitblokk med POPCNT er raskest på CPU når kallene batches** (2,2 µs). Som enkeltkall fra Python er den bare litt raskere enn gallop.
* **GPU-SpMM er ≈ 30× raskere enn den beste CPU-kjernen og ≈ 3 000× raskere enn ett Python-kall per par.** Tett GEMM over hele vokabularet er raskest der den får plass, opptil ≈ 10k × 500k i fp32. Shard 0 trenger 395 GB i tett format og 100k-korpuset trenger 186 GB, så der er den uaktuell.

### 2.5 PMI-transformasjon (hele korpuset)

| backend | 10k | shard 0 | 100k |
|---|---|---|---|
| `PartialFunc.pmi` per bok (est) | 15,7 s | 14,6 s | 152 s |
| numpy/scipy vektorisert | 6,9 s | 3,8 s | – (for mye minne) |
| Numba CSR, 40 tråder | 0,32 s | 0,37 s | 3,6 s |
| `torch.sparse`, GPU | 0,035 s | 0,019 s | 0,35 s |

---

## 3. Tolkning

1. **Eksakt likhet:** `torch.sparse` på GPU slår Numba med 40 tråder på alle skalaer: 10–20× for cosinus og Hellinger, men bare ≈ 3× for JSD. Prikkproduktmålene (cosinus, Hellinger) går gjennom cuSPARSE SpMM. JSD regnes med gather og `segment_reduce` over CSR-verdiene. Det er O(nnz) per spørring og derfor dyrere: JSD alle-mot-alle på 100k tar ≈ 9 timer selv på GPU. 100k bøker med 15k distinkte ord (1,55 G nnz) får plass i 12 GB GPU-minne (fp32 + int32).
2. **Tett GPU over de 50k vanligste ordene er en dårlig avveining.** GEMM-en er rask, men recall@10 faller til 0,48–0,69 for PPMI-cosinus, fordi PPMI-vekten ligger i den sjeldne halen som kuttes bort. For JSD er tett ikke engang raskere enn sparse GPU. Tett format fungerer bare for `cos_tf` på ekte data (0,98), og det målet domineres av funksjonsord.
3. **Projeksjoner alene er ikke gode nok som svar.** SVD-256 og RP-1024 gir recall 0,47–0,85. Sparse-JL (hashing over hele vokabularet) holder seg bedre enn SVD når N vokser. SVD er tilpasset ≤ 20k bøker og et vokabular kuttet ved 200k, og falt fra 0,73 til 0,49 for JSD fra 10k til 100k. På ekte PPMI-data er RP svak (0,10–0,25), fordi OCR-støy med høy PPMI havner i de samme hash-bøttene.
4. **Hybriden henter tilbake kvaliteten** til 0,94–1,00 med C = 200 kandidater (unntak: RP som generator for PPMI på ekte data, 0,25), til en brøkdel av prisen for eksakt GPU-JSD. På 100k tar hybrid JSD (rp256, C = 50) 22 min med 0,96 recall, mens eksakt `torch.sparse` tar ≈ 9 t. Omrangeringen er her én Python-tråd. Batchet i Numba over kandidatradene ville den bli rundt 70× raskere (se 2.4) uten å endre tallene.
5. **`PartialFunc` er ikke en masseberegningsmotor.** Den er 2 000–3 000× tregere enn GPU-SpMM per par og 1,1–1,4× større i minnet enn CSR (`float64` + Roaring mot `float32` + `int32`). Verdien ligger i den eksakte, ettersporbare dekomponeringen på de få parene som betyr noe. Den koster 0,1–2 ms per forklart par.

---

## 4. Anbefaling: når vinner hva?

| situasjon | velg | hvorfor |
|---|---|---|
| < ~2k bøker, eller én spørring om gangen interaktivt | sparse CPU (Numba CSR eller `PartialFunc` + `SparseWorkspace`) | under 0,1 s, ingen GPU-overføring, full ettersporbarhet |
| eksakt én-mot-alle eller alle-mot-alle med cosinus, PPMI eller Hellinger opptil ~100k bøker | **sparse GPU** (`torch.sparse` CSR) | eksakt; 9 ms per spørring og 5 min alle-mot-alle (PPMI) på 100k; nnz må få plass på GPU (≈ 8 B per nnz) |
| alle-mot-alle JSD eller nabograf over > 10k bøker | **hybrid**: projisert GPU (rp256/svd256), deretter eksakt `PartialFunc`/Numba-omrangering (C = 50–200) | 0,94–0,98 recall (C = 200) eller 0,96 (rp256, C = 50) til 1/25–1/6 av eksakt GPU-tid; topp-k er eksakte og forklarbare |
| PPMI-cosinus med ekte, støyfylt vokabular | hybrid med **SVD** som kandidatgenerator (ikke RP) og C ≥ 200 | RP-hashing blander OCR-støy (0,25 recall); SVD gir 0,95 |
| visualisering, klynging, «grovt nabolag» | ren projeksjon (svd/rp 256–1024 i fp16) | 50–200 MB for 100k bøker; alle-mot-alle under 1 s; recall 0,47–0,85 er greit når forklaring per ord ikke trengs |
| tett GPU over avkortet vokabular | **unngå** for PPMI og JSD | treg for JSD, taper halen; bare greit for `cos_tf` |
| 700k bøker | ikke kjørt; se under | |

**700k (estimat, ikke målt):** Med 15k distinkte ord per bok blir det ≈ 10,8 G nnz, eller 87 GB i CSR (fp32 + int32). Det får verken plass i ledig RAM (≈ 47 GB) eller i GPU-minnet (48 GB). Eksakt søk må derfor skje i shards på ≈ 100k bøker som strømmes gjennom GPU-en. Det blir ≈ 7 × 7 blokkpar à ~5 min for PPMI-cosinus, ≈ 4 t alle-mot-alle. En projisert indeks for 700k bøker tar 0,36 GB (rp256, fp16) eller 1,4 GB (rp1024) og får plass på GPU-en i sin helhet. Alle-mot-alle-GEMM-en der skalerer med N² og blir ≈ 50× tiden for 100k, altså ~20 s. Hybriden skalerer derfor best: projisert indeks på GPU, og eksakt omrangering der `PartialFunc`/CSR leses fra shard-lageret bare for kandidatene.

### Hva mister man av ettersporbarhet?

* **Sparse CPU og sparse GPU:** ingenting. Det er den samme partielle funksjonen og det samme snittet $I$, bare en annen motor. GPU-skåren kan når som helst dekomponeres med `PartialFunc.explain_*` for samme par. Avviket fra float64 er ~1e-5 i fp32.
* **Tett, avkortet vokabular:** skårene er definert på en annen funksjon, nemlig $f$ restriktert til de K vanligste ordene og renormalisert. En forklaring dekker bare ordene innenfor K, og de sjeldne ordene som bærer PPMI-likheten mangler helt. Dette er tap av både kvalitet og forklarbarhet.
* **Projeksjoner (SVD/RP):** hver koordinat er en blanding av tusenvis av ord, så likheten kan ikke dekomponeres per ord. Med SVD kan man i beste fall forklare via `word_basis` (Vₖ). Med RP er det umulig, fordi hashing blander ord vilkårlig.
* **Hybrid:** selve rangeringen og alle skårer i topp-k er eksakte og fullt dekomponerbare. Det man mister, er en garanti for **fullstendighet**: en sann nabo som projeksjonen ikke fant blant de C kandidatene, kommer aldri med (recall < 1). Dette kan måles med recall@k mot eksakt GPU på et utvalg, slik som her, og styres med C.

---

## 5. Begrensninger

* Tider merket «(est)» er ekstrapolert lineært fra tidsbudsjetterte delmålinger. GPU-SpMM og tett GEMM er kjørt fullt der det står uten «(est)».
* Omrangeringen i hybriden og `PartialFunc`-løkkene kjører i én Python-tråd. Numba med 40 tråder er målt separat (2.4), men ikke koblet inn i hybriden.
* Den syntetiske temastrukturen bestemmer de absolutte recall-tallene. Shard 0 er det beste estimatet for ekte data, men det er bare én shard og ett korpus.
* SVD er tilpasset på ≤ 20k bøker med vokabularet kuttet ved 200k ord. Mer data i tilpasningen ville trolig gitt bedre resultat på 100k.
* Minnetallene er representasjonsstørrelse (CPU) og `torch.cuda.max_memory_allocated` per operasjon (GPU). Prosessens maks-RSS var 41 GB på 100k.

## 6. Reproduksjon

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "numpy>=2.0" scipy pyroaring numba pandas pyarrow
export PARTIAL_FUNCTIONS_DIR=../partial-functions   # mappen med partial_func.py
python bench_similarity_gpu.py --source mat --out bench_results/mat.jsonl
python bench_similarity_gpu.py --source synthetic --n-books 10000 --out bench_results/syn_10k.jsonl
python extract_shard_counts.py PATH/TO/shard.db data/shard0_counts.npz   # skrivebeskyttet, kun tellinger
python bench_similarity_gpu.py --source shard --shard-npz data/shard0_counts.npz --n-books 0 --out bench_results/shard0.jsonl
python bench_pairwise_dot.py --source synthetic --n-books 10000 --out bench_results/pw_syn_10k.jsonl
python bench_similarity_gpu.py --report bench_results/{mat,syn_1k,syn_10k,shard0,syn_100k}.jsonl
python bench_pairwise_dot.py  --report bench_results/pw_*.jsonl
```

Med `NUMBA_THREADING_LAYER=omp` unngår man TBB-advarselen på eldre TBB. Kjøringen på 100k tok 32 min og trengte ≈ 41 GB RAM og ≤ 28 GB GPU-minne.
