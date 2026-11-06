# Doc-to-Vec · Eksplisitt PPMI, Intern Chunk-Burstiness & Dual-Graph Ordgalakser

**Interaktivt nettleser-verktøy:** [https://yoonsen.github.io/doc-to-vec/](https://yoonsen.github.io/doc-to-vec/)

Et transparent, ikke-nevrobasert rammeverk for dokument- og begrepsvektorisering bygget over Nasjonalbibliotekets DHlab-API (`https://api.nb.no/dhlab`). Verktøyet kombinerer makro-spesifisitet mot nasjonalbibliotekets referansekorpus (`nb.totals`) med mikro-spredning (**chunk-burstiness** over tekstblokker à 250–300 ord inne i hver enkelt bok), **Trunkert SVD** i sanntid og **overlappende Dual-Graph ordgalakser ($L(G)$)**.

---

## 1. Hvorfor ikke bare én vektor? (Multi-vektor & Tensor-perspektivet)

En bok har ikke én entydig trekkvektor. To bøker kan ligne på hverandre i **fortellerramme og sjanger**, men handle om helt ulike spesialtemaer — eller de kan dele ett **intenst drøftet undertema i et enkeltkapittel** på tvers av ulike sjangre.

I stedet for å tvinge hver bok ned i én ugjennomsiktig vektor, representerer `doc-to-vec` relasjonen mellom bok $d$ og ord $w$ som en **4-kanals partiell funksjon (tensor)**:

| Vektorkanal | Matematisk kjerne | Hva fanger vektoren? | Typisk anvendelse |
| :--- | :--- | :--- | :--- |
| **1. Burstiness ($\mathbf{q}_{\text{burst}}$)** | $\text{Fano} = \frac{\sigma^2}{\mu} \gg 1$ og lav $\frac{\text{DF}_{\text{chunk}}}{\mathbb{E}[\text{DF}_{\text{chunk}}]}$ | **Episodiske kapitler & dypdrøftinger:** Ord som klumper seg tett i bestemte avsnitt/kapitler i boka. Filtrerer automatisk bort metaforer, idiomer og tilfeldige nevnelser (som følger en Poisson-prosess med $\text{Fano} \approx 1$). | **Emneordsprediksjon (Subject Headings)**, finmasket temagjenfinning, sakprosa- og fagboksanalyse. |
| **2. Bærende ramme ($\mathbf{q}_{\text{frame}}$)** | Høy spredning ($\frac{\text{DF}_{\text{chunk}}}{\mathbb{E}[\text{DF}]} \approx 1$) $\times$ PPMI mot `nb.totals` | **Gjennomgående kulisse, stil & sjanger:** Ord som dukker opp jevnt gjennom hele boka, men som er høyspesifikke sammenlignet med norsk språk generelt. | Sjangerklassifisering, forfattersignatur, skille mellom f.eks. oppskriftsbøker, romaner og lærebøker. |
| **3. Ren PPMI ($\mathbf{q}_{\text{PPMI}}$)** | $\max\!\left(0, \log_2 \frac{P(w \mid d)}{P_{\text{bg}}(w)^\alpha} - \log_2 k\right)$ | **Makro-spesifisitet:** Hva som skiller bokas samlede ordforråd fra de 500 000 vanligste ordene i Nasjonalbiblioteket (`nb.totals`). | Klassisk overordnet emneklassifisering (**DDK / Dewey**) og vokabularsammenligning. |
| **4. Dual-Graph Galakser ($\mathbf{q}_{\text{galaxy}}$)** | Linjegraf-klustring $L(G)$ på $M^T M$ | **Overlappende temaklynger:** Grupperer ord i tolkbare temagalakser der flertydige ord (bro-ord) kan tilhøre flere galakser samtidig. | Forklarbar klustring av anbefalinger, tematisk profilering uten enkeltord-støy. |

> **Anvendelse på DDK (Dewey) og emneordsprediksjon (Subject Headings):**
> Tidligere forsøk på automatisk klassifikasjon med vanlige sekk-av-ord- eller TF-IDF-vektorer blander ofte sammen *gjennomgående sjangertrekk* og *spesifikke undertemaer*, samtidig som høyfrekvente metaforer (f.eks. *«i blinde»*, *«for døve ører»*) skaper falske positiver. Ved å skille $\mathbf{q}_{\text{frame}}$ + $\mathbf{q}_{\text{PPMI}}$ (som treffer bred DDK-klasse) fra $\mathbf{q}_{\text{burst}}$ (som isolerer reelle, lokaliserte drøftinger med $\text{Fano} > 1.3$ for spesifikke emneord) får en klassifikator langt renere signaler.

---

## 2. Store vs. små korpus — anbefalt arbeidsflyt

Verktøyet er bygget slik at **alle tunge matrise-, SVD- og grafberegninger kjører lokalt i nettleserens minne** via glissen potensiterasjon ($M(M^T u)$ i $O(\text{nnz})$ tid), mens selve tekstuthentingen fra Nasjonalbiblioteket kan gjøres enten direkte i nettleseren (små korpus) eller via et raskt Python-skript (store korpus).

### Oversikt: Når bruker du hva?

| Korpusstørrelse | Anbefalt byggemåte | Byggetid (NB API) | SVD & graf-beregning i nettleser |
| :--- | :--- | :--- | :--- |
| **Små korpus (20–250 bøker)** | Rett i nettleseren via **`Bygg fra NB API`** (søk eller lim inn URN-liste) | 10–45 sekunder | ~15–30 ms |
| **Mellomstore korpus (250–600 bøker)** | Nettleser (`Bygg fra NB API`) *eller* Python CLI (`build_bundle_from_urns.py`) | 45–90 sekunder | ~30–60 ms |
| **Store korpus (600–3 000+ bøker)** | Lokalt i Python med **`build_bundle_from_urns.py`** (12 parallelle tråder) $\rightarrow$ **`Last opp .json`** i nettleseren | 1–4 minutter i Python | ~60–150 ms |

### A. Små korpus (20–250 bøker): Rett i nettleseren
For et avgrenset studentprosjekt (f.eks. en liste på 50–200 anbefalte bøker, en forfatters bibliografi eller en bestemt DDK-undergruppe):
1. Åpne [https://yoonsen.github.io/doc-to-vec/](https://yoonsen.github.io/doc-to-vec/) og klikk **`Bygg fra NB API`**.
2. **Fra URN-liste:** Lim inn URN-er (eller klikk **`Last inn .csv / .txt med URN-er`**) i feltet øverst i dialogen. Verktøyet trekker automatisk ut alle `URN:NBN:no-nb_digibok_...` (eller `digibok_...`) og henter metadata og 250-ords tekstblokker.
3. **Fra metadata-søk:** Eller fyll inn forfatter, tittel, emneord eller DDK (f.eks. `641*`, `839*`) og tidsperiode.
4. Klikk **`Lagre .json`** i toppmenyen når korpuset er ferdig bygget. Da har studenten en ferdig `.json`-pakke på maskinen sin som senere kan åpnes øyeblikkelig med **`Last opp .json`** (eller ved å dra-og-slippe filen inn i nettleservinduet).

### B. Store korpus (300–3 000+ bøker): Bygg i Python, utforsk og eksporter i nettleseren
Når et korpus inneholder mange hundre eller flere tusen bøker, må det hentes 250-ords chunks for hver bok fra NB API. I stedet for å vente på dette i en nettleserfane, kjører man kommandolinjeskriptet `build_bundle_from_urns.py` som henter chunks med 12 parallelle tråder og komprimerer resultatet til en kompakt CSR `.json`-fil:

```bash
python build_bundle_from_urns.py student_urns.csv \
  --name "Anbefalingskorpus (Bachelor)" \
  --out anbefalinger_bundle.json \
  --chunk-size 250 \
  --max-words 5500 \
  --workers 12
```

* `student_urns.csv` kan være en hvilken som helst `.csv`, `.txt` eller `.json`-fil som inneholder URN-er (`URN:NBN:no-nb_digibok_...` eller `digibok_...`).
* En ferdig pakke på 300–400 bøker tar ca. **3–4 MB**, og selv et stort korpus på 1 800 bøker tar kun **~14 MB**.
* **Personvern og upubliserte forskningskorpus:** Når du åpner en lokal `.json`-pakke i nettleseren via **`Last opp .json`** (eller drag-and-drop), skjer all prosessering 100 % lokalt i nettleserens minne (`FileReader`). Ingen data fra filen sendes over nettet.

---

## 3. Vektoreksport for videre analyse i Python (`Eksporter vektorer`)

Når et korpus er lastet inn i nettleseren, kan studenten eller forskeren klikke på **`Eksporter vektorer`** i toppmenyen for å laste ned vektormodellen til videre bruk i Python, R eller andre språk.

Her står valget mellom to fundamentalt ulike måter å representere bøkene på:

### Representasjon 1: Partielle funksjoner (`.jsonl` uten nuller — ukomprimert ordrom)
I stedet for å tenke på en glissen vektor som en lang tuppel full av nuller, representeres hver bok som en **partiell funksjon** $f_d : V_d \to \mathbb{R}$ (et JSON-oppslagsverk `{"ord": vekt}`) der kun ordene som faktisk er virksomme i boka er med. Fravær av en nøkkel betyr $0$.
* **Fordel 1 (Full tolkbarhet):** Du ser nøyaktig hvilke ord som driver likheten mellom to bøker.
* **Fordel 2 (Åpent korpus / nye bøker utenfra):** Fordi dimensjonene er selve ordene (nøklene), er det trivielt å ta inn nye bøker utenfra og måle likhet mot korpuset uten å måtte re-estimere en SVD-modell.
* **To varianter i eksport-dialogen:**
  1. **Partiell funksjon — Aktiv vektkanal (`.jsonl`):** `{"ord": vekt}` for den vektmodusen du har stilt inn i kontrollpanelet (eller en hurtigknapp for *Ren Burstiness*, *Bærende ramme*, *Ren PPMI* eller *Hybrid*).
  2. **Partiell funksjon — 4-kanals Tensor (`.jsonl`):** `{"ord": [burst, frame, ppmi, hybrid]}` per bok, slik at du i Python kan eksperimentere fritt med hvilken kanal (eller kombinasjon av kanaler) som gir best klustring eller DDK/emneord-prediksjon.

#### Python-eksempel: Lese inn partielle funksjoner (`.jsonl`) for klustring eller DDK/emneord-prediksjon
```python
import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

books = []
partial_vectors = []

with open("korpus_partial_burst.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        books.append(row)
        # row["weights"] er den partielle funksjonen {"ord": vekt}
        partial_vectors.append(row["weights"])

# Konverter samlingen av partielle funksjoner til en glissen CSR-matrise i scikit-learn
vec = DictVectorizer(sparse=True)
X = normalize(vec.fit_transform(partial_vectors), norm="l2")

# Finn de nærmeste bøkene til bok nr. 0
sims = cosine_similarity(X[0], X).ravel()
top_idx = np.argsort(-sims)[1:6]
for i in top_idx:
    print(f"{sims[i]:.3f} | {books[i]['title']} ({books[i]['year']})")
```

Hvis du har eksportert **4-kanals tensoren (`_tensor4d.jsonl`)**, kan du i Python velge kanal (eller konkatenere dem for en klassifikator):
```python
# Kanal 0 = burst, 1 = frame, 2 = ppmi, 3 = hybrid
burst_only = [{w: vals[0] for w, vals in b["tensor"].items() if vals[0] > 0} for b in books]
frame_only = [{w: vals[1] for w, vals in b["tensor"].items() if vals[1] > 0} for b in books]
```

---

### Representasjon 2: Komprimerte, tette dimensjoner (`.csv` / `.json`)
Når alle bøkene som skal sammenlignes eller klustres allerede ligger inne i korpuset (et **lukket korpus**), er det ofte praktisk å komprimere ordrommet ned til en vanlig tabell uten nuller:

1. **Komprimert SVD-matrise (`N × 10 .csv`):**
   Hver bok får 10 ortogonale, L2-normaliserte koordinater (`svd_1` … `svd_10`) fra den trunkerte SVD-faktoriseringen $U_k \Sigma_k^{1/2}$. Klar til å leses rett inn med `pd.read_csv()` for `KMeans`, `AgglomerativeClustering`, `UMAP` eller `NearestNeighbors`.
2. **Tolkbare Dual-Graph Temagalakser (`N × C .csv`):**
   Hver kolonne er ikke en abstrakt SVD-akse, men en **konkret, navngitt ordgalakse** fra linjegraf-klustringen $L(G)$ (f.eks. `k01_baking_gjær_hvetemel_deig`, `k02_vin_rødvin_hvitvin_ost`). Verdien i cellen angir hvor sterkt boka aktiverer den aktuelle temagalaksen.
3. **SVD-modell + Partiell Ord-basis (`.json` for fold-in av nye bøker):**
   Hvis man bruker komprimerte SVD-dimensjoner, men *likevel* vil kunne projisere nye bøker utenfra inn i det samme 10-dimensjonale rommet i Python, inneholder denne JSON-filen både bøkenes SVD-vektorer og projeksjonsmatrisen $V_k \Sigma_k^{-1}$ lagret som en partiell funksjon `word_basis[ord] = [v1, ..., v10]`.

#### Python-eksempel: Projisere («folde inn») en ny bok utenfra i det komprimerte SVD-rommet
```python
import json
import numpy as np

with open("korpus_svd_model.json", encoding="utf-8") as f:
    model = json.load(f)

word_basis = model["word_basis"]  # Partiell funksjon: ord -> 10D projeksjonsvektor
corpus_vecs = np.array([b["svd_norm"] for b in model["books"]])

# Anta at vi har en ny bok representert som en partiell funksjon new_book_weights = {"ord": vekt}
def project_new_book(new_book_weights: dict[str, float]) -> np.ndarray:
    q = np.zeros(model["k_dimensions"], dtype=float)
    for word, weight in new_book_weights.items():
        if word in word_basis:
            q += weight * np.asarray(word_basis[word], dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q
```

---

## 4. Sjekk enkeltbok mot korpuset (Fane 5: Multi-vektor dekomponering)

I fane **`5. Sjekk enkeltbok (Multi-vektor)`** i nettleseren kan du lime inn en hvilken som helst URN fra Nasjonalbiblioteket (enten den ligger i det aktive korpuset eller hentes direkte utenfra via NB API):
* Boka deles i 250-ords blokker i sanntid og dekomponeres i de fire trekkvektorene ($\mathbf{q}_{\text{burst}}$, $\mathbf{q}_{\text{frame}}$, $\mathbf{q}_{\text{PPMI}}$ og **Dual-Graph temagalakser**).
* Du ser side-ved-side hvilke bøker i korpuset som ligger nærmest langs hver av de 4 vektorene, og hvilke delte ord som driver hvert treff.
* Boka projiseres samtidig inn i korpusets SVD-rom ($q_{\text{SVD}} = q \cdot V_k \Sigma_k^{-1}$) og tegnes som en **gul stjerne (★)** i 2D-bokkartet.

---

## 5. Filer i repoet

* `index.html`, `app.js`, `style.css` — Frittstående nettleser-applikasjon (ingen byggesteg eller eksterne rammeverk nødvendig; kjører direkte på GitHub Pages eller via `python -m http.server`).
* `mat_bundle.json` — Demo-korpus 1: **Norske kokebøker & matkultur (DDK 641, 1840–2022)** (338 bøker, 5 588 ord).
* `kane_bundle.json` — Demo-korpus 2: **Morgan Kane / Louis Masterson (1965–2015)** (255 bøker, 5 287 ord).
* `build_bundle_from_urns.py` — Kommandolinjeskript for å bygge en `.json`-korpuspakke raskt fra en `.csv`- eller `.txt`-fil med URN-er.
* `build_mat_bundle.py`, `build_kane_bundle.py` — Skriptene som bygger de to offentlige demokorpusene fra DHlab API.
* `dual_graph_clustering.py`, `burstiness_doc2vec_and_graphs.py` — Referanseimplementasjoner i Python for chunk-burstiness, PPMI-SVD og Adamic–Adar-vektet linjegraf-klustring $L(G)$.
