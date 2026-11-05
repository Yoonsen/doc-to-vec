# Mention vs. Thematization: Double Burstiness and Dual-Graph Spectral Geometry for Transparent Topic Discovery in Cultural History and Children’s Literature

**Target Venue:** DH2027 (ADHO Digital Humanities Conference 2027) — Long Paper / Methodological & Empirical Contribution  
**Authors:** Lars G. Johnsen et al. (National Library of Norway / DH-lab)  
**Interactive Companion & Code:** `https://yoonsen.github.io/doc-to-vec/` (`https://github.com/Yoonsen/doc-to-vec`)

---

## 1. Introduction: Between Forgotten LSA and Opaque Neural Embeddings

For two decades, computational literary and historical studies have oscillated between two poles of text representation. On one side lie classical count-based models—Latent Semantic Analysis (LSA; Deerwester et al., 1990) and Latent Dirichlet Allocation (LDA; Blei et al., 2003)—which offer global mathematical structure but treat entire books as flat bags of words and rely heavily on brittle, language-specific stopword lists. On the other side lie dense neural embeddings (`word2vec`, `doc2vec`, and Transformer encoders) and LLM-based zero-shot classifiers, which capture rich local semantics at the cost of corpus-level transparency, reproducibility, and linear decomposability.

Although Levy and Goldberg (2014) famously demonstrated that Skip-Gram with Negative Sampling (`word2vec`) implicitly factorizes a shifted Positive Pointwise Mutual Information (PPMI) matrix, Digital Humanities has largely abandoned explicit matrix factorization in favor of neural black boxes or stochastic topic models. In this paper, we argue that returning to explicit linear algebra—augmented by **two-level lexical burstiness ("Double Burstiness")** and **Dual-Graph ($L(G)$) overlapping clustering**—solves three longstanding methodological problems in distant reading:

1. **Multi-century and multi-norm stopword elimination without word lists:** By treating each book as its own reference corpus of fixed-length chunks ($250\text{–}300$ words), internal Shannon entropy and the Fano index ($\sigma^2/\mu$) automatically separate grammatical glue from thematic content across centuries and orthographic norms.
2. **Operationalizing "Mention vs. Thematization":** Raw document frequency conflates casual passing mentions, spatial backdrops, and idioms with sustained narrative or analytical engagement. Combining macro-level corpus specificity with micro-level within-book chunk burstiness separates incidental lexical occurrence from genuine thematic concentration.
3. **Polysemy and multi-membership in word networks:** Standard graph clustering forces words into disjoint partitions. Clustering the Adamic–Adar weighted line graph $L(G)$ of chunk co-occurrences ($C^T C$) allows polysemous terms and thematic bridge concepts to belong to multiple overlapping semantic galaxies while pruning high-entropy hub clusters.

We validate the method across two contrasting macro-corpora from the National Library of Norway (NB DH-lab), totaling over $450,000$ text chunks ($>120$ million tokens), and cross-validate our purely distributional burstiness metrics against $6,137$ LLM-annotated concordance fragments (*luminons*).

---

## 2. Mathematical Framework: Double Burstiness and Dual-Graph SVD

### 2.1 Level 1 (Macro): Shifted PPMI Against a National Reference Corpus
Given a target corpus of $D$ books and an external reference baseline $P_{\text{ref}}(w)$ derived from the top $500,000$ word forms in the National Library of Norway (`nb.totals`), we define the $\alpha$-smoothed document-level association weight for word $w$ in book $d$ as:

$$\text{PPMI}_\alpha(d, w) = \max\!\left(0,\; \log_2 \frac{P(w \mid d)}{P_{\text{ref}}(w)^\alpha + \epsilon}\right), \qquad M_{d,w} = \log_2(1 + f_{d,w}) \cdot \text{PPMI}_\alpha(d, w)$$

where $\alpha = 0.75$ dampens extreme rare-word leverage (Levy et al., 2015) and rows of $M$ are $L_2$-normalized prior to Truncated SVD ($M \approx U_k \Sigma_k V_k^T$).

### 2.2 Level 2 (Micro): Within-Book Chunk Burstiness and Poisson Deviation
A book $d$ of length $N_d$ is partitioned into $K_d$ contiguous chunks of $250\text{–}300$ words (approximating one printed page). Let $c_{d,k}(w)$ denote the count of word $w$ in chunk $k \in \{1,\dots,K_d\}$, with book total $F_{d,w} = \sum_k c_{d,k}(w)$, chunk mean $\mu_{d,w} = F_{d,w}/K_d$, and chunk document frequency $\text{DF}_{\text{chunk}}(d,w) = \sum_k \mathbb{I}[c_{d,k}(w) > 0]$.

Under a homogeneous Poisson null model (Church & Gale, 1995; Katz, 1996; Pierrehumbert, 2012), a word distributed randomly across the book has expected chunk dispersion $\mathbb{E}[\text{DF}_{\text{chunk}}] = K_d(1 - e^{-\mu_{d,w}})$ and Fano factor:

$$\text{Fano}(d,w) = \frac{\sigma_{d,w}^2}{\mu_{d,w}} \approx 1.0, \qquad \frac{\text{DF}_{\text{chunk}}(d,w)}{\mathbb{E}[\text{DF}_{\text{chunk}}(d,w)]} \approx 1.0$$

Conversely, a word that clusters inside specific chapters, scenes, or analytical sections exhibits $\text{Fano}(d,w) \gg 1$ and $\text{DF}_{\text{chunk}} / \mathbb{E}[\text{DF}_{\text{chunk}}] \ll 1$, and carries high internal information gain $\overline{\text{PMI}}_{\text{intern}}(d,w) = \log_2 K_d - H(w \mid d)$, where $H(w \mid d)$ is Shannon entropy across the book’s chunks.

### 2.3 The 2×2 Double Burstiness Taxonomy
Combining Macro document spread ($\text{DF}_{\text{book}}/D$ or external $\text{PPMI}_\alpha$) with Micro within-book burstiness ($\text{median Fano}_w$) partitions any corpus vocabulary into four interpretable quadrants:

| Quadrant | Macro Spread across Books | Micro Burstiness inside Books ($\text{Fano}$) | Linguistic & Literary Function |
| :--- | :--- | :--- | :--- |
| **Q1. Structural Glue** | High ($\text{DF}_{\text{book}} \gg 0$) | Low ($\text{Fano} \approx 1.0$) | Grammatical function words across all historical/orthographic norms (`og`, `havde`, `ikkje`) & narrative glue (`men`, `bare`, `også`). |
| **Q2. Ambient Frame / Idiom** | Low–Medium | Low ($\text{Fano} \approx 1.0$) | Unreflected background frames (`norge`, `europa`), spatial landmarks, and dead metaphors (`i blinde`, `et skeivt smil`). |
| **Q3. Episodic Archetypes** | High | High ($\text{Fano} > 1.5$) | Recurring institutional or narrative scenes localized to specific chapters (`presten`, `legen`, `politimannen`, `rettssaken`, `båten`). |
| **Q4. Focused Thematic Motifs** | Low | High ($\text{Fano} > 1.6$) | Specialized historical events (`eidsvoll`, `unionsoppløsningen`, `pest`) and minority lived worlds (`ramadan`, `joik`, `asylmottaket`, `tegnspråk`). |

### 2.4 Dual-Graph ($L(G)$) Overlapping Clustering of Chunk Co-Occurrences
From the sparse chunk-by-word matrix $C$ (weighted by $\log_2(1+c)$ and restricted to bursty content words in Q3–Q4), we compute the normalized chunk co-occurrence graph $W = \tilde{C}^T \tilde{C}$. Rather than partitioning nodes of $G$ directly into disjoint clusters, we construct the Adamic–Adar weighted line graph $L(G)$ whose nodes are the *edges* (semantic relations) of $G$, apply Leiden modularity optimization to both $G$ and $L(G)$, and project edge communities back to words via shielded internal support and Shannon neighborhood-entropy hub pruning ($\tau_H = 0.78$). Words participating in multiple relational contexts naturally emerge as multi-cluster bridge nodes.

---

## 3. Empirical Case Studies

### Case A: 260 Years of Norwegian Historiography (Tvedt Corpus, 1753–2015)
We apply the pipeline to Terje Tvedt’s historiographical corpus of $D = 352$ Norwegian history books ($162,261$ chunks of $300$ words; $897,907$ book-word profiles).
1. **Language-agnostic stopword invariance:** Across 18th-century Danish-Norwegian (`havde`, `af`, `paa`, `blev`), 20th-century Nynorsk (`dei`, `vart`, `ikkje`), and modern Bokmål (`og`, `var`, `ikke`), grammatical words cluster tightly at $\text{Fano} \in [0.88, 1.08]$ and $\text{DF}/\mathbb{E}[\text{DF}] \in [0.97, 1.02]$ inside every single volume, enabling multi-century SVD without curated historical stopword lists.
2. **Testing Tvedt’s "Methodological Nationalism" Hypothesis:** Comparing national and international geographical terms across all $352$ volumes reveals a striking structural divide:
   - *Norway* (`norge`, `norges`, $122,493$ occurrences) appears in nearly every book with a median chunk dispersion of **$39.1\%$** ($\text{DF}/\mathbb{E}[\text{DF}] = 0.82$), functioning as the ubiquitous grammatical and narrative frame.
   - *Europe* (`europa`, $5,433$ occurrences across $236$ books) and *Asia* (`asia`, $526$ occurrences across $52$ books) exhibit **$\text{median Fano} = 1.133$ ($\text{DF}/\mathbb{E}[\text{DF}] = 0.934$)** and **$\text{median Fano} = 0.994$ ($\text{DF}/\mathbb{E}[\text{DF}] = 1.003$)**, respectively. Their distribution is statistically indistinguishable from random Poisson scattering—confirming quantitatively that "Europe" and "Asia" appear predominantly as single-sentence passing mentions rather than sustained comparative analysis.
   - By contrast, Nordic neighbors and great powers involved in specific wars, unions, or treaties—*Iceland* (`island`, $\text{Fano} = 1.725$), *Sweden* (`sverige`, $\text{Fano} = 1.660$), *Denmark* (`danmark`, $\text{Fano} = 1.641$), and *Russia* (`russland`, $\text{Fano} = 1.634$)—concentrate in dense chapter-level bursts ($\text{DF}/\mathbb{E}[\text{DF}] \in [0.71, 0.74]$).

### Case B: Diversity in Contemporary Children’s and YA Fiction (BBI Corpus, 2000–2024)
To test whether micro-level chunk burstiness predicts semantic depth at the sentence level, we analyze the BBI Diversity corpus of $7,012$ non-translated Norwegian Bokmål children’s and YA novels (2000–2024), focusing on the $1,794$ unique books ($29,164,438$ chunk postings; $2,500,954$ book-word profiles) containing $6,137$ concordance fragments annotated with `gemini-2.5-flash` and validated against `gemini-3.1-pro-preview` ($\kappa = 0.55\text{–}0.82$) across five diversity domains (*disability, religion, gender/sexuality, Sámi/national minorities, multicultural/migration*).

1. **Cross-Validating Chunk Burstiness Against LLM Semantic Categories:**
   Across all $6,137$ annotated fragments, occurrences classified by the LLM as **metaphors, idioms, or homonyms** ($n = 1,342$) exhibit near-Poisson behavior ($\text{mean Fano} = 1.19$, $\text{DF}/\mathbb{E}[\text{DF}] = 0.920$, only $21.2\%$ bursty at $\text{Fano} > 1.15$), whereas **lived practice/identity** ($n = 2,720$) and **sustained thematic discourse** ($n = 1,428$) exhibit strong within-book clustering ($41.7\%$ and $58.2\%$ bursty; $\text{mean subgroup Fano} = 1.41$ and $1.64$). Within ambiguous lexical subgroups, chunk burstiness cleanly separates denotative representation from figurative noise:
   - *Blindness (`blind/blinde`):* Denotative character disability ($n=123$) averages $5.83$ hits/book with **$\text{Fano} = 1.56$** ($61.0\%$ bursty, $\text{DF}/\mathbb{E}[\text{DF}] = 0.775$), whereas metaphorical usage (*i blinde*, $n=77$) averages $2.44$ hits/book with **$\text{Fano} = 1.06$** ($16.9\%$ bursty, $\text{DF}/\mathbb{E}[\text{DF}] = 0.929$).
   - *Deafness (`døv/døve`) & Paralysis (`lam/lamme`):* Denotative deafness ($n=111$) has $\text{Fano} = 1.30$ ($36.0\%$ bursty) vs. metaphorical *for døve ører* ($n=73$, $\text{Fano} = 1.11$, $8.2\%$ bursty) and temporary ear ringing ($n=31$, $\text{Fano} = 0.99$, $0.0\%$ bursty). Figurative paralysis (*lam i knotten*, $n=37$) sits at exact Poisson noise ($\text{Fano} = 0.985$, $\text{DF}/\mathbb{E}[\text{DF}] = 0.999$, $0.0\%$ bursty), compared to $\text{Fano} = 1.46$ for actual paralysis ($n=62$).
   - *Queer (`skeiv*`) & Escape (`flukt/flykte*`):* Physical descriptions (*et skeivt smil*, $n=206$) and generic action/fantasy chases ($n=143$) exhibit pure Poisson dispersion ($\text{Fano} = 1.011$ and $1.046$; $\text{DF}/\mathbb{E}[\text{DF}] = 0.987$ and $0.948$).
   - *Sámi cultural markers (`joik`, `kofte`, `lavvo`):* Lived Sámi cultural practice ($n=100$) yields $\text{subgroup Fano} = 1.82$ ($\text{DF}/\mathbb{E}[\text{DF}] = 0.776$), whereas homonymous everyday objects (knitted cardigans, school camping tents, $n=97$) drop to $\text{Fano} = 1.18$ ($\text{DF}/\mathbb{E}[\text{DF}] = 0.920$).
   - *Mosques (`moské*`):* Books depicting lived Islamic worship ($n=145$) contain more than twice as many occurrences per book ($\text{mean TF} = 8.56$, subgroup $\text{TF} = 14.52$, $66.2\%$ bursty) as books where a mosque merely serves as an urban spatial landmark ($n=135$, $\text{mean TF} = 4.15$) or historical reference ($n=39$, $\text{mean TF} = 3.08$).

2. **Unsupervised Vocabulary Expansion & Dual-Graph $L(G)$ Clustering:**
   Using Double Burstiness ($\text{Q3–Q4}$) and Dual-Graph $L(G)$ clustering on the BBI chunk matrix ($C^T C$), our pipeline automatically recovers coherent overlapping clusters for all five diversity domains while discovering dozens of high-burst thematic markers absent from the hand-crafted 2021 lexicon—including `bønnerommet`, `profeten`, `imam` (religion); `hjelpemidler`, `likeverd`, `venterommet` (disability); `paraden`, `homofili`, `unormal` (queer/norm critique); `reinsdyr`, `karasjok`, `kautokeino`, `nordlyset`, `lasso`, `skallene`, `somby` (Sámi); and `udi`, `asyl`, `søknaden`, `mottaket` (migration/asylum).

---

## 4. Two-Stage Open Architecture for Scale
To support corpora scaling from $350$ to $10,000+$ volumes, we pair a fast local Python preparation pipeline (`scipy.sparse`, `igraph`, and DH-lab API batching) with a zero-backend, browser-native interactive inspector (`https://yoonsen.github.io/doc-to-vec/`) that ships a compressed sparse row (CSR) bundle for instant client-side SVD axis inspection, on-the-fly cosine similarity, and Dual-Graph exploration.
