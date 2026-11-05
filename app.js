// Doc2Vec · PPMI, Chunk-Burstiness & Dual-Graph Galakser (Ren klient-side JS)

const state = {
  bundle: null,
  nBooks: 0,
  nWords: 0,
  nnz: 0,
  rowPtr: null,
  colIdx: null,
  freq: null,
  fano: null,
  dfRatio: null,
  docTokens: null,
  wordTfBg: null,
  wordStop: null,
  // Aktive beregnede strukturer
  weights: null,      // Float32Array(nnz) - L2-normalisert per bok
  rawWeights: null,   // Float32Array(nnz) - før L2-norm
  wordTotalWeight: null, // Float32Array(nWords)
  kDims: 10,
  singularValues: null, // Float64Array(kDims)
  U: null,              // Array of Float64Array(nBooks) per dim
  V: null,              // Array of Float64Array(nWords) per dim
  G: null,              // Float32Array(nBooks * nBooks) - eksakt sparse cosinus mellom bøker
  selectedBookId: 4,
  axisX: 1,             // Default Akse 2 (indeks 1)
  axisY: 3,             // Default Akse 4 (indeks 3)
  // Graf-tilstand
  graphNodes: [],
  graphEdges: [],
  graphClusters: [],
  selectedWordHighlight: "",
};

const CLUSTER_COLORS = [
  "#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
  "#fb7185", "#2dd4bf", "#f97316", "#60a5fa", "#a3e635",
  "#e879f9", "#eab308"
];

const COUNTRIES_SET = new Set([
  "norge", "norges", "noreg", "norsk", "norske", "danmark", "sverige",
  "england", "storbritannia", "island", "europa", "russland", "sovjet",
  "tyskland", "preussen", "frankrike", "italia", "rom", "spania",
  "nederland", "holland", "finland", "amerika", "usa", "afrika", "asia", "india", "kina"
]);

async function init() {
  bindUIEvents();
  try {
    const resp = await fetch("tvedt_bundle.json");
    const bundle = await resp.json();
    loadBundleIntoMemory(bundle);
    recomputeAll();
  } catch (err) {
    document.getElementById("corpusSubtitle").textContent =
      "Kunne ikke laste tvedt_bundle.json automatisk: " + err.message;
  }
}

function loadBundleIntoMemory(bundle) {
  state.bundle = bundle;
  state.nBooks = bundle.meta.n_books;
  state.nWords = bundle.meta.n_words;
  state.nnz = bundle.meta.nnz;

  state.rowPtr = new Int32Array(bundle.csr.row_ptr);
  state.colIdx = new Int32Array(bundle.csr.col_idx);
  state.freq = new Int32Array(bundle.csr.freq);

  state.fano = new Float32Array(state.nnz);
  state.dfRatio = new Float32Array(state.nnz);
  for (let i = 0; i < state.nnz; i++) {
    state.fano[i] = bundle.csr.fano_x100[i] / 100.0;
    state.dfRatio[i] = bundle.csr.dfratio_x100[i] / 100.0;
  }

  state.docTokens = new Float32Array(state.nBooks);
  for (let d = 0; d < state.nBooks; d++) {
    state.docTokens[d] = Math.max(1, bundle.books[d].doc_tokens);
  }

  state.wordTfBg = new Float64Array(state.nWords);
  state.wordStop = new Uint8Array(state.nWords);
  for (let w = 0; w < state.nWords; w++) {
    state.wordTfBg[w] = Math.max(1, bundle.words[w].tf_bg);
    state.wordStop[w] = bundle.words[w].stop ? 1 : 0;
  }

  state.weights = new Float32Array(state.nnz);
  state.rawWeights = new Float32Array(state.nnz);
  state.wordTotalWeight = new Float32Array(state.nWords);

  document.getElementById("corpusSubtitle").textContent =
    `${bundle.meta.corpus_name} · ${state.nBooks} bøker · ${state.nWords.toLocaleString("no-NO")} ord · ${state.nnz.toLocaleString("no-NO")} par`;

  populateSelects();
}

function populateSelects() {
  const axisXSel = document.getElementById("axisXSelect");
  const axisYSel = document.getElementById("axisYSelect");
  axisXSel.innerHTML = "";
  axisYSel.innerHTML = "";
  for (let k = 0; k < state.kDims; k++) {
    const optX = document.createElement("option");
    optX.value = k;
    optX.textContent = `Akse ${k + 1}`;
    if (k === state.axisX) optX.selected = true;
    axisXSel.appendChild(optX);

    const optY = document.createElement("option");
    optY.value = k;
    optY.textContent = `Akse ${k + 1}`;
    if (k === state.axisY) optY.selected = true;
    axisYSel.appendChild(optY);
  }

  const bookSel = document.getElementById("bookSelect");
  bookSel.innerHTML = "";
  state.bundle.books.forEach((b) => {
    const opt = document.createElement("option");
    opt.value = b.id;
    const auth = b.authors ? b.authors.split("/")[0].trim() : "";
    opt.textContent = `[${b.year || "?"}] ${b.title.slice(0, 62)} (${auth})`;
    if (b.id === state.selectedBookId) opt.selected = true;
    bookSel.appendChild(opt);
  });
}

function recomputeAll() {
  const t0 = performance.now();
  computeWeightsAndSVD();
  const dtSvd = performance.now() - t0;

  const t1 = performance.now();
  buildWordGraphAndDualClusters();
  const dtGraph = performance.now() - t1;

  document.getElementById("perfBadge").textContent =
    `SVD: ${dtSvd.toFixed(0)} ms · Graf: ${dtGraph.toFixed(0)} ms`;

  renderScatterPlot();
  renderBookInspector();
  renderAxes();
  renderGalaxyAndClusters();
  renderWordTable();
}

function computeWeightsAndSVD() {
  const mode = document.getElementById("weightMode").value;
  const filterStop = document.getElementById("filterStop").checked;
  const sublinearTf = document.getElementById("sublinearTf").checked;
  const minFano = parseFloat(document.getElementById("minFano").value);
  const maxDfRatio = parseFloat(document.getElementById("maxDfRatio").value);
  const alpha = parseFloat(document.getElementById("ppmiAlpha").value);

  const nBooks = state.nBooks;
  const nWords = state.nWords;
  const rowPtr = state.rowPtr;
  const colIdx = state.colIdx;
  const freq = state.freq;
  const fano = state.fano;
  const dfRatio = state.dfRatio;
  const wordStop = state.wordStop;
  const weights = state.weights;
  const rawWeights = state.rawWeights;
  const wordTotalWeight = state.wordTotalWeight;
  wordTotalWeight.fill(0);

  // Forhåndsberegn log2(P_alpha(w)) for PPMI
  const log2Pw = new Float32Array(nWords);
  if (mode === "ppmi" || mode === "hybrid") {
    let bgSum = 0.0;
    const powered = new Float64Array(nWords);
    for (let w = 0; w < nWords; w++) {
      const p = alpha === 1.0 ? state.wordTfBg[w] : Math.pow(state.wordTfBg[w], alpha);
      powered[w] = p;
      bgSum += p;
    }
    // Skaler med total_bg_sum for realistisk vokabulardekning
    const scaleBg = alpha === 1.0 ? Math.max(bgSum, state.bundle.meta.total_bg_sum) : bgSum * 4.5;
    for (let w = 0; w < nWords; w++) {
      log2Pw[w] = Math.log2(powered[w] / scaleBg);
    }
  }

  // 1. Beregn vekter og L2-normaliser hver bokrad
  for (let d = 0; d < nBooks; d++) {
    const start = rowPtr[d];
    const end = rowPtr[d + 1];
    const log2DocLen = Math.log2(state.docTokens[d]);
    let normSq = 0.0;

    for (let idx = start; idx < end; idx++) {
      const w = colIdx[idx];
      if (filterStop && wordStop[w]) {
        weights[idx] = 0;
        rawWeights[idx] = 0;
        continue;
      }

      const fn = fano[idx];
      const dfr = dfRatio[idx];
      if (fn < minFano || dfr > maxDfRatio) {
        weights[idx] = 0;
        rawWeights[idx] = 0;
        continue;
      }

      const c = freq[idx];
      const tfScale = sublinearTf ? Math.log2(1.0 + c) : 1.0;
      let val = 0.0;

      if (mode === "burst") {
        const excessFano = Math.max(0.0, fn - minFano + 0.15);
        const conc = Math.max(0.05, 1.0 - dfr * 0.85);
        val = tfScale * excessFano * conc;
      } else if (mode === "ppmi") {
        const pmi = Math.log2(c) - log2DocLen - log2Pw[w];
        val = pmi > 0 ? tfScale * pmi : 0.0;
      } else {
        // Hybrid: PPMI * overdispersjon
        const pmi = Math.log2(c) - log2DocLen - log2Pw[w];
        const excessFano = Math.max(0.0, fn - minFano + 0.2);
        val = pmi > 0 ? tfScale * pmi * excessFano : 0.0;
      }

      rawWeights[idx] = val;
      weights[idx] = val;
      normSq += val * val;
    }

    const invNorm = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 0.0;
    for (let idx = start; idx < end; idx++) {
      const wNorm = weights[idx] * invNorm;
      weights[idx] = wNorm;
      wordTotalWeight[colIdx[idx]] += wNorm;
    }
  }

  // 2. Beregn Gram-matrisen G = M * M^T (nBooks x nBooks)
  // Siden M er glissen med radpekere, er G[d1, d2] = dot(row_d1, row_d2)
  // Vi bygger en rask kolonne-posting-liste kun for ikke-null elementer
  const colCounts = new Int32Array(nWords);
  for (let idx = 0; idx < state.nnz; idx++) {
    if (weights[idx] > 0) colCounts[colIdx[idx]]++;
  }
  const colPtr = new Int32Array(nWords + 1);
  for (let w = 0; w < nWords; w++) colPtr[w + 1] = colPtr[w] + colCounts[w];

  const activeNnz = colPtr[nWords];
  const cscDoc = new Int32Array(activeNnz);
  const cscVal = new Float32Array(activeNnz);
  const curPos = new Int32Array(colPtr.subarray(0, nWords));

  for (let d = 0; d < nBooks; d++) {
    for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
      const v = weights[idx];
      if (v > 0) {
        const w = colIdx[idx];
        const p = curPos[w]++;
        cscDoc[p] = d;
        cscVal[p] = v;
      }
    }
  }

  const G = new Float32Array(nBooks * nBooks);
  for (let w = 0; w < nWords; w++) {
    const start = colPtr[w];
    const end = colPtr[w + 1];
    for (let i = start; i < end; i++) {
      const d1 = cscDoc[i] * nBooks;
      const v1 = cscVal[i];
      for (let j = i; j < end; j++) {
        G[d1 + cscDoc[j]] += v1 * cscVal[j];
      }
    }
  }
  // Symmetriser nedre trekant
  for (let d1 = 0; d1 < nBooks; d1++) {
    for (let d2 = 0; d2 < d1; d2++) {
      G[d1 * nBooks + d2] = G[d2 * nBooks + d1];
    }
  }
  state.G = G;

  // 3. Deflatert potensiterasjon med Gram-Schmidt på G (nBooks x nBooks) for topp kDims egenvektorer U_k
  const kDims = state.kDims;
  const U = [];
  const singularValues = new Float64Array(kDims);

  for (let k = 0; k < kDims; k++) {
    const u = new Float64Array(nBooks);
    // Deterministisk startvektor
    for (let d = 0; d < nBooks; d++) {
      u[d] = Math.sin((d + 1) * (k + 1.37));
    }

    const nextU = new Float64Array(nBooks);
    for (let iter = 0; iter < 28; iter++) {
      // nextU = G * u
      for (let d1 = 0; d1 < nBooks; d1++) {
        let sum = 0.0;
        const rowOff = d1 * nBooks;
        for (let d2 = 0; d2 < nBooks; d2++) {
          sum += G[rowOff + d2] * u[d2];
        }
        nextU[d1] = sum;
      }

      // Ortogonaliser mot tidligere egenvektorer U[0..k-1]
      for (let prev = 0; prev < k; prev++) {
        const up = U[prev];
        let dot = 0.0;
        for (let d = 0; d < nBooks; d++) dot += nextU[d] * up[d];
        for (let d = 0; d < nBooks; d++) nextU[d] -= dot * up[d];
      }

      let norm = 0.0;
      for (let d = 0; d < nBooks; d++) norm += nextU[d] * nextU[d];
      norm = Math.sqrt(norm) || 1.0;
      for (let d = 0; d < nBooks; d++) u[d] = nextU[d] / norm;
    }

    // Beregn Rayleigh-kvotient lambda = u^T G u -> sigma = sqrt(lambda)
    let lambda = 0.0;
    for (let d1 = 0; d1 < nBooks; d1++) {
      let rowSum = 0.0;
      const rowOff = d1 * nBooks;
      for (let d2 = 0; d2 < nBooks; d2++) rowSum += G[rowOff + d2] * u[d2];
      lambda += u[d1] * rowSum;
    }
    singularValues[k] = Math.sqrt(Math.max(1e-9, lambda));

    // Perron-Frobenius: for k=0 (tyngdepunkt), sørg for positiv sum
    if (k === 0) {
      let s = 0;
      for (let d = 0; d < nBooks; d++) s += u[d];
      if (s < 0) for (let d = 0; d < nBooks; d++) u[d] = -u[d];
    }
    U.push(u);
  }

  // 4. Projiser ordvektorer V_k = M^T * U_k / sigma_k
  const V = [];
  for (let k = 0; k < kDims; k++) {
    const v = new Float64Array(nWords);
    const u = U[k];
    const invSigma = 1.0 / singularValues[k];
    for (let d = 0; d < nBooks; d++) {
      const ud = u[d] * invSigma;
      for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
        const wVal = weights[idx];
        if (wVal > 0) v[colIdx[idx]] += wVal * ud;
      }
    }
    V.push(v);
  }

  state.singularValues = singularValues;
  state.U = U;
  state.V = V;
}

// =============================================================================
// TAB 1: 2D BOK-KART & BOK-INSPEKTØR
// =============================================================================

function getEraColor(year) {
  if (!year || year < 1890) return "#f472b6";
  if (year <= 1917) return "#fbbf24";
  if (year <= 1969) return "#34d399";
  return "#38bdf8";
}

function renderScatterPlot() {
  const canvas = document.getElementById("scatterCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const ux = state.U[state.axisX];
  const uy = state.U[state.axisY];
  const sx = Math.sqrt(state.singularValues[state.axisX]);
  const sy = Math.sqrt(state.singularValues[state.axisY]);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let d = 0; d < state.nBooks; d++) {
    const x = ux[d] * sx;
    const y = uy[d] * sy;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = 42;
  const spanX = (maxX - minX) || 1;
  const spanY = (maxY - minY) || 1;

  const toScreenX = (x) => pad + ((x - minX) / spanX) * (W - 2 * pad);
  const toScreenY = (y) => H - pad - ((y - minY) / spanY) * (H - 2 * pad);

  // Tegn null-linjer hvis i rekkevidde
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  if (minX < 0 && maxX > 0) {
    const zx = toScreenX(0);
    ctx.beginPath(); ctx.moveTo(zx, 0); ctx.lineTo(zx, H); ctx.stroke();
  }
  if (minY < 0 && maxY > 0) {
    const zy = toScreenY(0);
    ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(W, zy); ctx.stroke();
  }

  state.scatterPoints = [];
  for (let d = 0; d < state.nBooks; d++) {
    const b = state.bundle.books[d];
    const px = toScreenX(ux[d] * sx);
    const py = toScreenY(uy[d] * sy);
    state.scatterPoints.push({ id: d, x: px, y: py });

    const isSel = d === state.selectedBookId;
    ctx.beginPath();
    ctx.arc(px, py, isSel ? 8 : 4.5, 0, Math.PI * 2);
    ctx.fillStyle = getEraColor(b.year);
    ctx.globalAlpha = isSel ? 1.0 : 0.78;
    ctx.fill();
    if (isSel) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1.0;

  // Etikett for valgt bok
  const selPt = state.scatterPoints[state.selectedBookId];
  if (selPt) {
    const b = state.bundle.books[state.selectedBookId];
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillText(`[${b.year}] ${b.title.slice(0, 42)}`, Math.min(W - 240, selPt.x + 10), Math.max(18, selPt.y - 8));
  }
}

function renderBookInspector() {
  const d = state.selectedBookId;
  const b = state.bundle.books[d];
  const container = document.getElementById("bookInspector");

  // Beregn SVD-cosinus og sparse-cosinus mot alle andre bøker
  const kDims = state.kDims;
  const mySvd = new Float64Array(kDims);
  let myNorm = 0;
  for (let k = 0; k < kDims; k++) {
    const v = state.U[k][d] * Math.sqrt(state.singularValues[k]);
    mySvd[k] = v;
    myNorm += v * v;
  }
  myNorm = Math.sqrt(myNorm) || 1;

  const sims = [];
  for (let other = 0; other < state.nBooks; other++) {
    if (other === d) continue;
    let dot = 0, otherNorm = 0;
    for (let k = 0; k < kDims; k++) {
      const ov = state.U[k][other] * Math.sqrt(state.singularValues[k]);
      dot += mySvd[k] * ov;
      otherNorm += ov * ov;
    }
    const svdSim = dot / (myNorm * (Math.sqrt(otherNorm) || 1));
    const sparseSim = state.G[d * state.nBooks + other];
    sims.push({ id: other, svdSim, sparseSim });
  }
  sims.sort((a, b) => b.svdSim - a.svdSim);

  // Hent toppord i denne boka etter gjeldende vekting
  const bookWords = [];
  for (let idx = state.rowPtr[d]; idx < state.rowPtr[d + 1]; idx++) {
    const wVal = state.rawWeights[idx];
    if (wVal > 0) {
      const wId = state.colIdx[idx];
      bookWords.push({
        word: state.bundle.words[wId].w,
        freq: state.freq[idx],
        fano: state.fano[idx],
        dfRatio: state.dfRatio[idx],
        weight: wVal,
      });
    }
  }
  bookWords.sort((a, b) => b.weight - a.weight);

  const nbHtml = sims.slice(0, 6).map((nb, idx) => {
    const ob = state.bundle.books[nb.id];
    const auth = ob.authors ? ob.authors.split("/")[0].trim() : "";
    return `
      <div style="padding:0.4rem 0;border-block-end:1px solid var(--border);font-size:0.82rem;cursor:pointer;"
           onclick="selectBook(${ob.id})">
        <strong>${idx + 1}. [${ob.year}] ${ob.title.slice(0, 56)}</strong>
        <div style="color:var(--text-muted);display:flex;justify-content:space-between;">
          <span>${auth}</span>
          <span style="color:var(--accent);">SVD: ${nb.svdSim.toFixed(3)} · Sparse: ${nb.sparseSim.toFixed(3)}</span>
        </div>
      </div>`;
  }).join("");

  const wordsHtml = bookWords.slice(0, 18).map((bw) =>
    `<span class="tag" title="freq=${bw.freq}, Fano=${bw.fano.toFixed(2)}, DF/E[DF]=${bw.dfRatio.toFixed(2)}"
           onclick="jumpToWord('${bw.word}')">${bw.word} <small style="opacity:0.75">(${bw.freq} · F:${bw.fano.toFixed(1)})</small></span>`
  ).join("");

  container.innerHTML = `
    <div class="axis-card" style="margin-block-end:0.65rem;">
      <div style="font-weight:700;font-size:0.95rem;">${b.title}</div>
      <div style="font-size:0.8rem;color:var(--text-muted);margin-block-top:0.2rem;">
        ${b.authors || "Ukjent forfatter"} · Utgitt ${b.year} · ${b.n_chunks} chunks (à 300 ord)
      </div>
    </div>
    <h4 style="margin:0.5rem 0 0.35rem;font-size:0.83rem;color:var(--accent);text-transform:uppercase;">
      Topp 6 nærmeste naboer i vektorrommet
    </h4>
    <div>${nbHtml}</div>
    <h4 style="margin:0.75rem 0 0.35rem;font-size:0.83rem;color:var(--accent);text-transform:uppercase;">
      Bokas sterkeste signaturord (klikk for å inspisere)
    </h4>
    <div>${wordsHtml || "<em style='font-size:0.82rem;color:var(--text-muted);'>Ingen ord passerer filtrene.</em>"}</div>
  `;
}

window.selectBook = function(bookId) {
  state.selectedBookId = bookId;
  document.getElementById("bookSelect").value = bookId;
  renderScatterPlot();
  renderBookInspector();
};

window.jumpToWord = function(word) {
  document.querySelector('[data-tab="tab-world"]').click();
  document.getElementById("presetFilter").value = "all";
  const searchInput = document.getElementById("wordTableSearch");
  searchInput.value = word;
  renderWordTable();
};

// =============================================================================
// TAB 2: SVD-AKSE-INSPEKTØR
// =============================================================================

function renderAxes() {
  const container = document.getElementById("axesContainer");
  let html = "";

  for (let k = 0; k < state.kDims; k++) {
    const sigma = state.singularValues[k];
    const v = state.V[k];
    const u = state.U[k];

    const wordIndices = Array.from({ length: state.nWords }, (_, i) => i)
      .filter((i) => state.wordTotalWeight[i] > 0);
    wordIndices.sort((a, b) => v[b] - v[a]);

    const posWords = wordIndices.slice(0, 10).map((i) =>
      `<span class="tag" onclick="jumpToWord('${state.bundle.words[i].w}')">${state.bundle.words[i].w}</span>`
    ).join("");
    const negWords = wordIndices.slice(-10).reverse().map((i) =>
      `<span class="tag" onclick="jumpToWord('${state.bundle.words[i].w}')">${state.bundle.words[i].w}</span>`
    ).join("");

    const bookIndices = Array.from({ length: state.nBooks }, (_, i) => i);
    bookIndices.sort((a, b) => u[b] - u[a]);

    const fmtBook = (idx) => {
      const b = state.bundle.books[idx];
      return `<a href="#" style="color:var(--accent);text-decoration:none;" onclick="event.preventDefault();document.querySelector('[data-tab=\\'tab-map\\']').click();selectBook(${b.id});">[${b.year}] ${b.title.slice(0, 44)}</a>`;
    };

    const posBooks = bookIndices.slice(0, 2).map(fmtBook).join(" · ");
    const negBooks = bookIndices.slice(-2).reverse().map(fmtBook).join(" · ");

    html += `
      <div class="axis-card">
        <div class="axis-header">
          <span>Akse ${k + 1} ${k === 0 ? "(Tyngdepunkt / Perron–Frobenius)" : "(Bipolar kontrastakse)"}</span>
          <span style="color:var(--accent);font-variant-numeric:tabular-nums;">σ = ${sigma.toFixed(2)}</span>
        </div>
        <div class="pole-row">
          <span class="pole-pos">(+) Pol:</span> ${posWords}
          <div style="font-size:0.78rem;color:var(--text-muted);margin-inline-start:1rem;">Bøker: ${posBooks}</div>
        </div>
        ${k > 0 ? `
        <div class="pole-row" style="margin-block-top:0.4rem;">
          <span class="pole-neg">(−) Pol:</span> ${negWords}
          <div style="font-size:0.78rem;color:var(--text-muted);margin-inline-start:1rem;">Bøker: ${negBooks}</div>
        </div>` : ""}
      </div>
    `;
  }
  container.innerHTML = html;
}

// =============================================================================
// TAB 3: ORD-MOT-ORD DOT-PRODUKT & DUAL-GRAPH CLUSTRING (G + L(G))
// =============================================================================

function buildWordGraphAndDualClusters() {
  const topN = parseInt(document.getElementById("graphTopWords").value, 10);
  const topK = parseInt(document.getElementById("graphTopK").value, 10);

  // 1. Velg toppN ord med høyest samlet vekt (kun alfabetiske ord, og ikke over-spredte gjennomgangsord)
  const cand = [];
  for (let w = 0; w < state.nWords; w++) {
    const wm = state.bundle.words[w];
    if (state.wordTotalWeight[w] > 0 && wm.md <= 0.16 && wm.w.length >= 3) {
      cand.push({ wId: w, score: state.wordTotalWeight[w] });
    }
  }
  cand.sort((a, b) => b.score - a.score);
  const selected = cand.slice(0, topN).map((x) => x.wId);
  const V_count = selected.length;
  if (V_count < 5) {
    state.graphNodes = [];
    state.graphEdges = [];
    state.graphClusters = [];
    return;
  }

  const globalToLocal = new Int32Array(state.nWords).fill(-1);
  selected.forEach((wId, idx) => { globalToLocal[wId] = idx; });

  // 2. Bygg L2-normaliserte kolonnevektorer for de valgte ordene over de 352 bøkene
  const colVecs = Array.from({ length: V_count }, () => new Float32Array(state.nBooks));
  for (let d = 0; d < state.nBooks; d++) {
    for (let idx = state.rowPtr[d]; idx < state.rowPtr[d + 1]; idx++) {
      const loc = globalToLocal[state.colIdx[idx]];
      if (loc >= 0 && state.weights[idx] > 0) {
        colVecs[loc][d] = state.weights[idx];
      }
    }
  }
  for (let i = 0; i < V_count; i++) {
    const vec = colVecs[i];
    let norm = 0;
    for (let d = 0; d < state.nBooks; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < state.nBooks; d++) vec[d] /= norm;
  }

  // 3. Ord-mot-ord dot-produkt og k-NN sparsifisering
  const edgeMap = new Map();
  const adj = Array.from({ length: V_count }, () => new Set());
  const nodeStrength = new Float32Array(V_count);

  for (let i = 0; i < V_count; i++) {
    const vi = colVecs[i];
    const rowSims = [];
    for (let j = 0; j < V_count; j++) {
      if (i === j) continue;
      const vj = colVecs[j];
      let dot = 0;
      for (let d = 0; d < state.nBooks; d++) dot += vi[d] * vj[d];
      if (dot >= 0.14) rowSims.push({ j, sim: dot });
    }
    rowSims.sort((a, b) => b.sim - a.sim);
    for (let m = 0; m < Math.min(topK, rowSims.length); m++) {
      const { j, sim } = rowSims[m];
      const u = Math.min(i, j);
      const v = Math.max(i, j);
      const key = `${u}_${v}`;
      if (!edgeMap.has(key) || edgeMap.get(key).w < sim) {
        edgeMap.set(key, { u, v, w: sim });
      }
    }
  }

  const edges = Array.from(edgeMap.values());
  edges.forEach((e) => {
    adj[e.u].add(e.v);
    adj[e.v].add(e.u);
    nodeStrength[e.u] += e.w;
    nodeStrength[e.v] += e.w;
  });

  // 4. Grådig Louvain/Modularitets-klustring på G
  const baseComm = runFastModularityClustering(V_count, edges);

  // 5. Dual-Graph / Linjegraf L(G) med Adamic-Adar straff + Shielded Set-Cover overlapp (som i galaxies)
  const nodeToEdges = Array.from({ length: V_count }, () => []);
  edges.forEach((e, eIdx) => {
    nodeToEdges[e.u].push(eIdx);
    nodeToEdges[e.v].push(eIdx);
  });

  const dualEdges = [];
  for (let v = 0; v < V_count; v++) {
    const inc = nodeToEdges[v];
    const deg = inc.length;
    if (deg < 2) continue;
    const penalty = deg > 2 ? 1.0 / Math.log(deg) : 1.0;
    for (let a = 0; a < deg; a++) {
      for (let b = a + 1; b < deg; b++) {
        const e1 = edges[inc[a]];
        const e2 = edges[inc[b]];
        dualEdges.push({ u: inc[a], v: inc[b], w: e1.w * e2.w * penalty });
      }
    }
  }
  const dualComm = edges.length > 0 ? runFastModularityClustering(edges.length, dualEdges) : [];

  // Koble orig-klynger med beste dual-klynge og utvid med forankringstest (support >= 2 interne naboer)
  const origClusters = new Map();
  for (let i = 0; i < V_count; i++) {
    const c = baseComm[i];
    if (!origClusters.has(c)) origClusters.set(c, new Set());
    origClusters.get(c).add(i);
  }

  const dualClusterNodes = new Map();
  for (let eIdx = 0; eIdx < edges.length; eIdx++) {
    const dc = dualComm[eIdx];
    if (!dualClusterNodes.has(dc)) dualClusterNodes.set(dc, new Set());
    dualClusterNodes.get(dc).add(edges[eIdx].u);
    dualClusterNodes.get(dc).add(edges[eIdx].v);
  }

  const overlappingClusters = [];
  const nodeClusterCount = new Int32Array(V_count);

  origClusters.forEach((origSet) => {
    if (origSet.size < 3) return;
    let bestDualSet = new Set();
    let bestOv = 0;
    dualClusterNodes.forEach((dSet) => {
      let ov = 0;
      origSet.forEach((n) => { if (dSet.has(n)) ov++; });
      if (ov > bestOv) { bestOv = ov; bestDualSet = dSet; }
    });

    const expanded = new Set(origSet);
    const candidates = new Set(bestDualSet);
    origSet.forEach((u) => adj[u].forEach((nbr) => candidates.add(nbr)));

    candidates.forEach((candNode) => {
      if (expanded.has(candNode)) return;
      let support = 0;
      adj[candNode].forEach((nbr) => { if (origSet.has(nbr)) support++; });
      if (support >= 2) expanded.add(candNode);
    });

    const members = Array.from(expanded).sort((a, b) => nodeStrength[b] - nodeStrength[a]);
    members.forEach((n) => nodeClusterCount[n]++);
    overlappingClusters.push({ members });
  });

  overlappingClusters.sort((a, b) => b.members.length - a.members.length);

  // Tilordne primærfarge per node
  const primaryCluster = new Int32Array(V_count).fill(0);
  overlappingClusters.forEach((cl, cIdx) => {
    cl.color = CLUSTER_COLORS[cIdx % CLUSTER_COLORS.length];
    const w0 = state.bundle.words[selected[cl.members[0]]].w;
    const w1 = cl.members.length > 1 ? state.bundle.words[selected[cl.members[1]]].w : "";
    cl.name = w1 ? `${w0}–${w1}` : w0;
  });
  // Siste pass for primærfarge fra største/første klynge
  for (let cIdx = overlappingClusters.length - 1; cIdx >= 0; cIdx--) {
    overlappingClusters[cIdx].members.forEach((n) => { primaryCluster[n] = cIdx; });
  }

  // Initialiser 2D posisjoner medSVD-koordinatorer (V[1], V[2]) + kort force-directed avslapning
  const nodes = selected.map((wId, i) => ({
    id: i,
    wId,
    word: state.bundle.words[wId].w,
    strength: nodeStrength[i],
    clusterIdx: primaryCluster[i],
    nClusters: nodeClusterCount[i],
    x: (state.V[1][wId] || (Math.random() - 0.5)) * 1800 + 380,
    y: (state.V[2][wId] || (Math.random() - 0.5)) * 1800 + 260,
    vx: 0,
    vy: 0,
  }));

  runForceLayout(nodes, edges, 95);

  state.graphNodes = nodes;
  state.graphEdges = edges;
  state.graphClusters = overlappingClusters;
}

function runFastModularityClustering(nNodes, edges) {
  const comm = new Int32Array(nNodes);
  for (let i = 0; i < nNodes; i++) comm[i] = i;

  const nbrs = Array.from({ length: nNodes }, () => []);
  let totalWeight = 0;
  const deg = new Float32Array(nNodes);
  edges.forEach((e) => {
    nbrs[e.u].push({ to: e.v, w: e.w });
    nbrs[e.v].push({ to: e.u, w: e.w });
    deg[e.u] += e.w;
    deg[e.v] += e.w;
    totalWeight += e.w;
  });
  const m2 = Math.max(1e-6, 2.0 * totalWeight);
  const commTot = new Float32Array(deg);

  for (let pass = 0; pass < 12; pass++) {
    let moved = false;
    for (let i = 0; i < nNodes; i++) {
      const oldC = comm[i];
      const ki = deg[i];
      commTot[oldC] -= ki;

      const cWeights = new Map();
      nbrs[i].forEach((edge) => {
        const c = comm[edge.to];
        cWeights.set(c, (cWeights.get(c) || 0) + edge.w);
      });

      let bestC = oldC;
      let bestGain = (cWeights.get(oldC) || 0) - (commTot[oldC] * ki) / m2;

      cWeights.forEach((wIn, candC) => {
        const gain = wIn - (commTot[candC] * ki) / m2;
        if (gain > bestGain + 1e-6) {
          bestGain = gain;
          bestC = candC;
        }
      });

      comm[i] = bestC;
      commTot[bestC] += ki;
      if (bestC !== oldC) moved = true;
    }
    if (!moved) break;
  }
  return comm;
}

function runForceLayout(nodes, edges, iterations = 80) {
  const W = 760, H = 520;
  const n = nodes.length;
  for (let it = 0; it < iterations; it++) {
    const temp = 1.0 - it / iterations;
    // Frastøting
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = nodes[i].x - nodes[j].x;
        let dy = nodes[i].y - nodes[j].y;
        const distSq = dx * dx + dy * dy + 25;
        if (distSq > 22000) continue;
        const f = (420 / distSq) * temp;
        nodes[i].vx += dx * f;
        nodes[i].vy += dy * f;
        nodes[j].vx -= dx * f;
        nodes[j].vy -= dy * f;
      }
    }
    // Tiltrekning langs kanter
    edges.forEach((e) => {
      const u = nodes[e.u];
      const v = nodes[e.v];
      const dx = v.x - u.x;
      const dy = v.y - u.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (dist - 48) * 0.035 * e.w * temp;
      const fx = (dx / dist) * f;
      const fy = (dy / dist) * f;
      u.vx += fx; u.vy += fy;
      v.vx -= fx; v.vy -= fy;
    });
    // Sentrering og oppdatering
    for (let i = 0; i < n; i++) {
      const nd = nodes[i];
      nd.vx += (W / 2 - nd.x) * 0.012;
      nd.vy += (H / 2 - nd.y) * 0.012;
      nd.x = Math.max(36, Math.min(W - 36, nd.x + nd.vx * 0.25));
      nd.y = Math.max(24, Math.min(H - 24, nd.y + nd.vy * 0.25));
      nd.vx *= 0.6;
      nd.vy *= 0.6;
    }
  }
}

function renderGalaxyAndClusters() {
  const canvas = document.getElementById("galaxyCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const nodes = state.graphNodes;
  const edges = state.graphEdges;
  const clusters = state.graphClusters;
  const q = (document.getElementById("galaxySearch").value || "").trim().toLowerCase();

  // Tegn kanter
  ctx.lineWidth = 1;
  edges.forEach((e) => {
    const u = nodes[e.u];
    const v = nodes[e.v];
    ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
    ctx.beginPath();
    ctx.moveTo(u.x, u.y);
    ctx.lineTo(v.x, v.y);
    ctx.stroke();
  });

  // Tegn noder
  nodes.forEach((nd) => {
    const isMatch = q && nd.word.includes(q);
    const r = Math.min(11, 3.5 + Math.sqrt(nd.strength) * 2.2);
    const color = clusters[nd.clusterIdx] ? clusters[nd.clusterIdx].color : "#38bdf8";

    ctx.beginPath();
    ctx.arc(nd.x, nd.y, isMatch ? r + 4 : r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = q && !isMatch ? 0.22 : 0.92;
    ctx.fill();

    if (nd.nClusters >= 2 || isMatch) {
      ctx.lineWidth = isMatch ? 2.5 : 1.5;
      ctx.strokeStyle = isMatch ? "#ffffff" : "#fbbf24";
      ctx.stroke();
    }

    if (nd.strength > 0.75 || isMatch || nodes.length <= 140) {
      ctx.fillStyle = isMatch ? "#ffffff" : "#e2e8f0";
      ctx.font = isMatch ? "700 12px system-ui, sans-serif" : "10.5px system-ui, sans-serif";
      ctx.fillText(nd.word, nd.x + r + 3, nd.y + 3);
    }
  });
  ctx.globalAlpha = 1.0;

  // Render klynge-kort og bro-ord
  const clContainer = document.getElementById("clustersContainer");
  const bridges = nodes.filter((nd) => nd.nClusters >= 2).sort((a, b) => b.nClusters - a.nClusters || b.strength - a.strength);

  let html = "";
  if (bridges.length > 0) {
    html += `
      <div class="cluster-card" style="border-color:rgba(251,191,36,0.4);">
        <div style="font-weight:700;font-size:0.84rem;color:var(--warn);margin-block-end:0.35rem;">
          Bro-ord mellom tematiske klynger (${bridges.length} ord i ≥ 2 klynger via L(G))
        </div>
        <div>
          ${bridges.slice(0, 18).map((b) => `<span class="tag bridge" onclick="highlightGalaxyWord('${b.word}')">${b.word} (${b.nClusters})</span>`).join("")}
        </div>
      </div>
    `;
  }

  clusters.forEach((cl) => {
    const wordsHtml = cl.members.slice(0, 22).map((mIdx) => {
      const nd = nodes[mIdx];
      return `<span class="tag" onclick="highlightGalaxyWord('${nd.word}')">${nd.word}</span>`;
    }).join("");
    html += `
      <div class="cluster-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-block-end:0.3rem;">
          <strong style="color:${cl.color};font-size:0.88rem;">● ${cl.name}</strong>
          <span style="font-size:0.76rem;color:var(--text-muted);">${cl.members.length} ord</span>
        </div>
        <div>${wordsHtml}</div>
      </div>
    `;
  });

  clContainer.innerHTML = html;
}

window.highlightGalaxyWord = function(w) {
  const inp = document.getElementById("galaxySearch");
  inp.value = inp.value === w ? "" : w;
  renderGalaxyAndClusters();
};

// =============================================================================
// TAB 4: «NORGE VS. VERDEN» & BEGREPSTABELL
// =============================================================================

function describeBehavior(w) {
  if (w.mf >= 2.0 && w.mr <= 0.72) return `<span style="color:var(--pos);font-weight:600;">Sterkt bursty (drøftes i egne avsnitt)</span>`;
  if (w.mf >= 1.55 && w.mr <= 0.82) return `<span style="color:var(--accent);">Tematisk klumpet</span>`;
  if (w.md >= 0.20) return `<span style="color:var(--warn);">Gjennomgående bakgrunnsramme</span>`;
  return `<span style="color:var(--text-muted);">Jevnt / tilfeldig dryss (Poisson ≈ 1)</span>`;
}

function renderWordTable() {
  const preset = document.getElementById("presetFilter").value;
  const q = (document.getElementById("wordTableSearch").value || "").trim().toLowerCase();
  const tbody = document.getElementById("wordTableBody");

  let list = state.bundle.words.slice();
  if (q) {
    list = list.filter((x) => x.w.includes(q));
  } else if (preset === "countries") {
    list = list.filter((x) => COUNTRIES_SET.has(x.w));
    list.sort((a, b) => b.cf - a.cf);
  } else if (preset === "top_burst") {
    list = list.filter((x) => !x.stop && x.nb >= 8 && x.md <= 0.14);
    list.sort((a, b) => b.mf - a.mf);
    list = list.slice(0, 150);
  } else if (preset === "even_bg") {
    list = list.filter((x) => x.nb >= 40);
    list.sort((a, b) => a.mf - b.mf);
    list = list.slice(0, 120);
  } else {
    list.sort((a, b) => b.cf - a.cf);
    list = list.slice(0, 200);
  }

  tbody.innerHTML = list.map((w) => `
    <tr>
      <th>${w.w}</th>
      <td>${w.nb}</td>
      <td>${w.cf.toLocaleString("no-NO")}</td>
      <td>${(w.md * 100).toFixed(1)} %</td>
      <td>${w.mr.toFixed(2)}</td>
      <td><strong>${w.mf.toFixed(2)}</strong></td>
      <td>${describeBehavior(w)}</td>
    </tr>
  `).join("");
}

// =============================================================================
// DIREKTE HENTING FRA NB API (https://api.nb.no/dhlab) I NETTLESEREN
// =============================================================================

async function runLiveApiFetch() {
  const prog = document.getElementById("apiProgress");
  const subject = document.getElementById("apiSubject").value.trim() || null;
  const author = document.getElementById("apiAuthor").value.trim() || null;
  const title = document.getElementById("apiTitle").value.trim() || null;
  const fromYear = parseInt(document.getElementById("apiFromYear").value, 10) || 1850;
  const toYear = parseInt(document.getElementById("apiToYear").value, 10) || 2020;
  const limit = Math.min(350, Math.max(10, parseInt(document.getElementById("apiLimit").value, 10) || 50));
  const alsoChunks = document.getElementById("apiFetchChunks").checked;

  try {
    prog.textContent = "1/3 Bygger korpus fra https://api.nb.no/dhlab/build_corpus ...";
    const corpusParams = { doctype: "digibok", from_year: fromYear, to_year: toYear, limit };
    if (subject) corpusParams.subject = subject;
    if (author) corpusParams.author = author;
    if (title) corpusParams.title = title;

    const cResp = await fetch("https://api.nb.no/dhlab/build_corpus", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corpusParams),
    });
    const cJson = await cResp.json();
    const urnMap = cJson.urn || {};
    const titleMap = cJson.title || {};
    const authMap = cJson.authors || {};
    const yearMap = cJson.year || {};
    const keys = Object.keys(urnMap);
    if (keys.length === 0) {
      prog.textContent = "Ingen bøker funnet med disse søkekriteriene.";
      return;
    }

    const books = keys.map((k, idx) => ({
      id: idx,
      urn: urnMap[k],
      title: titleMap[k] || "Uten tittel",
      authors: authMap[k] || "",
      year: parseInt(yearMap[k], 10) || 0,
      n_chunks: 100,
      doc_tokens: 1,
    }));

    // Behold bakgrunnsordbok fra nåværende bundle slik at vi slipper å laste 500k på nytt
    const bgLookup = new Map();
    const stopLookup = new Set();
    state.bundle.words.forEach((w) => {
      bgLookup.set(w.w, w.tf_bg);
      if (w.stop) stopLookup.add(w.w);
    });

    const docWordList = Array.from({ length: books.length }, () => new Map());

    if (alsoChunks) {
      for (let i = 0; i < books.length; i++) {
        prog.textContent = `2/3 Henter 300-ords chunks i nettleseren (${i + 1}/${books.length})...`;
        const chResp = await fetch(`https://api.nb.no/dhlab/chunks?urn=${encodeURIComponent(books[i].urn)}&chunk_size=300`);
        const chunks = await chResp.json();
        if (!Array.isArray(chunks) || chunks.length === 0) continue;
        const K = chunks.length;
        books[i].n_chunks = K;
        const wPost = new Map();
        chunks.forEach((ch) => {
          const lc = new Map();
          for (const [rawW, cnt] of Object.entries(ch)) {
            const wl = String(rawW).toLowerCase();
            lc.set(wl, (lc.get(wl) || 0) + Number(cnt));
          }
          lc.forEach((cnt, wl) => {
            if (!wPost.has(wl)) wPost.set(wl, []);
            wPost.get(wl).push(cnt);
          });
        });
        let docSum = 0;
        wPost.forEach((arr, wl) => {
          const tot = arr.reduce((a, b) => a + b, 0);
          if (tot < 5 || wl.length < 2) return;
          docSum += tot;
          const df = arr.length;
          const expDf = K * (1 - Math.exp(-tot / K));
          const dfRatio = expDf > 0 ? df / expDf : 1.0;
          const mu = tot / K;
          const sqSum = arr.reduce((a, b) => a + b * b, 0);
          const fano = mu > 0 ? (sqSum / K - mu * mu) / mu : 1.0;
          docWordList[i].set(wl, { freq: tot, fano, dfRatio, disp: df / K });
        });
        books[i].doc_tokens = Math.max(1, docSum);
      }
    } else {
      prog.textContent = `2/3 Henter frekvenser (cutoff=5) for ${books.length} bøker...`;
      const urns = books.map((b) => b.urn);
      const fResp = await fetch("https://api.nb.no/dhlab/frequencies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urns, cutoff: 5, words: null }),
      });
      const fJson = await fResp.json();
      fJson.forEach((docEntries, i) => {
        if (!Array.isArray(docEntries) || i >= books.length) return;
        let docSum = 0;
        docEntries.forEach((entry) => {
          const wl = String(entry[1]).toLowerCase();
          const c = Number(entry[2]);
          if (wl.length < 2) return;
          docSum += c;
          const prev = docWordList[i].get(wl);
          const newFreq = (prev ? prev.freq : 0) + c;
          docWordList[i].set(wl, { freq: newFreq, fano: 1.8, dfRatio: 0.72, disp: 0.05 });
        });
        books[i].doc_tokens = Math.max(1, docSum);
      });
    }

    prog.textContent = "3/3 Bygger CSR-matrise og kjører SVD...";
    const wordAgg = new Map();
    docWordList.forEach((m) => {
      m.forEach((stats, wl) => {
        if (!wordAgg.has(wl)) wordAgg.set(wl, { nb: 0, cf: 0, disps: [], ratios: [], fanos: [] });
        const a = wordAgg.get(wl);
        a.nb++;
        a.cf += stats.freq;
        a.disps.push(stats.disp);
        a.ratios.push(stats.dfRatio);
        a.fanos.push(stats.fano);
      });
    });

    const wordsMeta = [];
    const wordToId = new Map();
    Array.from(wordAgg.entries())
      .filter(([, a]) => a.nb >= 2)
      .sort((a, b) => b[1].cf - a[1].cf)
      .slice(0, 5000)
      .forEach(([wl, a], idx) => {
        wordToId.set(wl, idx);
        const mid = Math.floor(a.fanos.length / 2);
        wordsMeta.push({
          w: wl,
          tf_bg: bgLookup.get(wl) || a.cf * 250,
          nb: a.nb,
          cf: a.cf,
          md: a.disps.sort()[mid] || 0.05,
          mr: a.ratios.sort()[mid] || 0.75,
          mf: a.fanos.sort()[mid] || 1.7,
          stop: stopLookup.has(wl) ? 1 : 0,
        });
      });

    const rowPtr = [0];
    const colIdx = [];
    const freqs = [];
    const fano100 = [];
    const dfr100 = [];

    for (let i = 0; i < books.length; i++) {
      const entries = [];
      docWordList[i].forEach((st, wl) => {
        if (wordToId.has(wl)) entries.push({ wId: wordToId.get(wl), st });
      });
      entries.sort((a, b) => a.wId - b.wId);
      entries.forEach(({ wId, st }) => {
        colIdx.push(wId);
        freqs.push(st.freq);
        fano100.push(Math.round(st.fano * 100));
        dfr100.push(Math.round(st.dfRatio * 100));
      });
      rowPtr.push(colIdx.length);
    }

    const newBundle = {
      meta: {
        corpus_name: `NB API Korpus (${subject || author || title || "utvalg"})`,
        n_books: books.length,
        n_words: wordsMeta.length,
        nnz: colIdx.length,
        total_bg_sum: state.bundle.meta.total_bg_sum,
      },
      books,
      words: wordsMeta,
      csr: { row_ptr: rowPtr, col_idx: colIdx, freq: freqs, fano_x100: fano100, dfratio_x100: dfr100 },
    };

    state.selectedBookId = 0;
    if (!alsoChunks) document.getElementById("weightMode").value = "ppmi";
    loadBundleIntoMemory(newBundle);
    recomputeAll();
    document.getElementById("apiModal").close();
    prog.textContent = "";
  } catch (err) {
    prog.textContent = "Feil ved API-kall: " + err.message;
  }
}

function bindUIEvents() {
  // Faner
  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      ["tab-map", "tab-axes", "tab-galaxy", "tab-world"].forEach((id) => {
        document.getElementById(id).classList.toggle("hidden", id !== target);
      });
    });
  });

  // Kontroll-slidere som trigger full rekalkulering
  const bindSlider = (id, valId) => {
    const el = document.getElementById(id);
    const vEl = document.getElementById(valId);
    el.addEventListener("input", () => {
      if (vEl) vEl.textContent = el.value;
      recomputeAll();
    });
  };
  bindSlider("minFano", "minFanoVal");
  bindSlider("maxDfRatio", "maxDfRatioVal");
  bindSlider("ppmiAlpha", "ppmiAlphaVal");
  bindSlider("graphTopWords", "graphTopWordsVal");
  bindSlider("graphTopK", "graphTopKVal");

  ["weightMode", "filterStop", "sublinearTf"].forEach((id) => {
    document.getElementById(id).addEventListener("change", () => recomputeAll());
  });

  document.getElementById("axisXSelect").addEventListener("change", (e) => {
    state.axisX = parseInt(e.target.value, 10);
    renderScatterPlot();
  });
  document.getElementById("axisYSelect").addEventListener("change", (e) => {
    state.axisY = parseInt(e.target.value, 10);
    renderScatterPlot();
  });
  document.getElementById("bookSelect").addEventListener("change", (e) => {
    selectBook(parseInt(e.target.value, 10));
  });

  // Klikk i scatter-plottet for å velge bok
  const scatterCanvas = document.getElementById("scatterCanvas");
  scatterCanvas.addEventListener("click", (e) => {
    if (!state.scatterPoints) return;
    const rect = scatterCanvas.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * scatterCanvas.width;
    const my = ((e.clientY - rect.top) / rect.height) * scatterCanvas.height;
    let bestId = -1, bestDist = 400;
    state.scatterPoints.forEach((pt) => {
      const d2 = (pt.x - mx) ** 2 + (pt.y - my) ** 2;
      if (d2 < bestDist) { bestDist = d2; bestId = pt.id; }
    });
    if (bestId >= 0) selectBook(bestId);
  });

  // Galakse-søk og relayout
  document.getElementById("galaxySearch").addEventListener("input", () => renderGalaxyAndClusters());
  document.getElementById("reLayoutBtn").addEventListener("click", () => {
    runForceLayout(state.graphNodes, state.graphEdges, 70);
    renderGalaxyAndClusters();
  });

  // Begrepstabell-søk
  document.getElementById("presetFilter").addEventListener("change", () => renderWordTable());
  document.getElementById("wordTableSearch").addEventListener("input", () => renderWordTable());

  // API-modal
  const modal = document.getElementById("apiModal");
  document.getElementById("openApiModalBtn").addEventListener("click", () => modal.showModal());
  document.getElementById("closeApiModalBtn").addEventListener("click", () => modal.close());
  document.getElementById("runApiFetchBtn").addEventListener("click", () => runLiveApiFetch());
}

window.addEventListener("DOMContentLoaded", init);
