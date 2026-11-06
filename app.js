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

const BBI_DIVERSITY_SET = new Set([
  "moske", "moské", "moskeen", "moskéen", "synagoge", "synagogen", "kirke", "kirken", "kirka", "bedehus", "bedehuset",
  "muslim", "muslimer", "muslimene", "muslimsk", "islam", "imam", "imamen", "koranen", "ramadan", "hijab", "hijaben", "bønnerommet", "profeten",
  "jøde", "jøder", "jødene", "jødisk", "jødedommen", "rabbiner", "toraen", "sabbat", "kippa", "tora",
  "kristen", "kristne", "kristendommen", "jesus", "bibelen", "gud",
  "blind", "blinde", "svaksynt", "svaksynte", "synshemmet", "blindestokk", "blindeskrift", "punktskrift", "førerhund",
  "døv", "døve", "tunghørt", "hørselshemmet", "tegnspråk", "høreapparat",
  "lam", "lamme", "funksjonshemmet", "funksjonshemmede", "handikappet", "rullestol", "rullestolen", "protese", "krykker", "krykkene", "hjelpemidler", "likeverd",
  "homo", "homofil", "homofile", "homofili", "homse", "homser", "lesbisk", "lesbiske", "bifil", "pride", "paraden", "skeiv", "skeive", "streit", "hetero", "transperson", "unormal",
  "samisk", "samiske", "samer", "samene", "same", "joik", "joike", "joiker", "kofte", "kofta", "lavvo", "lavvoen", "reinsdyr", "reinen", "kautokeino", "karasjok", "nordlyset", "skallene", "lasso",
  "kven", "kvensk", "romfolk", "sigøyner", "tater", "tatere",
  "innvandrer", "innvandrere", "innvandring", "flerkulturell", "flyktning", "flyktninger", "flyktningene", "asyl", "asylsøker", "asylsøkere", "asylmottak", "asylmottaket", "udi", "søknaden", "mottaket",
  "flukt", "flykte", "flykter", "flyktet", "hudfarge", "mørkhudet", "mørkhudede", "rasisme", "rasist", "rasister", "rasistisk"
]);

const FOOD_SET = new Set([
  "salt", "pepper", "smør", "sukker", "vann", "mel", "hvetemel", "egg", "melk", "fløte", "olje", "olivenolje",
  "løk", "hvitløk", "poteter", "potet", "tomater", "tomat", "ost", "fisk", "torsk", "laks", "sild", "lutefisk",
  "klippfisk", "reker", "krabbe", "blåskjell", "kjøtt", "oksekjøtt", "svinekjøtt", "lam", "lammekjøtt", "fårikål",
  "pinnekjøtt", "ribbe", "elg", "hjort", "rein", "rype", "kylling", "kalkun", "pølser", "kjøttkaker", "kraft",
  "buljong", "suppe", "saus", "brød", "deig", "deigen", "gjær", "surdeig", "kake", "kaker", "sjokolade", "krem",
  "mandler", "marsipan", "is", "bær", "jordbær", "epler", "sitron", "ingefær", "chili", "karri", "koriander",
  "basilikum", "timian", "persille", "kanel", "kardemomme", "vanilje", "soyasaus", "ris", "pasta", "spaghetti",
  "pizza", "vin", "rødvin", "hvitvin", "øl", "malt", "humle", "akevitt", "kaffe", "ovn", "ovnen", "stekeovn"
]);

const WESTERN_SET = new Set([
  "kane", "morgan", "kanes", "marshal", "stjernen", "revolver", "revolveren", "colt", "colten", "winchester",
  "gevær", "rifle", "hagle", "hylsteret", "avtrekkeren", "kulen", "kuler", "skudd", "krutt", "dynamitt",
  "hest", "hesten", "hestene", "salen", "sporene", "kjerra", "vogna", "diligencen", "toget", "lokomotivet",
  "saloon", "saloonen", "bardisken", "bartenderen", "whisky", "tequila", "øl", "glass", "flaske", "sigar",
  "poker", "kortene", "potten", "rulett", "dollar", "gull", "sølv", "banken", "pengeskapet",
  "sheriff", "sheriffen", "dommeren", "juryen", "retten", "fengsel", "cellen", "galgen", "løkken", "lovløs", "dusør", "pinkerton",
  "mexico", "mexicanske", "mexicanere", "gringo", "texas", "arizona", "sonora", "durango",
  "indianere", "indianerne", "apache", "apachene", "comanche", "høvdingen", "kavaleriet", "fortet", "obersten", "generalen", "soldatene",
  "ørkenen", "prærien", "canyon", "leirbålet", "kveg", "ranchen", "cowboy", "linda", "jesse", "rawson", "custer"
]);

async function init() {
  bindUIEvents();
  await fetchAndLoadPresetBundle("mat_bundle.json");
}

async function fetchAndLoadPresetBundle(filename) {
  try {
    document.getElementById("corpusSubtitle").textContent = `Laster ${filename}...`;
    const resp = await fetch(filename);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} (${filename})`);
    const bundle = await resp.json();
    state.selectedBookId = 0;
    loadBundleIntoMemory(bundle);
    const pf = document.getElementById("presetFilter");
    if (pf) {
      if (filename.includes("mat")) pf.value = "food";
      else if (filename.includes("kane")) pf.value = "western";
      else pf.value = "countries";
    }
    recomputeAll();
  } catch (err) {
    document.getElementById("corpusSubtitle").textContent =
      `Kunne ikke laste ${filename}: ` + err.message;
  }
}

async function handleLocalBundleUpload(file) {
  if (!file) return;
  try {
    document.getElementById("corpusSubtitle").textContent = `Leser lokal fil: ${file.name}...`;
    const text = await file.text();
    const bundle = JSON.parse(text);
    if (!bundle || !bundle.meta || !bundle.csr || !bundle.books || !bundle.words) {
      throw new Error("Ugyldig bundle-format (mangler meta, books, words eller csr).");
    }
    state.selectedBookId = 0;
    loadBundleIntoMemory(bundle);
    const pf = document.getElementById("presetFilter");
    if (pf) {
      const fn = file.name.toLowerCase();
      if (fn.includes("bbi")) pf.value = "bbi_diversity";
      else if (fn.includes("tvedt") || fn.includes("vann")) pf.value = "countries";
      else if (fn.includes("mat")) pf.value = "food";
      else pf.value = "top_burst";
    }
    recomputeAll();
  } catch (err) {
    alert("Kunne ikke lese JSON-bundle: " + err.message);
    document.getElementById("corpusSubtitle").textContent = "Feil ved opplasting: " + err.message;
  }
}

function downloadCurrentBundle() {
  if (!state.bundle) return;
  const jsonStr = JSON.stringify(state.bundle);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeName = (state.bundle.meta.corpus_name || "corpus_bundle")
    .toLowerCase()
    .replace(/[^a-z0-9æøå]+/g, "_")
    .replace(/^_|_$/g, "");
  a.href = url;
  a.download = `${safeName}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
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

  updateEraBuckets(bundle.books);
  populateSelects();
}

function updateEraBuckets(books) {
  const years = books.map((b) => b.year).filter((y) => y > 1500).sort((a, b) => a - b);
  if (years.length === 0) {
    state.eraCuts = [1890, 1917, 1969];
    return;
  }
  const minY = years[0];
  const maxY = years[years.length - 1];
  if (minY >= 1995 && maxY <= 2030) {
    state.eraCuts = [2006, 2012, 2018];
    const leg = document.getElementById("scatterLegend");
    if (leg) {
      leg.innerHTML = `
        <span><i class="legend-dot" style="background:#f472b6"></i>${minY}–2006</span>
        <span><i class="legend-dot" style="background:#fbbf24"></i>2007–2012</span>
        <span><i class="legend-dot" style="background:#34d399"></i>2013–2018</span>
        <span><i class="legend-dot" style="background:#38bdf8"></i>2019–${maxY}</span>
      `;
    }
  } else {
    const q1 = years[Math.floor(years.length * 0.25)];
    const q2 = years[Math.floor(years.length * 0.50)];
    const q3 = years[Math.floor(years.length * 0.75)];
    state.eraCuts = [q1, q2, q3];
    const leg = document.getElementById("scatterLegend");
    if (leg) {
      leg.innerHTML = `
        <span><i class="legend-dot" style="background:#f472b6"></i>≤ ${q1}</span>
        <span><i class="legend-dot" style="background:#fbbf24"></i>${q1 + 1}–${q2}</span>
        <span><i class="legend-dot" style="background:#34d399"></i>${q2 + 1}–${q3}</span>
        <span><i class="legend-dot" style="background:#38bdf8"></i>${q3 + 1}–${maxY}</span>
      `;
    }
  }
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
  const probeSel = document.getElementById("probeCorpusBookSelect");
  if (probeSel) probeSel.innerHTML = `<option value="">— Velg en bok fra korpuset —</option>`;

  state.bundle.books.forEach((b) => {
    const auth = b.authors ? b.authors.split("/")[0].trim() : "";
    const label = `[${b.year || "?"}] ${b.title.slice(0, 62)} (${auth})`;
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = label;
    if (b.id === state.selectedBookId) opt.selected = true;
    bookSel.appendChild(opt);

    if (probeSel) {
      const optP = document.createElement("option");
      optP.value = b.id;
      optP.textContent = label;
      probeSel.appendChild(optP);
    }
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

  // 2. Deflatert potensiterasjon direkte via glissen M * (M^T * u) uten å bygge tett (nBooks x nBooks) Gram-matrise!
  // Dette skalerer lineært O(nnz) slik at 1 800–10 000 bøker går på ~80 ms i nettleseren.
  const kDims = state.kDims;
  const U = [];
  const singularValues = new Float64Array(kDims);
  const tmpW = new Float64Array(nWords);

  for (let k = 0; k < kDims; k++) {
    const u = new Float64Array(nBooks);
    for (let d = 0; d < nBooks; d++) {
      u[d] = Math.sin((d + 1) * (k + 1.37));
    }

    const nextU = new Float64Array(nBooks);
    const nIter = nBooks > 800 ? 18 : 26;
    for (let iter = 0; iter < nIter; iter++) {
      // tmpW = M^T * u
      tmpW.fill(0.0);
      for (let d = 0; d < nBooks; d++) {
        const ud = u[d];
        if (ud === 0.0) continue;
        const end = rowPtr[d + 1];
        for (let idx = rowPtr[d]; idx < end; idx++) {
          const wVal = weights[idx];
          if (wVal > 0) tmpW[colIdx[idx]] += wVal * ud;
        }
      }
      // nextU = M * tmpW
      for (let d = 0; d < nBooks; d++) {
        let sum = 0.0;
        const end = rowPtr[d + 1];
        for (let idx = rowPtr[d]; idx < end; idx++) {
          const wVal = weights[idx];
          if (wVal > 0) sum += wVal * tmpW[colIdx[idx]];
        }
        nextU[d] = sum;
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

    // Rayleigh-kvotient: lambda = ||M^T u||^2 -> sigma = ||M^T u||
    tmpW.fill(0.0);
    for (let d = 0; d < nBooks; d++) {
      const ud = u[d];
      const end = rowPtr[d + 1];
      for (let idx = rowPtr[d]; idx < end; idx++) {
        const wVal = weights[idx];
        if (wVal > 0) tmpW[colIdx[idx]] += wVal * ud;
      }
    }
    let lambda = 0.0;
    for (let w = 0; w < nWords; w++) lambda += tmpW[w] * tmpW[w];
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
  const cuts = state.eraCuts || [2006, 2012, 2018];
  if (!year || year <= cuts[0]) return "#f472b6";
  if (year <= cuts[1]) return "#fbbf24";
  if (year <= cuts[2]) return "#34d399";
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

  // Tegn innsjekket testbok (state.probeOverlay) som gullmarkør hvis aktiv
  if (state.probeOverlay && state.probeOverlay.uProj) {
    const uxProbe = state.probeOverlay.uProj[state.axisX] * sx;
    const uyProbe = state.probeOverlay.uProj[state.axisY] * sy;
    const px = Math.max(18, Math.min(W - 18, toScreenX(uxProbe)));
    const py = Math.max(18, Math.min(H - 18, toScreenY(uyProbe)));
    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, 9.5, 0, Math.PI * 2);
    ctx.fillStyle = "#fde047";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#0f172a";
    ctx.stroke();
    ctx.fillStyle = "#fde047";
    ctx.font = "700 12.5px system-ui, sans-serif";
    ctx.fillText(`★ ${state.probeOverlay.title.slice(0, 38)}`, Math.min(W - 240, px + 13), Math.max(20, py - 10));
    ctx.restore();
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

  // Beregn eksakt glissen cosinus-likhet mellom bok d og alle andre bøker på 0.2 ms
  const dVec = new Float32Array(state.nWords);
  for (let idx = state.rowPtr[d]; idx < state.rowPtr[d + 1]; idx++) {
    dVec[state.colIdx[idx]] = state.weights[idx];
  }

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
    let sparseSim = 0.0;
    for (let idx = state.rowPtr[other]; idx < state.rowPtr[other + 1]; idx++) {
      const wVal = state.weights[idx];
      if (wVal > 0) sparseSim += wVal * dVec[state.colIdx[idx]];
    }
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
  } else if (preset === "food") {
    list = list.filter((x) => FOOD_SET.has(x.w));
    list.sort((a, b) => b.cf - a.cf);
  } else if (preset === "western") {
    list = list.filter((x) => WESTERN_SET.has(x.w));
    list.sort((a, b) => b.cf - a.cf);
  } else if (preset === "countries") {
    list = list.filter((x) => COUNTRIES_SET.has(x.w));
    list.sort((a, b) => b.cf - a.cf);
  } else if (preset === "bbi_diversity") {
    list = list.filter((x) => BBI_DIVERSITY_SET.has(x.w));
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

// =============================================================================
// TAB 5: SJEKK ENKELTBOK MOT KORPUSET (MULTI-VEKTOR DEKOMPONERING)
// =============================================================================

async function probeSearchOrFetchNb() {
  const q = (document.getElementById("probeNbQuery").value || "").trim();
  const statusEl = document.getElementById("probeStatus");
  const resEl = document.getElementById("probeNbResults");
  if (!q) return;

  if (q.startsWith("URN:NBN:")) {
    resEl.innerHTML = "";
    await fetchAndProbeNbUrn(q, q, "", 2000);
    return;
  }

  try {
    statusEl.textContent = `Søker i Nasjonalbiblioteket etter «${q}»...`;
    let cJson = {};
    for (const field of ["title", "author", "freetext"]) {
      const body = { doctype: "digibok", limit: 12 };
      body[field] = q;
      const resp = await fetch("https://api.nb.no/dhlab/build_corpus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      cJson = await resp.json();
      if (cJson.urn && Object.keys(cJson.urn).length > 0) break;
    }

    const urnMap = cJson.urn || {};
    const keys = Object.keys(urnMap);
    if (keys.length === 0) {
      statusEl.textContent = `Fant ingen bøker i NB på «${q}».`;
      resEl.innerHTML = "";
      return;
    }
    statusEl.textContent = `Fant ${keys.length} treff i NB — klikk på en bok for å hente 250-ords chunks og beregne trekkvektorene:`;
    resEl.innerHTML = keys.map((k) => {
      const u = urnMap[k];
      const t = (cJson.title && cJson.title[k]) || "Uten tittel";
      const a = (cJson.authors && cJson.authors[k]) || "";
      const y = parseInt((cJson.year && cJson.year[k]) || 0, 10) || 0;
      const safeT = t.replace(/'/g, "\\'").replace(/"/g, "&quot;");
      const safeA = a.replace(/'/g, "\\'").replace(/"/g, "&quot;");
      return `<div style="padding:0.3rem 0;border-block-end:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
        <span><strong>[${y || "?"}] ${t}</strong> <small style="color:var(--text-muted);">(${a})</small></span>
        <button onclick="fetchAndProbeNbUrn('${u}', '${safeT}', '${safeA}', ${y})">Velg</button>
      </div>`;
    }).join("");
  } catch (err) {
    statusEl.textContent = "Feil ved søk mot NB API: " + err.message;
  }
}

window.fetchAndProbeNbUrn = async function(urn, title, authors, year) {
  const statusEl = document.getElementById("probeStatus");
  try {
    statusEl.textContent = `Henter 250-ords chunks for «${title}» (${urn})...`;
    const resp = await fetch(`https://api.nb.no/dhlab/chunks?urn=${encodeURIComponent(urn)}&chunk_size=250`);
    const chunks = await resp.json();
    if (!Array.isArray(chunks) || chunks.length === 0) {
      statusEl.textContent = `Ingen tekst-chunks tilgjengelig for ${urn}.`;
      return;
    }
    const K = chunks.length;
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

    const wordStats = new Map();
    let docTokens = 0;
    wPost.forEach((arr, wl) => {
      const tot = arr.reduce((a, b) => a + b, 0);
      if (tot < 2 || wl.length < 3) return;
      docTokens += tot;
      const df = arr.length;
      const expDf = K * (1 - Math.exp(-tot / K));
      const dfRatio = expDf > 0 ? df / expDf : 1.0;
      const mu = tot / K;
      const sqSum = arr.reduce((a, b) => a + b * b, 0);
      const fano = mu > 0 ? (sqSum / K - mu * mu) / mu : 1.0;
      wordStats.set(wl, { freq: tot, fano, dfRatio, disp: df / K });
    });

    runMultiVectorProbeAnalysis({
      urn,
      title,
      authors,
      year: year || 2000,
      nChunks: K,
      docTokens: Math.max(1, docTokens),
      wordStats,
    });
  } catch (err) {
    statusEl.textContent = "Feil ved henting av chunks: " + err.message;
  }
};

function probeFromRawTextString(text, title = "Egen tekst / opplastet dokument") {
  const tokens = (text.toLowerCase().match(/[a-zæøå]+/g) || []).filter((w) => w.length >= 2);
  if (tokens.length < 40) {
    document.getElementById("probeStatus").textContent = "Teksten er for kort (legg inn minst 40 ord).";
    return;
  }
  const chunkSize = tokens.length < 1500 ? 100 : 250;
  const K = Math.max(2, Math.ceil(tokens.length / chunkSize));
  const wPost = new Map();
  for (let k = 0; k < K; k++) {
    const slice = tokens.slice(k * chunkSize, (k + 1) * chunkSize);
    const lc = new Map();
    slice.forEach((w) => lc.set(w, (lc.get(w) || 0) + 1));
    lc.forEach((cnt, wl) => {
      if (!wPost.has(wl)) wPost.set(wl, []);
      wPost.get(wl).push(cnt);
    });
  }

  const wordStats = new Map();
  let docTokens = 0;
  wPost.forEach((arr, wl) => {
    const tot = arr.reduce((a, b) => a + b, 0);
    if (tot < 2 || wl.length < 3) return;
    docTokens += tot;
    const df = arr.length;
    const expDf = K * (1 - Math.exp(-tot / K));
    const dfRatio = expDf > 0 ? df / expDf : 1.0;
    const mu = tot / K;
    const sqSum = arr.reduce((a, b) => a + b * b, 0);
    const fano = mu > 0 ? (sqSum / K - mu * mu) / mu : 1.0;
    wordStats.set(wl, { freq: tot, fano, dfRatio, disp: df / K });
  });

  runMultiVectorProbeAnalysis({
    urn: "LOCAL:TEXT",
    title,
    authors: "Opplastet / innlimt",
    year: new Date().getFullYear(),
    nChunks: K,
    docTokens: Math.max(1, docTokens),
    wordStats,
  });
}

function probeFromExistingCorpusBook(bookId) {
  if (bookId === null || bookId === undefined || isNaN(bookId)) return;
  const b = state.bundle.books[bookId];
  if (!b) return;
  const K = Math.max(5, b.n_chunks || 100);
  const wordStats = new Map();
  for (let idx = state.rowPtr[bookId]; idx < state.rowPtr[bookId + 1]; idx++) {
    const wId = state.colIdx[idx];
    const wl = state.bundle.words[wId].w;
    const freq = state.freq[idx];
    const fano = state.fano[idx];
    const dfRatio = state.dfRatio[idx];
    const expDf = K * (1 - Math.exp(-freq / K));
    const disp = Math.min(1.0, (dfRatio * expDf) / K);
    wordStats.set(wl, { freq, fano, dfRatio, disp });
  }
  runMultiVectorProbeAnalysis({
    urn: b.urn,
    title: b.title,
    authors: b.authors,
    year: b.year,
    nChunks: K,
    docTokens: b.doc_tokens,
    wordStats,
    existingBookId: bookId,
  });
}

function runMultiVectorProbeAnalysis(doc) {
  const statusEl = document.getElementById("probeStatus");
  const container = document.getElementById("probeResultsContainer");
  const nWords = state.nWords;
  const nBooks = state.nBooks;
  const alpha = parseFloat(document.getElementById("ppmiAlpha").value) || 0.85;

  // Map korpusets ord til indeks
  const wordToId = new Map();
  let bgSum = 0.0;
  const poweredBg = new Float64Array(nWords);
  for (let w = 0; w < nWords; w++) {
    wordToId.set(state.bundle.words[w].w, w);
    const p = Math.pow(state.wordTfBg[w], alpha);
    poweredBg[w] = p;
    bgSum += p;
  }
  const scaleBg = bgSum * 4.5;

  // Vi bygger TRE distinkte normaliserte trekkvektorer for samme bok:
  // 1) qBurst: ren kapittel-/scene-burst (høy Fano, lav DF/E[DF])
  // 2) qFrame: gjennomgående bakgrunnsramme / kulisse (lav Fano <= 1.35, høy spredning disp)
  // 3) qPpmi:  global PPMI mot nb.totals (uavhengig av Fano)
  const qBurst = new Float32Array(nWords);
  const qFrame = new Float32Array(nWords);
  const qPpmi  = new Float32Array(nWords);
  const qActive = new Float32Array(nWords); // etter gjeldende venstremeny-innstilling (for SVD fold-in)

  const burstWords = [];
  const frameWords = [];
  const ppmiWords  = [];

  const log2DocLen = Math.log2(Math.max(10, doc.docTokens));
  const minFanoSetting = parseFloat(document.getElementById("minFano").value) || 1.15;
  const maxDfrSetting  = parseFloat(document.getElementById("maxDfRatio").value) || 0.92;
  const activeMode     = document.getElementById("weightMode").value;

  doc.wordStats.forEach((st, wl) => {
    const wId = wordToId.get(wl);
    if (wId === undefined) return;
    const isStop = state.wordStop[wId] === 1;
    const tfScale = Math.log2(1.0 + st.freq);
    const log2Pw = Math.log2(poweredBg[wId] / scaleBg);
    const pmi = Math.max(0.0, Math.log2(st.freq) - log2DocLen - log2Pw);

    if (!isStop) {
      // 1. Burst-vektor (Fano >= 1.28 og dfRatio <= 0.88)
      if (st.fano >= 1.28 && st.dfRatio <= 0.88) {
        const wBurst = tfScale * (st.fano - 1.05) * Math.max(0.1, 1.0 - st.dfRatio * 0.8);
        qBurst[wId] = wBurst;
        burstWords.push({ wl, ...st, score: wBurst });
      }
      // 2. Bakgrunnsramme-vektor (jevnt spredt: Fano <= 1.32, dfRatio >= 0.82, minst 3 forekomster)
      if (st.fano <= 1.32 && st.dfRatio >= 0.82 && st.freq >= 3) {
        const wFrame = tfScale * Math.max(0.4, pmi) * (0.5 + st.disp * 4.0);
        qFrame[wId] = wFrame;
        frameWords.push({ wl, ...st, score: wFrame });
      }
      // 3. Global PPMI-vektor
      if (pmi > 0) {
        const wPpmi = tfScale * pmi;
        qPpmi[wId] = wPpmi;
        ppmiWords.push({ wl, ...st, pmi, score: wPpmi });
      }
      // 4. Aktiv vektor for SVD-projeksjon
      if (st.fano >= minFanoSetting && st.dfRatio <= maxDfrSetting) {
        if (activeMode === "burst") {
          qActive[wId] = tfScale * Math.max(0, st.fano - minFanoSetting + 0.15) * Math.max(0.05, 1.0 - st.dfRatio * 0.85);
        } else if (activeMode === "ppmi") {
          qActive[wId] = tfScale * pmi;
        } else {
          qActive[wId] = tfScale * pmi * Math.max(0, st.fano - minFanoSetting + 0.2);
        }
      }
    }
  });

  const l2Norm = (vec) => {
    let s = 0;
    for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
    const inv = s > 1e-12 ? 1.0 / Math.sqrt(s) : 0.0;
    for (let i = 0; i < vec.length; i++) vec[i] *= inv;
  };
  l2Norm(qBurst);
  l2Norm(qFrame);
  l2Norm(qPpmi);
  l2Norm(qActive);

  burstWords.sort((a, b) => b.score - a.score);
  frameWords.sort((a, b) => b.score - a.score);
  ppmiWords.sort((a, b) => b.score - a.score);

  // Fold-in i eksisterende SVD-rom: uProj[k] = (qActive . V[k]) / sigma[k]
  const uProj = new Float64Array(state.kDims);
  for (let k = 0; k < state.kDims; k++) {
    let dot = 0.0;
    const vk = state.V[k];
    for (let w = 0; w < nWords; w++) {
      if (qActive[w] > 0) dot += qActive[w] * vk[w];
    }
    uProj[k] = dot / Math.max(1e-6, state.singularValues[k]);
  }
  state.probeOverlay = { title: doc.title, uProj, doc };
  renderScatterPlot();

  // Finn nærmeste nabo-bøker i korpuset langs hver av de 3 vektorene!
  // For å sammenligne rettferdig tar vi prikkprodukt mot hver boks tilsvarende ordprofil i korpuset
  const rankBooksForVector = (qVec, modeType) => {
    const scores = [];
    for (let d = 0; d < nBooks; d++) {
      if (doc.existingBookId === d) continue;
      let dot = 0.0;
      let dNormSq = 0.0;
      for (let idx = state.rowPtr[d]; idx < state.rowPtr[d + 1]; idx++) {
        const wId = state.colIdx[idx];
        if (state.wordStop[wId]) continue;
        const c = state.freq[idx];
        const fn = state.fano[idx];
        const dfr = state.dfRatio[idx];
        let dv = 0.0;
        if (modeType === "burst") {
          if (fn >= 1.28 && dfr <= 0.88) dv = Math.log2(1 + c) * (fn - 1.05);
        } else if (modeType === "frame") {
          if (fn <= 1.32 && dfr >= 0.82) dv = Math.log2(1 + c);
        } else {
          dv = state.weights[idx];
        }
        if (dv > 0) {
          dNormSq += dv * dv;
          if (qVec[wId] > 0) dot += qVec[wId] * dv;
        }
      }
      const sim = dNormSq > 1e-12 ? dot / Math.sqrt(dNormSq) : 0.0;
      scores.push({ id: d, sim });
    }
    scores.sort((a, b) => b.sim - a.sim);
    return scores.slice(0, 5);
  };

  const topBurstBooks = rankBooksForVector(qBurst, "burst");
  const topFrameBooks = rankBooksForVector(qFrame, "frame");
  const topPpmiBooks  = rankBooksForVector(qPpmi, "ppmi");

  // 4. Dekomponer boka mot Dual-Graph Galaksene (state.graphClusters)
  const clusterHits = state.graphClusters.map((cl) => {
    let score = 0.0;
    const matched = [];
    cl.members.forEach((mIdx) => {
      const nd = state.graphNodes[mIdx];
      const st = doc.wordStats.get(nd.word);
      if (st) {
        const wScore = Math.log2(1 + st.freq) * st.fano;
        score += wScore;
        matched.push({ word: nd.word, freq: st.freq, fano: st.fano, wScore });
      }
    });
    matched.sort((a, b) => b.wScore - a.wScore);
    return { name: cl.name, color: cl.color, score, matched, totalSize: cl.members.length };
  }).filter((c) => c.matched.length > 0).sort((a, b) => b.score - a.score);

  statusEl.innerHTML = `Analyserte <strong>«${doc.title}»</strong> (${doc.nChunks} chunks · ${doc.wordStats.size.toLocaleString("no-NO")} distinkte ord). Boka er også tegnet inn som en <strong>★ gullstjerne</strong> i <em>1. Bok-kart</em>!`;

  const formatBookList = (list) => list.map((it, i) => {
    const b = state.bundle.books[it.id];
    const auth = b.authors ? b.authors.split("/")[0].trim() : "";
    return `<div style="padding:0.28rem 0;border-block-end:1px solid var(--border);font-size:0.8rem;display:flex;justify-content:space-between;gap:0.4rem;">
      <span><strong>${i + 1}. [${b.year}] ${b.title.slice(0, 46)}</strong> <small style="color:var(--text-muted);">(${auth})</small></span>
      <span style="color:var(--accent);font-family:var(--mono);">cos=${it.sim.toFixed(3)}</span>
    </div>`;
  }).join("");

  container.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-block-bottom:0.8rem;">
      <div>
        <span class="tag bridge">★ Plottet i 2D Bok-kart (Fane 1)</span>
        <span style="font-size:0.82rem;color:var(--text-muted);margin-inline-start:0.4rem;">
          Klikk på <strong>1. Bok-kart</strong> for å se hvor «${doc.title.slice(0, 40)}» lander, eller legg den fast inn i korpuset:
        </span>
      </div>
      ${doc.existingBookId === undefined ? `<button id="insertProbeBookBtn" class="primary">+ Legg denne boka inn i korpuset &amp; rekalkuler SVD</button>` : ""}
    </div>

    <div class="split-grid" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));">
      <!-- VEKTOR 1: SCENE / BURST -->
      <div class="axis-card">
        <div class="axis-header">
          <strong style="color:var(--pos);">1. Scene- &amp; Kapittel-burstvektor (q_burst)</strong>
          <span class="sigma-pill">Fano ≥ 1.28</span>
        </div>
        <p style="font-size:0.78rem;color:var(--text-muted);margin-block-start:0;">
          Ord som klumper seg tett i avgrensede kapitler eller scener i boka (lav DF/E[DF], høy Fano):
        </p>
        <div style="margin-block-end:0.6rem;">
          ${burstWords.slice(0, 15).map((x) => `<span class="tag" title="freq=${x.freq}, Fano=${x.fano.toFixed(2)}, DF/E[DF]=${x.dfRatio.toFixed(2)}">${x.wl} <small>(${x.freq} · F:${x.fano.toFixed(1)})</small></span>`).join("") || "<em style='font-size:0.8rem;color:var(--text-muted);'>Ingen tydelige burst-ord i felles vokabular</em>"}
        </div>
        <div style="font-size:0.8rem;font-weight:600;margin-block-end:0.25rem;">Nærmeste bøker i korpuset langs Scene-burstvektoren:</div>
        ${formatBookList(topBurstBooks)}
      </div>

      <!-- VEKTOR 2: BAKGRUNNSRAMME / KULISSE -->
      <div class="axis-card">
        <div class="axis-header">
          <strong style="color:var(--warn);">2. Bakgrunnsramme- &amp; Kulissevektor (q_frame)</strong>
          <span class="sigma-pill">Fano ≈ 1.0 · Høy disp</span>
        </div>
        <p style="font-size:0.78rem;color:var(--text-muted);margin-block-start:0;">
          Innholdsord som drysses jevnt utover hele boka som ureflektert sjangerbakgrunn eller kulisse:
        </p>
        <div style="margin-block-end:0.6rem;">
          ${frameWords.slice(0, 15).map((x) => `<span class="tag" title="freq=${x.freq}, disp=${(x.disp*100).toFixed(1)}%, Fano=${x.fano.toFixed(2)}">${x.wl} <small>(${(x.disp*100).toFixed(0)}% sider · F:${x.fano.toFixed(2)})</small></span>`).join("") || "<em style='font-size:0.8rem;color:var(--text-muted);'>Få jevnt spredte bakgrunnsord</em>"}
        </div>
        <div style="font-size:0.8rem;font-weight:600;margin-block-end:0.25rem;">Nærmeste bøker i korpuset langs Bakgrunnsrammen:</div>
        ${formatBookList(topFrameBooks)}
      </div>

      <!-- VEKTOR 3: GLOBAL PPMI & SVD-AKSER -->
      <div class="axis-card">
        <div class="axis-header">
          <strong style="color:var(--accent);">3. Global Sjeldenhetsvektor (q_PPMI mot nb.totals)</strong>
          <span class="sigma-pill">PPMI_α</span>
        </div>
        <p style="font-size:0.78rem;color:var(--text-muted);margin-block-start:0;">
          Ord som skiller boka sterkest fra norsk normalprosa (nb.totals), uavhengig av hvor i boka de står:
        </p>
        <div style="margin-block-end:0.6rem;">
          ${ppmiWords.slice(0, 15).map((x) => `<span class="tag" title="freq=${x.freq}, PPMI=${x.pmi.toFixed(1)}">${x.wl} <small>(${x.freq} · PPMI:${x.pmi.toFixed(1)})</small></span>`).join("")}
        </div>
        <div style="font-size:0.8rem;font-weight:600;margin-block-end:0.25rem;">Nærmeste bøker i korpuset langs Global PPMI:</div>
        ${formatBookList(topPpmiBooks)}
      </div>

      <!-- VEKTOR 4: DUAL-GRAPH GALAKSE-PROFIL -->
      <div class="axis-card">
        <div class="axis-header">
          <strong style="color:#a78bfa;">4. Dual-Graph Galakse-profil (L(G)-klynger)</strong>
          <span class="sigma-pill">${clusterHits.length} aktive klynger</span>
        </div>
        <p style="font-size:0.78rem;color:var(--text-muted);margin-block-start:0;">
          Slik fordeler boka seg utover korpusets overlappende ordgalakser (fra Fane 3):
        </p>
        <div>
          ${clusterHits.slice(0, 6).map((cl) => `
            <div style="padding:0.32rem 0;border-block-end:1px solid var(--border);font-size:0.8rem;">
              <div style="display:flex;justify-content:space-between;">
                <strong style="color:${cl.color};">● ${cl.name}</strong>
                <span style="color:var(--text-muted);">${cl.matched.length}/${cl.totalSize} ord · styrke ${cl.score.toFixed(1)}</span>
              </div>
              <div style="margin-block-top:0.2rem;">
                ${cl.matched.slice(0, 8).map((m) => `<span class="tag">${m.word} (${m.freq})</span>`).join("")}
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    </div>
  `;

  const insBtn = document.getElementById("insertProbeBookBtn");
  if (insBtn) {
    insBtn.addEventListener("click", () => insertCurrentProbeIntoCorpus());
  }
}

function insertCurrentProbeIntoCorpus() {
  if (!state.probeOverlay || !state.probeOverlay.doc) return;
  const doc = state.probeOverlay.doc;
  const b = state.bundle;
  const newId = b.books.length;

  const wordToId = new Map();
  b.words.forEach((w, idx) => wordToId.set(w.w, idx));

  const entries = [];
  doc.wordStats.forEach((st, wl) => {
    if (wordToId.has(wl)) entries.push({ wId: wordToId.get(wl), st });
  });
  entries.sort((a, c) => a.wId - c.wId);

  b.books.push({
    id: newId,
    urn: doc.urn,
    title: `★ ${doc.title}`,
    authors: doc.authors || "",
    year: doc.year || 2020,
    n_chunks: doc.nChunks,
    doc_tokens: doc.docTokens,
  });

  entries.forEach(({ wId, st }) => {
    b.csr.col_idx.push(wId);
    b.csr.freq.push(st.freq);
    b.csr.fano_x100.push(Math.round(st.fano * 100));
    b.csr.dfratio_x100.push(Math.round(st.dfRatio * 100));
  });
  b.csr.row_ptr.push(b.csr.col_idx.length);
  b.meta.n_books = b.books.length;
  b.meta.nnz = b.csr.col_idx.length;

  state.selectedBookId = newId;
  state.probeOverlay = null;
  loadBundleIntoMemory(b);
  recomputeAll();
  // Bytt til Fane 1 så brukeren ser boka valgt i kartet og inspektøren
  document.querySelector('nav.tabs button[data-tab="tab-map"]').click();
}

// =============================================================================
// EKSPORT AV VEKTORMODELL (PARTIELL FUNKSJON / KOMPRIMERT SVD / GALAKSE-TABELL / TENSOR)
// =============================================================================

function computeChannelWeightsForCorpus(channel) {
  const nBooks = state.nBooks;
  const nWords = state.nWords;
  const rowPtr = state.rowPtr;
  const colIdx = state.colIdx;
  const freq = state.freq;
  const fano = state.fano;
  const dfRatio = state.dfRatio;
  const filterStop = document.getElementById("filterStop").checked;
  const sublinearTf = document.getElementById("sublinearTf").checked;
  const minFano = parseFloat(document.getElementById("minFano").value) || 1.15;
  const maxDfRatio = parseFloat(document.getElementById("maxDfRatio").value) || 0.92;
  const alpha = parseFloat(document.getElementById("ppmiAlpha").value) || 0.85;

  const log2Pw = new Float32Array(nWords);
  let bgSum = 0.0;
  const powered = new Float64Array(nWords);
  for (let w = 0; w < nWords; w++) {
    const p = alpha === 1.0 ? state.wordTfBg[w] : Math.pow(state.wordTfBg[w], alpha);
    powered[w] = p;
    bgSum += p;
  }
  const scaleBg = alpha === 1.0 ? Math.max(bgSum, state.bundle.meta.total_bg_sum) : bgSum * 4.5;
  for (let w = 0; w < nWords; w++) {
    log2Pw[w] = Math.log2(powered[w] / scaleBg);
  }

  const outWeights = new Float32Array(state.nnz);
  for (let d = 0; d < nBooks; d++) {
    const start = rowPtr[d];
    const end = rowPtr[d + 1];
    const log2DocLen = Math.log2(state.docTokens[d]);
    let normSq = 0.0;

    for (let idx = start; idx < end; idx++) {
      const w = colIdx[idx];
      if (filterStop && state.wordStop[w]) continue;
      const c = freq[idx];
      const fn = fano[idx];
      const dfr = dfRatio[idx];
      const tfScale = sublinearTf ? Math.log2(1.0 + c) : 1.0;
      const pmi = Math.max(0.0, Math.log2(c) - log2DocLen - log2Pw[w]);
      let val = 0.0;

      if (channel === "burst") {
        if (fn >= minFano && dfr <= maxDfRatio) {
          const excessFano = Math.max(0.0, fn - minFano + 0.15);
          const conc = Math.max(0.05, 1.0 - dfr * 0.85);
          val = tfScale * excessFano * conc;
        }
      } else if (channel === "ppmi") {
        val = tfScale * pmi;
      } else if (channel === "frame") {
        if (fn <= 1.32 && dfr >= 0.82 && c >= 3) {
          val = tfScale * Math.max(0.35, pmi);
        }
      } else {
        // hybrid
        if (fn >= minFano && dfr <= maxDfRatio && pmi > 0) {
          val = tfScale * pmi * Math.max(0.0, fn - minFano + 0.2);
        }
      }

      outWeights[idx] = val;
      normSq += val * val;
    }

    const invNorm = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 0.0;
    for (let idx = start; idx < end; idx++) {
      outWeights[idx] *= invNorm;
    }
  }
  return outWeights;
}

function computeSvdFromSparseWeights(wArr, kDims = 10) {
  const nBooks = state.nBooks;
  const nWords = state.nWords;
  const rowPtr = state.rowPtr;
  const colIdx = state.colIdx;
  const U = [];
  const singularValues = new Float64Array(kDims);
  const tmpW = new Float64Array(nWords);

  for (let k = 0; k < kDims; k++) {
    const u = new Float64Array(nBooks);
    for (let d = 0; d < nBooks; d++) u[d] = Math.sin((d + 1) * (k + 1.37));
    const nextU = new Float64Array(nBooks);
    const nIter = nBooks > 800 ? 18 : 26;

    for (let iter = 0; iter < nIter; iter++) {
      tmpW.fill(0.0);
      for (let d = 0; d < nBooks; d++) {
        const ud = u[d];
        if (ud === 0.0) continue;
        for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
          const wVal = wArr[idx];
          if (wVal > 0) tmpW[colIdx[idx]] += wVal * ud;
        }
      }
      for (let d = 0; d < nBooks; d++) {
        let sum = 0.0;
        for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
          const wVal = wArr[idx];
          if (wVal > 0) sum += wVal * tmpW[colIdx[idx]];
        }
        nextU[d] = sum;
      }
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

    tmpW.fill(0.0);
    for (let d = 0; d < nBooks; d++) {
      const ud = u[d];
      for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
        const wVal = wArr[idx];
        if (wVal > 0) tmpW[colIdx[idx]] += wVal * ud;
      }
    }
    let lambda = 0.0;
    for (let w = 0; w < nWords; w++) lambda += tmpW[w] * tmpW[w];
    singularValues[k] = Math.sqrt(Math.max(1e-9, lambda));
    if (k === 0) {
      let s = 0;
      for (let d = 0; d < nBooks; d++) s += u[d];
      if (s < 0) for (let d = 0; d < nBooks; d++) u[d] = -u[d];
    }
    U.push(u);
  }

  const V = [];
  for (let k = 0; k < kDims; k++) {
    const v = new Float64Array(nWords);
    const u = U[k];
    const invSigma = 1.0 / singularValues[k];
    for (let d = 0; d < nBooks; d++) {
      const ud = u[d] * invSigma;
      for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
        const wVal = wArr[idx];
        if (wVal > 0) v[colIdx[idx]] += wVal * ud;
      }
    }
    V.push(v);
  }
  return { U, V, singularValues };
}

function updateExportHint() {
  const vecType = document.getElementById("exportVectorType").value;
  const fmt = document.getElementById("exportFormatType").value;
  const hintEl = document.getElementById("exportFormatHint");
  if (!hintEl) return;

  if (fmt === "partial_jsonl") {
    hintEl.innerHTML = `<strong>Partiell funksjon (JSONL):</strong> Hver bok eksporteres som en ord-ordbok <code>{ord: vekt}</code> uten nuller (L2-normalisert). Prikkprodukt mellom to bøkers ordbøker gir eksakt cosinus-likhet, og nye bøker utenfra kan sammenlignes direkte ved å slå opp felles ord.`;
  } else if (fmt === "dense_svd_csv") {
    hintEl.innerHTML = `<strong>Komprimert tett SVD-tabell (CSV, N × 10):</strong> Hver bok får 10 tette koordinater (<code>svd_1 .. svd_10</code>) beregnet fra valgt vektortype (<code>${vecType}</code>). Perfekt for intern klynging (<code>KMeans</code>, <code>HDBSCAN</code>, <code>UMAP</code>) og cosinus-anbefalinger mellom bøkene i korpuset.`;
  } else if (fmt === "dense_galaxies_csv") {
    hintEl.innerHTML = `<strong>Tolkbar tett Galakse-tabell (CSV, N × ${state.graphClusters.length} klynger):</strong> Hver kolonne er én av korpusets navngitte Dual-Graph-ordgalakser (f.eks. <code>${state.graphClusters.slice(0, 2).map((c) => c.name).join(", ")}</code>). Gir en komprimert tett tabell der hver dimensjon har en direkte tematisk tolkning!`;
  } else {
    hintEl.innerHTML = `<strong>Komprimert SVD-modell med partiell ord-basis (JSON):</strong> Inneholder både bøkenes 10 SVD-koordinater og den partielle funksjonen <code>word_basis: {ord: [v1..v10]}</code> (${"$V_k \\Sigma_k^{-1/2}$"}), slik at man også kan projisere nye bøker utenfra inn i det komprimerte rommet.`;
  }
}

function triggerBlobDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function runExportVectors() {
  if (!state.bundle) return;
  const vecType = document.getElementById("exportVectorType").value;
  const fmt = document.getElementById("exportFormatType").value;
  const topK = parseInt(document.getElementById("exportTopKWords").value, 10) || 0;

  const baseName = (state.bundle.meta.corpus_name || "corpus")
    .toLowerCase()
    .replace(/[^a-z0-9æøå]+/g, "_")
    .replace(/^_|_$/g, "");

  const nBooks = state.nBooks;
  const rowPtr = state.rowPtr;
  const colIdx = state.colIdx;
  const words = state.bundle.words;
  const books = state.bundle.books;

  if (fmt === "partial_jsonl") {
    const lines = [];
    if (vecType === "tensor") {
      const wBurst = computeChannelWeightsForCorpus("burst");
      const wFrame = computeChannelWeightsForCorpus("frame");
      const wPpmi  = computeChannelWeightsForCorpus("ppmi");
      const wHyb   = computeChannelWeightsForCorpus("hybrid");

      for (let d = 0; d < nBooks; d++) {
        const b = books[d];
        const items = [];
        for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
          const vb = wBurst[idx], vf = wFrame[idx], vp = wPpmi[idx], vh = wHyb[idx];
          const maxV = Math.max(vb, vf, vp, vh);
          if (maxV > 1e-4) {
            items.push({
              w: words[colIdx[idx]].w,
              maxV,
              tuple: [Number(vb.toFixed(4)), Number(vf.toFixed(4)), Number(vp.toFixed(4)), Number(vh.toFixed(4))],
            });
          }
        }
        items.sort((a, c) => c.maxV - a.maxV);
        const kept = topK > 0 ? items.slice(0, topK) : items;
        const tensorObj = {};
        kept.forEach((it) => { tensorObj[it.w] = it.tuple; });
        lines.push(JSON.stringify({
          id: b.id, urn: b.urn, year: b.year, title: b.title, authors: b.authors,
          channels: ["burst", "frame", "ppmi", "hybrid"],
          tensor: tensorObj,
        }));
      }
      triggerBlobDownload(lines.join("\n"), `${baseName}_tensor_partial.jsonl`, "application/x-ndjson");
    } else {
      const wArr = computeChannelWeightsForCorpus(vecType);
      for (let d = 0; d < nBooks; d++) {
        const b = books[d];
        const items = [];
        for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
          const val = wArr[idx];
          if (val > 1e-4) items.push({ w: words[colIdx[idx]].w, val });
        }
        items.sort((a, c) => c.val - a.val);
        const kept = topK > 0 ? items.slice(0, topK) : items;
        // Re-normaliser hvis vi trunkerer til topK slik at ||v||_2 = 1
        let normSq = 0.0;
        kept.forEach((it) => { normSq += it.val * it.val; });
        const inv = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 1.0;
        const vecObj = {};
        kept.forEach((it) => { vecObj[it.w] = Number((it.val * inv).toFixed(5)); });
        lines.push(JSON.stringify({
          id: b.id, urn: b.urn, year: b.year, title: b.title, authors: b.authors,
          vector_type: vecType,
          vector: vecObj,
        }));
      }
      triggerBlobDownload(lines.join("\n"), `${baseName}_${vecType}_partial.jsonl`, "application/x-ndjson");
    }
  } else if (fmt === "dense_svd_csv" || fmt === "svd_model_json") {
    const ch = vecType === "tensor" ? "hybrid" : vecType;
    const wArr = computeChannelWeightsForCorpus(ch);
    const { U, V, singularValues } = computeSvdFromSparseWeights(wArr, state.kDims);
    const kDims = state.kDims;

    if (fmt === "dense_svd_csv") {
      const header = ["id", "urn", "year", "title", "authors"];
      for (let k = 1; k <= kDims; k++) header.push(`${ch}_svd_${k}`);
      const rows = [header.join(",")];

      for (let d = 0; d < nBooks; d++) {
        const b = books[d];
        const coords = new Float64Array(kDims);
        let normSq = 0.0;
        for (let k = 0; k < kDims; k++) {
          const val = U[k][d] * Math.sqrt(singularValues[k]);
          coords[k] = val;
          normSq += val * val;
        }
        const inv = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 1.0;
        const esc = (s) => `"${String(s || "").replace(/"/g, '""')}"`;
        const row = [b.id, esc(b.urn), b.year || 0, esc(b.title), esc(b.authors)];
        for (let k = 0; k < kDims; k++) row.push((coords[k] * inv).toFixed(5));
        rows.push(row.join(","));
      }
      triggerBlobDownload(rows.join("\n"), `${baseName}_${ch}_svd10.csv`, "text/csv;charset=utf-8");
    } else {
      const bookVectors = [];
      for (let d = 0; d < nBooks; d++) {
        const b = books[d];
        const coords = [];
        let normSq = 0.0;
        for (let k = 0; k < kDims; k++) {
          const val = U[k][d] * Math.sqrt(singularValues[k]);
          coords.push(val);
          normSq += val * val;
        }
        const inv = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 1.0;
        bookVectors.push({
          id: b.id, urn: b.urn, year: b.year, title: b.title, authors: b.authors,
          svd_vec: coords.map((x) => Number((x * inv).toFixed(5))),
        });
      }
      const wordBasis = {};
      for (let w = 0; w < state.nWords; w++) {
        if (state.wordStop[w]) continue;
        let hasVal = false;
        const proj = [];
        for (let k = 0; k < kDims; k++) {
          const pv = V[k][w] / Math.sqrt(Math.max(1e-6, singularValues[k]));
          if (Math.abs(pv) > 1e-4) hasVal = true;
          proj.push(Number(pv.toFixed(5)));
        }
        if (hasVal) wordBasis[words[w].w] = proj;
      }
      const modelObj = {
        corpus_name: state.bundle.meta.corpus_name,
        vector_type: ch,
        k_dims: kDims,
        singular_values: Array.from(singularValues).map((x) => Number(x.toFixed(4))),
        books: bookVectors,
        word_basis_partial_fn: wordBasis,
      };
      triggerBlobDownload(JSON.stringify(modelObj, null, 2), `${baseName}_${ch}_svd_model.json`, "application/json");
    }
  } else if (fmt === "dense_galaxies_csv") {
    const ch = vecType === "tensor" ? "hybrid" : vecType;
    const wArr = computeChannelWeightsForCorpus(ch);
    const clusters = state.graphClusters;
    const colWordSets = clusters.map((cl) => {
      const s = new Set();
      cl.members.forEach((mIdx) => s.add(state.graphNodes[mIdx].wId));
      return s;
    });

    const cleanColName = (name, idx) =>
      `galakse_${idx + 1}_` + name.toLowerCase().replace(/[^a-z0-9æøå]+/g, "_").replace(/^_|_$/g, "");
    const header = ["id", "urn", "year", "title", "authors", ...clusters.map((c, i) => cleanColName(c.name, i))];
    const rows = [header.join(",")];

    for (let d = 0; d < nBooks; d++) {
      const b = books[d];
      const scores = new Float64Array(clusters.length);
      for (let idx = rowPtr[d]; idx < rowPtr[d + 1]; idx++) {
        const val = wArr[idx];
        if (val <= 0) continue;
        const wId = colIdx[idx];
        for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
          if (colWordSets[cIdx].has(wId)) scores[cIdx] += val;
        }
      }
      let normSq = 0.0;
      for (let cIdx = 0; cIdx < clusters.length; cIdx++) normSq += scores[cIdx] * scores[cIdx];
      const inv = normSq > 1e-12 ? 1.0 / Math.sqrt(normSq) : 1.0;

      const esc = (s) => `"${String(s || "").replace(/"/g, '""')}"`;
      const row = [b.id, esc(b.urn), b.year || 0, esc(b.title), esc(b.authors)];
      for (let cIdx = 0; cIdx < clusters.length; cIdx++) row.push((scores[cIdx] * inv).toFixed(4));
      rows.push(row.join(","));
    }
    triggerBlobDownload(rows.join("\n"), `${baseName}_${ch}_galaxies.csv`, "text/csv;charset=utf-8");
  }

  document.getElementById("exportModal").close();
}

function bindUIEvents() {
  // Faner
  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      ["tab-map", "tab-axes", "tab-galaxy", "tab-world", "tab-probe"].forEach((id) => {
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

  // Korpus-velger, lokal JSON-opplasting (knapp + drag-and-drop) og nedlasting
  const presetSel = document.getElementById("presetBundleSelect");
  if (presetSel) {
    presetSel.addEventListener("change", (e) => fetchAndLoadPresetBundle(e.target.value));
  }
  const uploadBtn = document.getElementById("uploadBundleBtn");
  const uploadInput = document.getElementById("uploadBundleInput");
  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener("click", () => uploadInput.click());
    uploadInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleLocalBundleUpload(e.target.files[0]);
        e.target.value = "";
      }
    });
  }
  const downloadBtn = document.getElementById("downloadBundleBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => downloadCurrentBundle());
  }
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0];
      if (f.name.endsWith(".json")) handleLocalBundleUpload(f);
    }
  });

  // API-modal
  const modal = document.getElementById("apiModal");
  document.getElementById("openApiModalBtn").addEventListener("click", () => modal.showModal());
  document.getElementById("closeApiModalBtn").addEventListener("click", () => modal.close());
  document.getElementById("runApiFetchBtn").addEventListener("click", () => runLiveApiFetch());

  // Eksport-modal for vektormodeller
  const exportModal = document.getElementById("exportModal");
  const openExportBtn = document.getElementById("openExportModalBtn");
  const closeExportBtn = document.getElementById("closeExportModalBtn");
  const runExportBtn = document.getElementById("runExportVectorsBtn");
  if (exportModal && openExportBtn) {
    openExportBtn.addEventListener("click", () => {
      updateExportHint();
      exportModal.showModal();
    });
    closeExportBtn.addEventListener("click", () => exportModal.close());
    runExportBtn.addEventListener("click", () => runExportVectors());
    document.getElementById("exportVectorType").addEventListener("change", () => updateExportHint());
    document.getElementById("exportFormatType").addEventListener("change", () => updateExportHint());
  }

  // Fallback for closedby="any" i nettlesere uten native dialog.closedBy (jf. modern-web-guidance: dialog)
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dlg) => {
      const r = dlg.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dlg of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dlg.addEventListener("pointerdown", (e) => { pressedOutside = outside(e, dlg); });
      dlg.addEventListener("pointerup", (e) => { if (pressedOutside && outside(e, dlg)) dlg.close(); });
    }
  }

  // Fane 5: Sjekk enkeltbok (Multi-vektor)
  const probeNbBtn = document.getElementById("probeNbSearchBtn");
  const probeNbInp = document.getElementById("probeNbQuery");
  if (probeNbBtn && probeNbInp) {
    probeNbBtn.addEventListener("click", () => probeSearchOrFetchNb());
    probeNbInp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") probeSearchOrFetchNb();
    });
  }
  const probeFileBtn = document.getElementById("probeFileBtn");
  const probeFileInp = document.getElementById("probeFileInput");
  if (probeFileBtn && probeFileInp) {
    probeFileBtn.addEventListener("click", () => probeFileInp.click());
    probeFileInp.addEventListener("change", async (e) => {
      if (e.target.files && e.target.files[0]) {
        const f = e.target.files[0];
        const txt = await f.text();
        probeFromRawTextString(txt, f.name);
        e.target.value = "";
      }
    });
  }
  const probeCorpSel = document.getElementById("probeCorpusBookSelect");
  if (probeCorpSel) {
    probeCorpSel.addEventListener("change", (e) => {
      if (e.target.value !== "") probeFromExistingCorpusBook(parseInt(e.target.value, 10));
    });
  }
  const probeTextBtn = document.getElementById("probeRunTextBtn");
  if (probeTextBtn) {
    probeTextBtn.addEventListener("click", () => {
      const txt = document.getElementById("probeRawText").value || "";
      probeFromRawTextString(txt, "Innlimt tekst");
    });
  }
}

window.addEventListener("DOMContentLoaded", init);
