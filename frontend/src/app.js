/* ── Config ──────────────────────────────────────────────────────────────── */
const PREDICT_URL  = '/api/predict/';
const HISTORY_URL  = '/api/history/';

const CLASS_COLORS = {
  'N (Normale)':            '#0284c7',
  'S (Sopraventricolare)':  '#7c3aed',
  'V (Ventricolare)':       '#dc2626',
  'F (Fusion)':             '#d97706',
  'Q (Non classificabile)': '#6b7280',
};
const DEFAULT_COLOR = '#6b7280';

// Mappa etichetta numerica MIT-BIH → stringa leggibile
const MIT_LABEL_MAP = {
  '0': 'N (Normale)',
  '1': 'S (Sopraventricolare)',
  '2': 'V (Ventricolare)',
  '3': 'F (Fusion)',
  '4': 'Q (Non classificabile)',
};

/* ── Stato ground truth corrente ─────────────────────────────────────────── */
// Stringa o null — viene passato all'API e usato nel banner risultati
let currentGroundTruth = null;

function setGroundTruth(value) {
  // Normalizza: se è un numero MIT-BIH (es. "2") → stringa leggibile
  const normalized = MIT_LABEL_MAP[String(value)] || value || null;
  currentGroundTruth = normalized;

  const badge = document.getElementById('gt-value');
  if (normalized) {
    badge.textContent = normalized;
    badge.className   = 'gt-badge gt-auto';
  } else {
    badge.textContent = 'Non disponibile';
    badge.className   = 'gt-badge';
  }
}

function clearGroundTruth() {
  currentGroundTruth = null;
  const badge = document.getElementById('gt-value');
  badge.textContent = 'Non disponibile';
  badge.className   = 'gt-badge';
}

/* ── Nav / tabs ──────────────────────────────────────────────────────────── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'history') loadHistory();
  });
});

/* ── Input tabs ──────────────────────────────────────────────────────────── */
document.querySelectorAll('.input-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('input-' + tab.dataset.input).classList.add('active');
    // Quando si torna al manuale, il ground truth non è disponibile
    if (tab.dataset.input === 'manual') clearGroundTruth();
  });
});

/* ── Sample counter ──────────────────────────────────────────────────────── */
const signalText  = document.getElementById('signal-text');
const sampleCount = document.getElementById('sample-count');

signalText.addEventListener('input', () => {
  const vals = parseSignalText(signalText.value);
  sampleCount.textContent = `${vals.length} / 187 campioni`;
  sampleCount.style.color = vals.length === 187 ? 'var(--accent)' : 'var(--text-dim)';
  if (vals.length === 187) drawECG(vals);
});

function parseSignalText(txt) {
  return txt.split(/[\s,;]+/).map(Number).filter(v => !isNaN(v) && String(v) !== '');
}

/* ── CSV upload ──────────────────────────────────────────────────────────── */
const dropZone = document.getElementById('drop-zone');
const csvInput = document.getElementById('csv-file');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  handleFile(e.dataTransfer.files[0]);
});
csvInput.addEventListener('change', () => handleFile(csvInput.files[0]));

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const line = e.target.result.split('\n')[0];
    let vals = line.split(/[\s,;]+/).map(Number).filter(v => !isNaN(v) && String(v) !== '');
    let label = null;
    if (vals.length === 188) {
      label = String(vals[187]);  // 188° valore = etichetta MIT-BIH (0-4)
      vals  = vals.slice(0, 187);
    }
    if (vals.length === 187) {
      setSignalNoTabSwitch(vals);
      dropZone.querySelector('.drop-text').textContent = `✓ ${file.name} caricato (187 campioni)`;
      if (label !== null) setGroundTruth(label);
      else clearGroundTruth();
    } else {
      alert(`Attese 187 valori, trovati ${vals.length}. Controlla il file.`);
    }
  };
  reader.readAsText(file);
}

/* ── Random signal ───────────────────────────────────────────────────────── */
document.getElementById('btn-random').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/history/random');
    if (!res.ok) throw new Error('Nessun campione disponibile');
    const detail = await res.json();
    setSignalNoTabSwitch(detail.signal);
    if (detail.ground_truth) setGroundTruth(detail.ground_truth);
    else clearGroundTruth();
  } catch (err) {
    alert('Errore: ' + err.message);
  }
});

function generateDemoSignal() {
  return Array.from({ length: 187 }, (_, i) => {
    const t = i / 187;
    let v = 0.05 * Math.sin(2 * Math.PI * t * 3);
    if (i > 60 && i < 65)   v += 0.15;
    if (i > 75 && i < 95)   v += Math.sin((i - 75) / 20 * Math.PI) * 0.9;
    if (i > 90 && i < 100)  v -= 0.3;
    if (i > 110 && i < 130) v += 0.25 * Math.sin((i - 110) / 20 * Math.PI);
    v += (Math.random() - 0.5) * 0.02;
    return +v.toFixed(4);
  });
}

/* ── setSignal: usato dallo storico ─────────────────────────────────────── */
function setSignal(vals, gt) {
  signalText.value = vals.join(', ');
  sampleCount.textContent = `${vals.length} / 187 campioni`;
  sampleCount.style.color = 'var(--accent)';
  drawECG(vals);
  document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-input="manual"]').classList.add('active');
  document.getElementById('input-manual').classList.add('active');
  if (gt) setGroundTruth(gt);
  else clearGroundTruth();
}

function setSignalNoTabSwitch(vals) {
  signalText.value = vals.join(', ');
  sampleCount.textContent = `${vals.length} / 187 campioni`;
  sampleCount.style.color = 'var(--accent)';
  drawECG(vals);
}

/* ── ECG canvas ──────────────────────────────────────────────────────────── */
function drawECG(vals) {
  const preview = document.getElementById('signal-preview');
  const canvas  = document.getElementById('ecg-canvas');
  preview.classList.remove('hidden');

  const dpr = window.devicePixelRatio || 1;
  const W   = preview.clientWidth || 600;
  const H   = 80;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const min   = Math.min(...vals);
  const max   = Math.max(...vals);
  const range = max - min || 1;
  const pad   = 6;

  ctx.beginPath();
  ctx.strokeStyle = '#0284c7';
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = 'rgba(2,132,199,.3)';
  ctx.shadowBlur  = 4;

  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * (W - pad * 2) + pad;
    const y = H - pad - ((v - min) / range) * (H - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

/* ── Predict ─────────────────────────────────────────────────────────────── */
const btnPredict = document.getElementById('btn-predict');
btnPredict.addEventListener('click', runPrediction);

async function runPrediction() {
  const vals = parseSignalText(signalText.value);

  if (vals.length !== 187) {
    alert('Inserisci esattamente 187 campioni numerici.');
    return;
  }

  setLoading(true);
  try {
    const body = { signal: vals };
    if (currentGroundTruth) body.ground_truth = currentGroundTruth;

    const res = await fetch(PREDICT_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Errore ${res.status}`);
    const data = await res.json();
    renderResults(data, currentGroundTruth);
  } catch (err) {
    alert('Errore durante la predizione: ' + err.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  btnPredict.disabled = on;
  btnPredict.querySelector('.btn-text').classList.toggle('hidden', on);
  btnPredict.querySelector('.btn-loader').classList.toggle('hidden', !on);
}

/* ── Render results ──────────────────────────────────────────────────────── */
function renderResults(data, groundTruth) {
  document.getElementById('results-section').classList.remove('hidden');
  renderModel('cnn', data.cnn);
  renderModel('rf',  data.rf);

  // Banner accordo/disaccordo
  const cnnDiag = data.cnn.diagnosi;
  const rfDiag  = data.rf.diagnosi;
  const banner  = document.getElementById('agreement-banner');
  if (cnnDiag === rfDiag) {
    banner.className   = 'agreement-banner agree';
    banner.textContent = `✓ Entrambi i modelli concordano: ${cnnDiag}`;
  } else {
    banner.className   = 'agreement-banner disagree';
    banner.textContent = `⚠ Disaccordo diagnostico — CNN: ${cnnDiag}  ·  RF: ${rfDiag}`;
  }

  // Banner ground truth vs predizioni (solo se GT disponibile)
  const gtBanner = document.getElementById('gt-result-banner');
  if (groundTruth) {
    const cnnOk = cnnDiag === groundTruth;
    const rfOk  = rfDiag  === groundTruth;
    gtBanner.className = 'gt-result-banner';
    gtBanner.innerHTML =
      `<span>GT: <strong>${groundTruth}</strong></span>` +
      `<span>CNN: ${cnnOk ? '✓ Corretta' : '✗ ' + cnnDiag}</span>` +
      `<span>RF:  ${rfOk  ? '✓ Corretta' : '✗ ' + rfDiag}</span>`;
  } else {
    gtBanner.className = 'gt-result-banner hidden';
  }

  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderModel(model, result) {
  const { diagnosi: diag, confidenza: conf, distribuzione: dist, stato_affidabilita: stato } = result;

  document.getElementById(`${model}-diagnosis`).textContent  = diag;
  document.getElementById(`${model}-confidence`).textContent = (conf * 100).toFixed(1) + '%';
  document.getElementById(`${model}-bar`).style.width        = (conf * 100) + '%';

  const relBadge = document.getElementById(`${model}-reliability`);
  relBadge.textContent = stato;
  relBadge.className   = 'reliability-badge ' + (stato.includes('alta') ? 'rel-high' : 'rel-low');

  const probContainer = document.getElementById(`${model}-probs`);
  probContainer.innerHTML = '';
  const baseColor = model === 'cnn' ? 'var(--cnn-color)' : 'var(--rf-color)';

  Object.entries(dist).forEach(([cls, prob]) => {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-cls">${cls.split(' ')[0]}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:${pct}%;background:${prob > 0.4 ? (CLASS_COLORS[cls] || DEFAULT_COLOR) : baseColor}"></div>
      </div>
      <span class="prob-val">${pct}%</span>
    `;
    probContainer.appendChild(row);
  });
}

/* ── History ─────────────────────────────────────────────────────────────── */
loadHistory();
document.getElementById('btn-refresh').addEventListener('click', loadHistory);

async function loadHistory() {
  const list = document.getElementById('history-list');
  list.innerHTML = '<div class="empty-state">Caricamento...</div>';
  try {
    const res  = await fetch(HISTORY_URL + '?limit=50');
    if (!res.ok) throw new Error(`Errore ${res.status}`);
    const data = await res.json();
    const preds = data.data || [];

    document.getElementById('history-count').textContent =
      preds.length > 0 ? `${preds.length} predizioni` : '';

    if (preds.length === 0) {
      list.innerHTML = '<div class="empty-state">Nessuna predizione disponibile. Esegui una classificazione per vederla qui.</div>';
      return;
    }

    list.innerHTML = '';
    preds.forEach(p => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const ts = new Date(p.timestamp).toLocaleString('it-IT');
      const gt = (p.ground_truth && p.ground_truth !== 'Non specificato')
        ? `<div class="hi-gt" style="font-family:var(--font-mono);font-size:.7rem;color:#059669;grid-column:1/-1;">GT: ${p.ground_truth}</div>`
        : '';
      item.innerHTML = `
        <div class="hi-time">${ts}</div>
        <div class="hi-diag">
          <span class="cnn-tag">CNN: ${p.cnn?.diagnosi ?? '—'}</span>
          &nbsp;·&nbsp;
          <span class="rf-tag">RF: ${p.rf?.diagnosi ?? '—'}</span>
        </div>
        <div class="hi-conf">
          CNN ${((p.cnn?.confidenza ?? 0) * 100).toFixed(0)}%<br/>
          RF&nbsp; ${((p.rf?.confidenza ?? 0) * 100).toFixed(0)}%
        </div>
        ${gt}
      `;
      list.appendChild(item);
    });
  } catch (err) {
    list.innerHTML = `<div class="empty-state">Errore nel caricamento: ${err.message}</div>`;
  }
}