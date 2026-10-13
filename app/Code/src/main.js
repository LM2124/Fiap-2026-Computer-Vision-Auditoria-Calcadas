import * as inf from './inference.js';

const video           = document.getElementById('video');
const staticImg       = document.getElementById('static-img');
const overlay         = document.getElementById('overlay');
const ctx             = overlay.getContext('2d');
const placeholder     = document.getElementById('placeholder');
const loader          = document.getElementById('loader');
const loaderMsg       = document.getElementById('loader-msg');
const errorScreen     = document.getElementById('error-screen');
const errorMsg        = document.getElementById('error-msg');
const fpsBadge        = document.getElementById('fps-badge');
const modelBadge      = document.getElementById('model-badge');
const btnCamera       = document.getElementById('btn-camera');
const btnUpload       = document.getElementById('btn-upload');
const fileInput       = document.getElementById('file-input');
const confSlider      = document.getElementById('conf-slider');
const confVal         = document.getElementById('conf-val');
const iouSlider       = document.getElementById('iou-slider');
const iouVal          = document.getElementById('iou-val');
const modelViewToggle = document.getElementById('model-view-toggle');
const viewport        = document.getElementById('viewport');
const btnTheme        = document.getElementById('btn-theme');
const btnFullscreen   = document.getElementById('btn-fullscreen');
const statusDetections= document.getElementById('status-detections');

let session          = null;
let cameraRunning    = false;
let inferenceRunning = false;
let showModelView    = true;
let rafId            = null;
let fpsSmooth        = 0;
let lastInfStart     = 0;
let displayedW       = 1;
let displayedH       = 1;
let offsetX          = 0;
let offsetY          = 0;
let currentModelKey  = 'sidewalk-640';

function updateDisplayMetrics() {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW || 640;
  const srcH = state.srcH || 640;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  if (srcAspect > canvasAspect) {
    displayedW = canvasW;
    displayedH = canvasW / srcAspect;
  } else {
    displayedH = canvasH;
    displayedW = canvasH * srcAspect;
  }

  offsetX = (canvasW - displayedW) / 2;
  offsetY = (canvasH - displayedH) / 2;
}

function toCanvasCoords(x, y) {
  const state = inf.getLastPreprocessState();
  const srcW = state.srcW || 640;
  const srcH = state.srcH || 640;
  const modW = state.inputW || 640;
  const modH = state.inputH || 640;
  const canvasW = overlay.width;
  const canvasH = overlay.height;
  const srcAspect = srcW / srcH;
  const canvasAspect = canvasW / canvasH;

  let dW, dH, oX, oY;
  if (srcAspect > canvasAspect) {
    dW = canvasW;
    dH = canvasW / srcAspect;
  } else {
    dH = canvasH;
    dW = canvasH * srcAspect;
  }
  oX = (canvasW - dW) / 2;
  oY = (canvasH - dH) / 2;

  let srcX, srcY;
  if (state.scale !== undefined) {
    srcX = (x - state.padX) / state.scale;
    srcY = (y - state.padY) / state.scale;
  } else {
    srcX = state.sx + x * (state.sw / modW);
    srcY = state.sy + y * (state.sh / modH);
  }

  return {
    x: oX + srcX * (dW / srcW),
    y: oY + srcY * (dH / srcH),
  };
}

// ── Tema ──────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  btnTheme.textContent = theme === 'dark' ? '☀' : '🌙';
  localStorage.setItem('yoloinfer-theme', theme);
}

btnTheme.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

// ── Classes e Modelos de Calçadas ─────────────────────────

const SIDEWALK_CLASSES = ['buraco', 'faixa de pedestre', 'lixo grande', 'poste', 'vegetacao'];
const SIDEWALK_COLORS = {
  0: '#ff9100', // buraco: laranja
  1: '#00e5ff', // faixa de pedestre: ciano
  2: '#ff1744', // lixo grande: vermelho
  3: '#d500f9', // poste: magenta
  4: '#00e676', // vegetacao: verde
};

const base = import.meta.env.BASE_URL;

const BUILTIN_MODELS = {
  'sidewalk-640': `${base}models/sidewalk-yolo11n-640.onnx`,
  'sidewalk-384': `${base}models/sidewalk-yolo11n-384.onnx`,
  'sidewalk-192': `${base}models/sidewalk-yolo11n-192.onnx`,
};

const MODEL_LABELS = {
  'sidewalk-640': 'Calçadas 640p',
  'sidewalk-384': 'Calçadas 384p',
  'sidewalk-192': 'Calçadas 192p',
};

let normalizedDetCoords = false;

function getClassName(classId) {
  return SIDEWALK_CLASSES[classId] ?? 'c' + classId;
}

function getClassColor(classId) {
  return SIDEWALK_COLORS[classId] ?? '#ffd600';
}

function updateConfDisplay() {
  inf.setConf(parseFloat(confSlider.value) / 100);
  confVal.textContent = (parseFloat(confSlider.value) / 100).toFixed(2);
}

function updateIouDisplay() {
  inf.setIou(parseFloat(iouSlider.value) / 100);
  iouVal.textContent = (parseFloat(iouSlider.value) / 100).toFixed(2);
}

confSlider.addEventListener('input', updateConfDisplay);
iouSlider.addEventListener('input', updateIouDisplay);

confSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

iouSlider.addEventListener('change', () => {
  if (!staticImg.hidden) runStaticImage();
});

function resizeCanvas() {
  const rect = overlay.getBoundingClientRect();
  overlay.width  = Math.max(1, Math.round(rect.width));
  overlay.height = Math.max(1, Math.round(rect.height));
  updateDisplayMetrics();
}

function updateModelBadge(inputDim) {
  const label = MODEL_LABELS[currentModelKey] || 'Calçadas';
  const dimStr = inputDim ? ` · ${inputDim}px` : '';
  modelBadge.textContent = label + dimStr;
  modelBadge.hidden = false;
}

// ── Câmera Traseira (Mobile) ──────────────────────────────

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'environment' },
      width:  { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  await video.play();

  if (video.videoWidth && video.videoHeight) {
    viewport.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  viewport.classList.add('camera-live');

  // Desabilita botões de amostra rápida enquanto a câmera está ligada
  document.querySelectorAll('.btn-sample').forEach(b => {
    b.disabled = true;
    b.classList.remove('active');
  });
}

function stopCamera() {
  cameraRunning = false;
  inferenceRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  video.hidden = true;
  fpsBadge.hidden = true;
  btnCamera.textContent = '📷 Iniciar Câmera';
  btnCamera.classList.remove('running');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  viewport.style.aspectRatio = '';
  viewport.classList.remove('camera-live');

  // Reabilita botões de amostra rápida
  document.querySelectorAll('.btn-sample').forEach(b => {
    b.disabled = false;
  });
}

async function runFrame() {
  if (!cameraRunning) return;
  if (inferenceRunning) {
    rafId = requestAnimationFrame(runFrame);
    return;
  }
  inferenceRunning = true;
  const t0 = performance.now();
  try {
    const results = await doInference(video);
    drawResults(results);
    if (lastInfStart > 0) {
      const dt = t0 - lastInfStart;
      fpsSmooth = fpsSmooth * 0.85 + (1000 / dt) * 0.15;
      fpsBadge.textContent = Math.round(fpsSmooth) + ' FPS';
    }
    lastInfStart = t0;
  } catch (e) {
    console.error(e);
  }
  inferenceRunning = false;
  if (cameraRunning) rafId = requestAnimationFrame(runFrame);
}

async function runStaticImage() {
  if (!staticImg.complete || staticImg.naturalWidth === 0) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  try {
    const results = await doInference(staticImg);
    drawResults(results);
  } catch (e) {
    console.error(e);
  }
}

function drawResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Se a visão do modelo estiver ativa, renderiza o frame pré-processado esticado para a proporção original
  if (showModelView) {
    const prepCanvas = inf.getPreprocessCanvas();
    const state = inf.getLastPreprocessState();
    if (prepCanvas && state.newW && state.newH) {
      ctx.imageSmoothingEnabled = false; // Pixels reais do modelo sem blur artificial
      ctx.drawImage(
        prepCanvas,
        state.padX, state.padY, state.newW, state.newH,
        offsetX, offsetY, displayedW, displayedH
      );
      ctx.imageSmoothingEnabled = true;
    }
  }

  if (statusDetections) {
    const count = results ? results.length : 0;
    statusDetections.textContent = `${count} objeto${count === 1 ? '' : 's'}`;
  }

  if (!results || results.length === 0) return;
  results.forEach(r => {
    const c = getClassColor(r.classId);
    drawBox(r, c);
  });
}

function drawBox(r, c) {
  const p1 = toCanvasCoords(r.x1, r.y1);
  const p2 = toCanvasCoords(r.x2, r.y2);
  const w = p2.x - p1.x;
  const h = p2.y - p1.y;

  ctx.strokeStyle = c;
  ctx.lineWidth   = 2.5;
  ctx.setLineDash([]);
  ctx.strokeRect(p1.x, p1.y, w, h);

  const lbl = getClassName(r.classId) + ': ' + (r.conf * 100).toFixed(0) + '%';
  ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  const lw = ctx.measureText(lbl).width;
  const lx = p1.x;
  const ly = Math.max(5, p1.y - 24);

  ctx.fillStyle = c;
  ctx.fillRect(lx, ly, lw + 10, 20);
  ctx.fillStyle = '#000000';
  ctx.fillText(lbl, lx + 5, ly + 15);
}

// Configurações do modelo carregado
async function configureLoadedModel() {
  const conventions = await inf.detectOutputConventions();
  if (!conventions) return;
  const { normalizedCoords, inputW } = conventions;

  normalizedDetCoords = normalizedCoords;
  inf.setPreprocessMode('letterbox');
  updateModelBadge(inputW);
}

// ── Botões de Ação ────────────────────────────────────────

btnCamera.addEventListener('click', async () => {
  if (cameraRunning) {
    stopCamera();
    placeholder.hidden = false;
    return;
  }
  if (!session) {
    errorMsg.textContent = 'Nenhum modelo carregado.';
    errorScreen.hidden = false;
    return;
  }
  staticImg.hidden = true;
  placeholder.hidden = true;
  video.hidden = false;
  fpsBadge.hidden = false;
  try {
    await startCamera();
  } catch {
    video.hidden = true;
    placeholder.hidden = true;
    fpsBadge.hidden = true;
    errorMsg.textContent = 'Acesso à câmera requer conexão HTTPS. Utilize localhost ou a rede local com SSL ativo.';
    errorScreen.hidden = false;
    return;
  }
  cameraRunning = true;
  btnCamera.textContent = '⏹ Parar Câmera';
  btnCamera.classList.add('running');
  fpsSmooth = 0;
  lastInfStart = 0;
  resizeCanvas();
  rafId = requestAnimationFrame(runFrame);
});

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;
  document.querySelectorAll('.btn-sample').forEach(b => b.classList.remove('active'));
  stopCamera();
  placeholder.hidden = true;
  const url = URL.createObjectURL(file);
  staticImg.src = url;
  await new Promise(r => { staticImg.onload = r; });
  staticImg.hidden = false;
  resizeCanvas();
  await runStaticImage();
  URL.revokeObjectURL(url);
  fileInput.value = '';
});

// Listener para as amostras rápidas de calçada
document.querySelectorAll('.btn-sample').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!session || cameraRunning) return;
    document.querySelectorAll('.btn-sample').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    placeholder.hidden = true;
    staticImg.src = btn.dataset.src;
    await new Promise(r => { staticImg.onload = r; });
    staticImg.hidden = false;
    resizeCanvas();
    await runStaticImage();
  });
});

// Listener para a barra de botões de perfil de modelo
document.querySelectorAll('.btn-model-choice').forEach(btn => {
  btn.addEventListener('click', async () => {
    const modelKey = btn.dataset.model;
    if (!modelKey || modelKey === currentModelKey) return;
    document.querySelectorAll('.btn-model-choice').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    await switchModel(modelKey);
  });
});

async function switchModel(modelKey) {
  const modelPath = BUILTIN_MODELS[modelKey];
  if (!modelPath) return;

  const wasCameraRunning = cameraRunning;
  if (cameraRunning) stopCamera();

  errorScreen.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Carregando ' + (MODEL_LABELS[modelKey] || modelKey) + '...';
  currentModelKey = modelKey;
  try {
    if (session) { session.endSession?.(); session = null; }
    session = await inf.loadModel(modelPath);
    await configureLoadedModel();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    loader.hidden = true;
    btnCamera.disabled = false;
    btnCamera.textContent = '📷 Iniciar Câmera';
    btnUpload.classList.remove('disabled');
    fpsBadge.hidden = true;

    if (!staticImg.hidden && staticImg.src) {
      await runStaticImage();
    } else if (wasCameraRunning) {
      btnCamera.click();
    }
  } catch (e) {
    loader.hidden = true;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Falha ao carregar modelo: ' + e.message;
    modelBadge.textContent = 'Erro';
  }
}

modelViewToggle.addEventListener('change', () => {
  showModelView = modelViewToggle.checked;
  if (showModelView) {
    viewport.classList.add('show-model-view');
  } else {
    viewport.classList.remove('show-model-view');
  }
  if (!staticImg.hidden) runStaticImage();
});

// ── Tela Cheia (Fullscreen) ───────────────────────────────
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    if (viewport.requestFullscreen) {
      viewport.requestFullscreen();
    } else if (viewport.webkitRequestFullscreen) {
      viewport.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    }
  }
}

if (btnFullscreen) {
  btnFullscreen.addEventListener('click', toggleFullscreen);
}

document.addEventListener('fullscreenchange', () => {
  const isFs = !!document.fullscreenElement;
  if (btnFullscreen) {
    btnFullscreen.textContent = isFs ? '🗗' : '⛶';
    btnFullscreen.title = isFs ? 'Sair da Tela Cheia' : 'Tela Cheia';
  }
  resizeCanvas();
  if (!staticImg.hidden) {
    runStaticImage();
  }
});
document.addEventListener('webkitfullscreenchange', () => {
  const isFs = !!document.fullscreenElement;
  if (btnFullscreen) {
    btnFullscreen.textContent = isFs ? '🗗' : '⛶';
  }
  resizeCanvas();
  if (!staticImg.hidden) {
    runStaticImage();
  }
});

window.addEventListener('resize', resizeCanvas);

// ── Inferência Principal ──────────────────────────────────

async function doInference(source) {
  if (!session) return [];
  const input = inf.preprocess(source);
  const feeds = { [session.inputNames[0]]: input };
  const out = await session.run(feeds);
  const output = out[session.outputNames[0]];
  const numBoxes = output.dims[2];
  const cTV = inf.getConf();
  const iTV = inf.getIou();

  const dims = inf.getInputDims();
  return inf.postprocessDetection(output, numBoxes, cTV, iTV, normalizedDetCoords ? dims.w : 1);
}

// ── Inicialização ─────────────────────────────────────────

async function init() {
  inf.setConf(0.25);
  inf.setIou(0.45);
  updateConfDisplay();
  updateIouDisplay();

  resizeCanvas();
  if (showModelView) {
    viewport.classList.add('show-model-view');
  }
  errorScreen.hidden = true;
  fpsBadge.hidden = true;
  modelBadge.hidden = true;
  loader.hidden = false;
  loaderMsg.textContent = 'Carregando modelo de calçadas...';

  try {
    currentModelKey = 'sidewalk-640';
    session = await inf.loadModel(BUILTIN_MODELS['sidewalk-640']);
    await configureLoadedModel();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    loader.hidden = true;
    btnCamera.disabled = false;
    btnCamera.textContent = '📷 Iniciar Câmera';
    btnUpload.classList.remove('disabled');
  } catch (e) {
    loader.hidden = true;
    errorScreen.hidden = false;
    errorMsg.textContent = 'Não foi possível carregar o modelo inicial: ' + e.message;
    fpsBadge.hidden = true;
    modelBadge.textContent = 'Sem Modelo';
    modelBadge.hidden = false;
  }
}

init();
