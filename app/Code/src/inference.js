import * as ort from 'onnxruntime-web';

let inputW = 640;
let inputH = 640;

let session = null;
let customPath = null;
let lastPreprocessState = { sx: 0, sy: 0, sw: 640, sh: 640, srcW: 640, srcH: 640, newW: 640, newH: 640, padX: 0, padY: 0, inputW: 640, inputH: 640 };
let preprocessMode = 'letterbox';

export function getInputDims() {
  return { w: inputW, h: inputH };
}

export function setInputDims(w, h) {
  inputW = w;
  inputH = h;
  _canvas.width = w;
  _canvas.height = h;
}

export function setPreprocessMode(mode) {
  preprocessMode = mode;
}

export function getPreprocessMode() {
  return preprocessMode;
}

// Descobre automaticamente a dimensão de entrada suportada pelo modelo ONNX
async function detectModelInputDims(sess, hintW = 640) {
  const candidates = [hintW, 640, 384, 192, 320, 256].filter((v, i, a) => a.indexOf(v) === i);
  for (const sz of candidates) {
    try {
      const dummy = new ort.Tensor('float32', new Float32Array(3 * sz * sz).fill(0.45), [1, 3, sz, sz]);
      const out = await sess.run({ [sess.inputNames[0]]: dummy });
      return { w: sz, h: sz, out };
    } catch {
      // Shape mismatch, tenta próxima resolução
    }
  }
  return { w: 640, h: 640, out: null };
}

// Detecta convenções de saída do modelo carregado (coordenadas e número de classes)
export async function detectOutputConventions() {
  if (!session) return null;

  let hint = 640;
  if (customPath && customPath.includes('192')) hint = 192;
  else if (customPath && customPath.includes('384')) hint = 384;
  else if (customPath && customPath.includes('640')) hint = 640;

  const detected = await detectModelInputDims(session, hint);
  setInputDims(detected.w, detected.h);

  const out = detected.out || await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * inputH * inputW).fill(0.45), [1, 3, inputH, inputW])
  });

  const o = out[session.outputNames[0]];
  const channels = o.dims[1];
  const numBoxes = o.dims[2];

  let maxCx = 0;
  const end = Math.min(numBoxes, 200);
  for (let i = 0; i < end; i++) maxCx = Math.max(maxCx, o.data[i]);
  const normalizedCoords = maxCx < 1.5;

  const numClasses = channels - 4;
  return { type: 'detection', normalizedCoords, numClasses, inputW, inputH };
}

export function getLastPreprocessState() {
  return lastPreprocessState;
}

export async function loadModel(overridePath) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const base = import.meta.env.BASE_URL;
  const modelPath = overridePath || `${base}models/sidewalk-yolo11n-640.onnx`;

  const providers = ['wasm'];
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
  return session;
}

export function getSession() {
  return session;
}

const _canvas = document.createElement('canvas');
_canvas.width = inputW;
_canvas.height = inputH;
const _ctx = _canvas.getContext('2d', { willReadFrequently: true });

export function getPreprocessCanvas() {
  return _canvas;
}

export function preprocess(source) {
  const srcW = source.videoWidth || source.naturalWidth || source.width || inputW;
  const srcH = source.videoHeight || source.naturalHeight || source.height || inputH;

  if (preprocessMode === 'letterbox') {
    const scale = Math.min(inputW / srcW, inputH / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.floor((inputW - newW) / 2);
    const padY = Math.floor((inputH - newH) / 2);
    lastPreprocessState = { srcW, srcH, scale, padX, padY, newW, newH, inputW, inputH };
    _ctx.fillStyle = '#000';
    _ctx.fillRect(0, 0, inputW, inputH);
    _ctx.drawImage(source, padX, padY, newW, newH);
  } else {
    const minDim = Math.min(srcW, srcH);
    const sx = Math.round((srcW - minDim) / 2);
    const sy = Math.round((srcH - minDim) / 2);
    lastPreprocessState = { sx, sy, sw: minDim, sh: minDim, srcW, srcH, inputW, inputH, newW: inputW, newH: inputH, padX: 0, padY: 0 };
    _ctx.drawImage(source, sx, sy, minDim, minDim, 0, 0, inputW, inputH);
  }

  const imgData = _ctx.getImageData(0, 0, inputW, inputH).data;
  const pxCount = inputW * inputH;
  const f32 = new Float32Array(3 * pxCount);

  for (let i = 0; i < pxCount; i++) {
    const s = i * 4;
    f32[i] = imgData[s] / 255.0;
    f32[i + pxCount] = imgData[s + 1] / 255.0;
    f32[i + pxCount * 2] = imgData[s + 2] / 255.0;
  }

  return new ort.Tensor('float32', f32, [1, 3, inputH, inputW]);
}

let confThresh = 0.25;
let iouThresh = 0.45;

export function getConf() { return confThresh; }
export function getIou() { return iouThresh; }
export function setConf(v) { confThresh = v; }
export function setIou(v) { iouThresh = v; }

export function nms(boxes, iouTV) {
  const byConf = [...boxes].sort((a, b) => b.conf - a.conf);
  const kept = [];
  const pruned = new Set();

  for (let i = 0; i < byConf.length; i++) {
    if (pruned.has(i)) continue;
    kept.push(byConf[i]);
    for (let j = i + 1; j < byConf.length; j++) {
      if (pruned.has(j)) continue;
      if (computeIoU(byConf[i], byConf[j]) < iouTV) continue;
      pruned.add(j);
    }
  }
  return kept;
}

function computeIoU(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bA = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / Math.max(aA + bA - inter, 0.0001);
}

export function postprocessDetection(outputTensor, numBoxes, confTV, iouTV, coordScale = 1) {
  const data = outputTensor.data;
  const numClasses = outputTensor.dims[1] - 4;
  const candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i] * coordScale;
    const cy = data[1 * numBoxes + i] * coordScale;
    const w = data[2 * numBoxes + i] * coordScale;
    const h = data[3 * numBoxes + i] * coordScale;

    let bestScore = 0;
    let bestClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const si = (4 + c) * numBoxes + i;
      if (data[si] > bestScore) {
        bestScore = data[si];
        bestClassId = c;
      }
    }

    const conf = Math.max(0, bestScore);
    if (conf < confTV) continue;

    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      classId: bestClassId,
      conf,
      type: 'bbox',
    });
  }

  return nms(candidates, iouTV);
}
