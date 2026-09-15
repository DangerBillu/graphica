/**
 * REACTIONARY — AI Emotion-to-Meme Studio Controller
 * Handles 10-emotion real-time telemetry, face presence alerts,
 * interactive emotion overrides, staged generation loading,
 * dual-QR code exhibition, and mobile phone download flow.
 */

// Application State
const state = {
  detectedEmotion: 'HAPPY',
  detectedConfidence: 85,
  overrideEmotion: null, // null = auto-detected, string = user override
  hasFace: false,
  numFaces: 1,
  warningMsg: null,
  isGenerating: false,
  latestMemeData: null,
  emotionsList: [],
};

// DOM Elements
const elements = {
  camLoader: document.getElementById('camLoader'),
  cameraStream: document.getElementById('cameraStream'),
  reticleLayer: document.getElementById('reticleLayer'),
  reticleBox: document.getElementById('reticleBox'),
  reticleLabel: document.getElementById('reticleLabel'),
  feedCoords: document.getElementById('feedCoords'),
  viewportAlert: document.getElementById('viewportAlert'),
  alertText: document.getElementById('alertText'),
  camStatusText: document.getElementById('camStatusText'),
  fpsCounter: document.getElementById('fpsCounter'),
  faceCountVal: document.getElementById('faceCountVal'),
  calibStatus: document.getElementById('calibStatus'),

  emotionSymbol: document.getElementById('emotionSymbol'),
  emotionLabel: document.getElementById('emotionLabel'),
  emotionDesc: document.getElementById('emotionDesc'),
  confidenceVal: document.getElementById('confidenceVal'),

  telSmile: document.getElementById('telSmile'),
  telBrow: document.getElementById('telBrow'),
  telJaw: document.getElementById('telJaw'),
  telSquint: document.getElementById('telSquint'),
  fillSmile: document.getElementById('fillSmile'),
  fillBrow: document.getElementById('fillBrow'),
  fillJaw: document.getElementById('fillJaw'),
  fillSquint: document.getElementById('fillSquint'),

  calloutMode: document.getElementById('calloutMode'),
  calloutVal: document.getElementById('calloutVal'),
  calloutIndicator: document.getElementById('calloutIndicator'),
  resetAutoBtn: document.getElementById('resetAutoBtn'),
  emotionPillsGrid: document.getElementById('emotionPillsGrid'),
  overrideBadge: document.getElementById('overrideBadge'),

  paramContrast: document.getElementById('paramContrast'),
  paramContrastVal: document.getElementById('paramContrastVal'),
  paramGrain: document.getElementById('paramGrain'),
  paramGrainVal: document.getElementById('paramGrainVal'),

  generateBtn: document.getElementById('generateBtn'),
  exhibitionModal: document.getElementById('exhibitionModal'),
  posterLoader: document.getElementById('posterLoader'),
  loadingStageText: document.getElementById('loadingStageText'),
  loadingStageSub: document.getElementById('loadingStageSub'),
  posterResultImg: document.getElementById('posterResultImg'),
  downloadQrImg: document.getElementById('downloadQrImg'),
  modalTitle: document.getElementById('modalTitle'),
  postEmotion: document.getElementById('postEmotion'),
  postCaption: document.getElementById('postCaption'),
  postConfidence: document.getElementById('postConfidence'),
  postDesignId: document.getElementById('postDesignId'),
};

// Initialize Application
async function initApp() {
  bindSliderInputs();
  bindKeyboardShortcuts();
  await loadEmotions();
  startTelemetryLoop();

  elements.cameraStream.onload = () => {
    elements.camLoader.classList.add('hidden');
    elements.camStatusText.textContent = 'OPTICAL FEED LIVE';
  };
  elements.cameraStream.onerror = () => {
    elements.camStatusText.textContent = 'OPTICAL FEED OFFLINE';
  };
}

// Bind Sliders
function bindSliderInputs() {
  elements.paramContrast.addEventListener('input', (e) => {
    elements.paramContrastVal.textContent = `+${e.target.value}% BOOST`;
  });
  elements.paramGrain.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    const tag = val > 20 ? 'HEAVY' : val > 8 ? 'MEDIUM' : 'LIGHT';
    elements.paramGrainVal.textContent = `${tag} (${val}%)`;
  });
}

// Bind Global Shortcuts
function bindKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body && !state.isGenerating) {
      e.preventDefault();
      triggerMemeGeneration();
    } else if (e.code === 'Escape') {
      closeExhibition();
    }
  });
}

// Fetch 10 Supported Emotions
async function loadEmotions() {
  try {
    const res = await fetch('/api/emotions');
    const data = await res.json();
    state.emotionsList = data.emotions || [];
    renderEmotionPills();
  } catch (err) {
    console.error('Failed to load emotions:', err);
  }
}

// Render 10 Emotion Selector Pills
function renderEmotionPills() {
  elements.emotionPillsGrid.innerHTML = '';

  state.emotionsList.forEach((item) => {
    const pill = document.createElement('div');
    const isSelected = (state.overrideEmotion || state.detectedEmotion) === item.id;
    pill.className = `emotion-pill ${isSelected ? 'selected' : ''}`;
    pill.dataset.emotionId = item.id;
    pill.onclick = () => selectEmotionOverride(item.id);

    pill.innerHTML = `
      <span class="pill-emoji">${item.symbol}</span>
      <div class="pill-info">
        <span class="pill-name">${item.label}</span>
        <span class="pill-tagline">${item.tagline}</span>
      </div>
    `;
    elements.emotionPillsGrid.appendChild(pill);
  });
}

// User selects an emotion pill to override detected emotion
function selectEmotionOverride(emotionId) {
  state.overrideEmotion = emotionId;
  updateActiveEmotionBanner();
  updatePillSelectionUI();
}

// Reset back to Auto Detected
function resetToAutoDetected() {
  state.overrideEmotion = null;
  updateActiveEmotionBanner();
  updatePillSelectionUI();
}

// Update Active Emotion Callout Banner
function updateActiveEmotionBanner() {
  const activeKey = state.overrideEmotion || state.detectedEmotion;
  const item = state.emotionsList.find((e) => e.id === activeKey) || {
    label: activeKey,
    symbol: '✨',
    accent: '#CCFF00',
  };

  if (state.overrideEmotion) {
    elements.calloutMode.textContent = 'USER OVERRIDE:';
    elements.calloutVal.textContent = `${item.label.toUpperCase()} (${item.symbol})`;
    elements.calloutIndicator.style.backgroundColor = item.accent || '#CCFF00';
    elements.resetAutoBtn.style.display = 'block';
    elements.overrideBadge.textContent = 'OVERRIDE ACTIVE';
  } else {
    elements.calloutMode.textContent = 'AUTO DETECTED:';
    elements.calloutVal.textContent = `${state.detectedEmotion} (${elements.emotionSymbol.textContent})`;
    elements.calloutIndicator.style.backgroundColor = '#CCFF00';
    elements.resetAutoBtn.style.display = 'none';
    elements.overrideBadge.textContent = 'AUTO MODE';
  }
}

// Update Selected Pill styling
function updatePillSelectionUI() {
  const activeKey = state.overrideEmotion || state.detectedEmotion;
  document.querySelectorAll('.emotion-pill').forEach((pill) => {
    if (pill.dataset.emotionId === activeKey) {
      pill.classList.add('selected');
    } else {
      pill.classList.remove('selected');
    }
  });
}

// Real-Time Telemetry Polling Loop
let lastFrameTime = performance.now();
let frameCount = 0;

function startTelemetryLoop() {
  async function poll() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        updateTelemetryUI(data);
      }
    } catch (e) {}

    // FPS calculation
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      elements.fpsCounter.textContent = (frameCount * 1000 / (now - lastFrameTime)).toFixed(1);
      frameCount = 0;
      lastFrameTime = now;
    }

    setTimeout(poll, 120);
  }
  poll();
}

// Update UI with Live Telemetry
function updateTelemetryUI(data) {
  state.hasFace = data.has_face;
  state.numFaces = data.num_faces || (data.has_face ? 1 : 0);
  state.detectedEmotion = data.emotion || 'NEUTRAL';
  state.detectedConfidence = data.confidence || 85;

  elements.faceCountVal.textContent = state.numFaces;

  // Face Boundary Warning Alert Handling
  if (data.warning_msg) {
    elements.viewportAlert.classList.remove('hidden');
    elements.alertText.textContent = data.warning_msg;
    if (!data.has_face) {
      elements.reticleBox.style.display = 'none';
    }
  } else {
    elements.viewportAlert.classList.add('hidden');
    elements.reticleBox.style.display = 'block';
  }

  // Position Face Reticle
  if (data.has_face && data.face_box) {
    const [x0, y0, x1, y1] = data.face_box;
    const cw = elements.cameraStream.clientWidth || 640;
    const ch = elements.cameraStream.clientHeight || 480;
    const fw = data.frame_w || 640;
    const fh = data.frame_h || 480;

    const rx = (x0 / fw) * cw;
    const ry = (y0 / fh) * ch;
    const rw = ((x1 - x0) / fw) * cw;
    const rh = ((y1 - y0) / fh) * ch;

    elements.reticleBox.style.left = `${rx}px`;
    elements.reticleBox.style.top = `${ry}px`;
    elements.reticleBox.style.width = `${rw}px`;
    elements.reticleBox.style.height = `${rh}px`;

    elements.reticleLabel.textContent = `${data.emotion} // ${data.confidence}%`;
    elements.feedCoords.textContent = `X: ${Math.round(rx)} | Y: ${Math.round(ry)} | W: ${Math.round(rw)}`;
  }

  // Emotion Readout Card
  elements.emotionSymbol.textContent = data.symbol || '😐';
  elements.emotionLabel.textContent = (data.label || data.emotion).toUpperCase();
  elements.emotionDesc.textContent = data.tagline || data.description || '';
  elements.confidenceVal.textContent = `${data.confidence || 85}%`;

  // Gauges
  const tel = data.telemetry || {};
  elements.telSmile.textContent = (tel.smile || 0).toFixed(2);
  elements.telBrow.textContent = (tel.brow_down || 0).toFixed(2);
  elements.telJaw.textContent = (tel.jaw || 0).toFixed(2);
  elements.telSquint.textContent = (tel.squint || 0).toFixed(2);

  elements.fillSmile.style.width = `${Math.min(100, (tel.smile || 0) * 160)}%`;
  elements.fillBrow.style.width = `${Math.min(100, (tel.brow_down || 0) * 180)}%`;
  elements.fillJaw.style.width = `${Math.min(100, (tel.jaw || 0) * 160)}%`;
  elements.fillSquint.style.width = `${Math.min(100, (tel.squint || 0) * 180)}%`;

  // Update selection banner if in auto mode
  if (!state.overrideEmotion) {
    updateActiveEmotionBanner();
    updatePillSelectionUI();
  }
}

// Trigger Meme Synthesis with Staged Loading Feedback
async function triggerMemeGeneration() {
  if (state.isGenerating) return;
  state.isGenerating = true;

  // Open Exhibition Modal in Loading State
  elements.exhibitionModal.classList.remove('hidden');
  elements.posterLoader.classList.remove('hidden');
  elements.posterResultImg.style.opacity = '0';

  // Staged Loading Text Progress
  elements.loadingStageText.textContent = 'Detecting your emotion...';
  elements.loadingStageSub.textContent = 'Analyzing facial blendshapes and expression tension';

  const stageTimer1 = setTimeout(() => {
    elements.loadingStageText.textContent = 'Turning that expression into a meme...';
    elements.loadingStageSub.textContent = 'Synthesizing graphic-design poster and typography';
  }, 450);

  const stageTimer2 = setTimeout(() => {
    elements.loadingStageText.textContent = 'Adding the finishing touches...';
    elements.loadingStageSub.textContent = 'Compositing Graphica Club logo & Instagram QR code';
  }, 950);

  const targetEmotion = state.overrideEmotion || state.detectedEmotion;

  const payload = {
    emotion: targetEmotion,
    override: state.overrideEmotion !== null,
    params: {
      contrast: 1.0 + parseFloat(elements.paramContrast.value) / 100.0,
      grain: parseFloat(elements.paramGrain.value) / 100.0,
    },
  };

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`Meme generation failed: ${res.statusText}`);

    const result = await res.json();
    state.latestMemeData = result;

    clearTimeout(stageTimer1);
    clearTimeout(stageTimer2);

    elements.loadingStageText.textContent = 'Your meme is ready!';
    elements.loadingStageSub.textContent = 'Preparing mobile download QR code...';

    setTimeout(() => {
      displayGeneratedMeme(result);
    }, 400);
  } catch (err) {
    clearTimeout(stageTimer1);
    clearTimeout(stageTimer2);
    alert(`Error generating meme: ${err.message}`);
    closeExhibition();
  } finally {
    state.isGenerating = false;
  }
}

// Display Generated Meme and Phone Download QR Code
function displayGeneratedMeme(data) {
  elements.posterLoader.classList.add('hidden');
  elements.posterResultImg.src = data.image_url || data.image_b64;
  elements.posterResultImg.style.opacity = '1';

  // Populate Dynamic Phone Download QR Code (QR Code 2)
  elements.downloadQrImg.src = data.download_qr_b64;

  elements.modalTitle.textContent = `GRAPHICA MEME ${data.design_id || '#RX-001'}`;
  elements.postEmotion.textContent = `${data.emotion} ${data.symbol || ''}`;
  elements.postCaption.textContent = data.caption || 'POV: MEME GENERATED';
  elements.postConfidence.textContent = `${data.confidence || 90}%`;
  elements.postDesignId.textContent = data.design_id || '#RX-042';
}

// Download Meme Image on Computer
function downloadOnComputer() {
  if (!state.latestMemeData) return;
  const link = document.createElement('a');
  link.href = state.latestMemeData.download_url || state.latestMemeData.image_url;
  link.download = `GRAPHICA_MEME_${state.latestMemeData.emotion}_${state.latestMemeData.design_id.replace('#', '')}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Close Exhibition View
function closeExhibition() {
  elements.exhibitionModal.classList.add('hidden');
}

window.addEventListener('DOMContentLoaded', initApp);
