/**
 * REACTIONARY — Frontend Application Controller
 * Handles live telemetry polling, reticle positioning, style selection,
 * and high-resolution poster generation.
 */

// Application State
const state = {
  mode: 'auto', // 'auto' | 'manual'
  selectedStyle: 'Swiss',
  autoRecommendedStyle: 'Swiss',
  currentEmotion: 'NEUTRAL',
  currentConfidence: 85,
  hasFace: false,
  latestPosterData: null,
  isGenerating: false,
  styles: [],
};

// DOM References
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
  calibStatus: document.getElementById('calibStatus'),
  fpsCounter: document.getElementById('fpsCounter'),
  
  emotionSymbol: document.getElementById('emotionSymbol'),
  emotionLabel: document.getElementById('emotionLabel'),
  emotionDesc: document.getElementById('emotionDesc'),
  confidenceVal: document.getElementById('confidenceVal'),
  
  telJaw: document.getElementById('telJaw'),
  telBrow: document.getElementById('telBrow'),
  telSmile: document.getElementById('telSmile'),
  telSquint: document.getElementById('telSquint'),
  fillJaw: document.getElementById('fillJaw'),
  fillBrow: document.getElementById('fillBrow'),
  fillSmile: document.getElementById('fillSmile'),
  fillSquint: document.getElementById('fillSquint'),
  
  modeAutoBtn: document.getElementById('modeAutoBtn'),
  modeManualBtn: document.getElementById('modeManualBtn'),
  modeCaption: document.getElementById('modeCaption'),
  stylesGrid: document.getElementById('stylesGrid'),
  activeStyleBadge: document.getElementById('activeStyleBadge'),
  
  paramGrain: document.getElementById('paramGrain'),
  paramGrainVal: document.getElementById('paramGrainVal'),
  paramContrast: document.getElementById('paramContrast'),
  paramContrastVal: document.getElementById('paramContrastVal'),
  paramGrid: document.getElementById('paramGrid'),
  paramGridVal: document.getElementById('paramGridVal'),
  
  generateBtn: document.getElementById('generateBtn'),
  exhibitionModal: document.getElementById('exhibitionModal'),
  posterLoader: document.getElementById('posterLoader'),
  posterResultImg: document.getElementById('posterResultImg'),
  modalTitle: document.getElementById('modalTitle'),
  postReaction: document.getElementById('postReaction'),
  postStyle: document.getElementById('postStyle'),
  postScore: document.getElementById('postScore'),
  postDesignId: document.getElementById('postDesignId'),
  postTypography: document.getElementById('postTypography'),
  postTimestamp: document.getElementById('postTimestamp'),
  swatchCluster: document.getElementById('swatchCluster'),
};

// Initialize Application
async function initApp() {
  bindSliderInputs();
  bindKeyboardShortcuts();
  await loadStyles();
  startTelemetryLoop();
  
  // Fade out loader once video feed is streaming
  elements.cameraStream.onload = () => {
    elements.camLoader.classList.add('hidden');
    elements.camStatusText.textContent = 'OPTICAL FEED LIVE';
  };
  elements.cameraStream.onerror = () => {
    elements.camStatusText.textContent = 'OPTICAL FEED OFFLINE';
  };
}

// Bind parameter sliders
function bindSliderInputs() {
  elements.paramGrain.addEventListener('input', (e) => {
    elements.paramGrainVal.textContent = `${e.target.value > 20 ? 'HEAVY' : e.target.value > 8 ? 'MEDIUM' : 'LIGHT'} (${e.target.value}%)`;
  });
  elements.paramContrast.addEventListener('input', (e) => {
    elements.paramContrastVal.textContent = `+${e.target.value}% CONTRAST`;
  });
  elements.paramGrid.addEventListener('input', (e) => {
    const labels = ['MINIMAL 4PT', 'CLEAN 6PT', 'MODULAR 8PT', 'DENSE 12PT', 'CHAOTIC 16PT'];
    elements.paramGridVal.textContent = labels[e.target.value - 1] || '8PT';
  });
}

// Bind global keyboard shortcuts
function bindKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body && !state.isGenerating) {
      e.preventDefault();
      generatePosterAction();
    } else if (e.code === 'Escape') {
      closeExhibition();
    }
  });
}

// Fetch available design styles
async function loadStyles() {
  try {
    const res = await fetch('/api/styles');
    const data = await res.json();
    state.styles = data.styles || [];
    renderStylesGrid();
  } catch (err) {
    console.error('Failed to load styles:', err);
  }
}

// Render style selection cards
function renderStylesGrid() {
  elements.stylesGrid.innerHTML = '';
  
  state.styles.forEach((s) => {
    const card = document.createElement('div');
    card.className = `style-card ${s.id === state.selectedStyle ? 'selected' : ''}`;
    card.dataset.styleId = s.id;
    card.onclick = () => selectStyleManual(s.id);
    
    card.innerHTML = `
      <div class="style-accent-pip" style="background-color: ${s.accent}"></div>
      <div class="style-info">
        <span class="style-name">${s.name}</span>
        <span class="style-era">${s.era}</span>
      </div>
    `;
    elements.stylesGrid.appendChild(card);
  });
}

// Select a style manually
function selectStyleManual(styleId) {
  state.selectedStyle = styleId;
  setMode('manual');
  updateSelectedStyleUI();
}

function updateSelectedStyleUI() {
  document.querySelectorAll('.style-card').forEach((card) => {
    if (card.dataset.styleId === state.selectedStyle) {
      card.classList.add('selected');
    } else {
      card.classList.remove('selected');
    }
  });
  elements.activeStyleBadge.textContent = state.selectedStyle.toUpperCase();
}

// Set Auto or Manual Mode
function setMode(mode) {
  state.mode = mode;
  if (mode === 'auto') {
    elements.modeAutoBtn.classList.add('active');
    elements.modeManualBtn.classList.remove('active');
    elements.modeCaption.innerHTML = '<strong>AUTO MODE ACTIVE:</strong> Design language dynamically synchronizes with detected facial reaction.';
    state.selectedStyle = state.autoRecommendedStyle;
    updateSelectedStyleUI();
  } else {
    elements.modeAutoBtn.classList.remove('active');
    elements.modeManualBtn.classList.add('active');
    elements.modeCaption.innerHTML = '<strong>MANUAL MODE ACTIVE:</strong> Custom graphic style selected. Emotion still guides composition & metadata.';
  }
}

// Polling telemetry loop
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
    } catch (e) {
      // transient network wait
    }
    
    // Compute UI FPS
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

// Update DOM with live telemetry data
function updateTelemetryUI(data) {
  state.hasFace = data.has_face;
  state.currentEmotion = data.emotion;
  state.currentConfidence = data.confidence || 80;
  state.autoRecommendedStyle = data.style || 'Swiss';
  
  if (state.mode === 'auto' && state.selectedStyle !== state.autoRecommendedStyle) {
    state.selectedStyle = state.autoRecommendedStyle;
    updateSelectedStyleUI();
  }
  
  // Alert banner
  if (!data.has_face) {
    elements.viewportAlert.classList.remove('hidden');
    elements.alertText.textContent = 'NO FACE DETECTED IN OPTICAL FRAME';
    elements.reticleBox.style.display = 'none';
  } else if (data.num_faces > 1) {
    elements.viewportAlert.classList.remove('hidden');
    elements.alertText.textContent = `MULTIPLE FACES DETECTED (${data.num_faces}) — TRACKING PRIMARY`;
  } else {
    elements.viewportAlert.classList.add('hidden');
    elements.reticleBox.style.display = 'block';
  }
  
  // Reticle positioning
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
  
  // Emotion readout
  elements.emotionSymbol.textContent = data.symbol || '😐';
  elements.emotionLabel.textContent = (data.label || 'Unimpressed').toUpperCase();
  elements.emotionDesc.textContent = data.description || '';
  elements.confidenceVal.textContent = `${data.confidence || 85}%`;
  
  // Blendshape telemetry
  const tel = data.telemetry || {};
  elements.telJaw.textContent = (tel.jaw || 0).toFixed(2);
  elements.telBrow.textContent = (tel.brow_down || 0).toFixed(2);
  elements.telSmile.textContent = (tel.smile || 0).toFixed(2);
  elements.telSquint.textContent = (tel.squint || 0).toFixed(2);
  
  elements.fillJaw.style.width = `${Math.min(100, (tel.jaw || 0) * 160)}%`;
  elements.fillBrow.style.width = `${Math.min(100, (tel.brow_down || 0) * 200)}%`;
  elements.fillSmile.style.width = `${Math.min(100, (tel.smile || 0) * 180)}%`;
  elements.fillSquint.style.width = `${Math.min(100, (tel.squint || 0) * 180)}%`;
  
  elements.calibStatus.textContent = data.is_calibrated ? 'YES (USER BASELINE)' : 'GENERIC';
}

// Generate Poster Action
async function generatePosterAction() {
  if (state.isGenerating) return;
  state.isGenerating = true;
  
  // Open exhibition modal in loading state
  elements.exhibitionModal.classList.remove('hidden');
  elements.posterLoader.classList.remove('hidden');
  elements.posterResultImg.style.opacity = '0';
  
  const payload = {
    style: state.selectedStyle,
    mode: state.mode,
    params: {
      grain: parseFloat(elements.paramGrain.value) / 100.0,
      contrast: 1.0 + (parseFloat(elements.paramContrast.value) / 100.0),
      grid: parseInt(elements.paramGrid.value, 10),
    }
  };
  
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) throw new Error(`Generation failed: ${res.statusText}`);
    
    const result = await res.json();
    state.latestPosterData = result;
    displayGeneratedPoster(result);
  } catch (err) {
    alert(`Error synthesizing poster: ${err.message}`);
    closeExhibition();
  } finally {
    state.isGenerating = false;
  }
}

// Display Generated Poster
function displayGeneratedPoster(data) {
  elements.posterLoader.classList.add('hidden');
  elements.posterResultImg.src = data.image_url || data.image_b64;
  elements.posterResultImg.style.opacity = '1';
  
  elements.modalTitle.textContent = `REACTION POSTER ${data.design_id || '#RX-001'}`;
  elements.postReaction.textContent = (data.label || state.currentEmotion).toUpperCase();
  elements.postStyle.textContent = (data.style || state.selectedStyle).toUpperCase();
  elements.postScore.textContent = `${data.confidence || 85}%`;
  elements.postDesignId.textContent = data.design_id || '#RX-042';
  elements.postTypography.textContent = data.typography || 'Helvetica / Grotesque Display';
  elements.postTimestamp.textContent = data.timestamp || new Date().toLocaleString();
  
  // Render Palette Swatches
  elements.swatchCluster.innerHTML = '';
  const palette = data.palette || ['#000000', '#FFFFFF', '#FF3B00', '#8E8E93'];
  palette.forEach((hex) => {
    const sw = document.createElement('div');
    sw.className = 'swatch-pill';
    sw.style.backgroundColor = hex;
    sw.textContent = hex;
    elements.swatchCluster.appendChild(sw);
  });
}

// Regenerate current reaction with same or tweaked settings
function regeneratePoster() {
  generatePosterAction();
}

// Close exhibition modal and resume live view
function closeExhibition() {
  elements.exhibitionModal.classList.add('hidden');
}

// Download Generated Poster
function downloadPoster() {
  if (!state.latestPosterData) return;
  const link = document.createElement('a');
  link.href = state.latestPosterData.download_url || state.latestPosterData.image_url;
  link.download = `REACTIONARY_${state.latestPosterData.design_id || 'POSTER'}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Auto bootstrap on window load
window.addEventListener('DOMContentLoaded', initApp);

