const $ = (id) => document.getElementById(id);
const els = Object.fromEntries([
  'targetDrop','targetInput','targetPlaceholder','targetPreview','targetBadge','changeTarget','threshold','thresholdValue',
  'cameraTab','videoTab','videoInput','cameraSelect','fullscreen','stage','sourceVideo','overlay','captureCanvas','emptyState',
  'loadingState','sourceBadge','deviceBadge','stopBtn','pauseBtn','shotBtn','liveText','fpsValue','bestPhoto','bestPlaceholder',
  'bestScore','matchBadge','bestTrack','bestTime','bestCamera','bestFrame','peopleCount','tracksCount','statSimilarity','statFps',
  'historyBody','historySearch','statusFilter','toast','currentDate','currentTime','systemStatus'
].map(id => [id, $(id)]));

const state = {
  sessionId: null, socket: null, stream: null, source: null, busy: false, paused: false, running: false,
  lastSend: 0, boxes: [], history: [], lastHistory: new Map(), cameraName: 'Camera mặc định', best: null,
  sourceRequest: 0, targetObjectUrl: null,
};
const overlayCtx = els.overlay.getContext('2d');
const captureCtx = els.captureCanvas.getContext('2d', { alpha: false });

function toast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.className = `toast show${isError ? ' error' : ''}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => els.toast.className = 'toast', 3200);
}

function updateClock() {
  const now = new Date();
  els.currentDate.textContent = new Intl.DateTimeFormat('en-GB', { day:'2-digit', month:'short', year:'numeric' }).format(now);
  els.currentTime.textContent = now.toLocaleTimeString('en-GB');
}
setInterval(updateClock, 1000); updateClock();

fetch('/api/health').then(r => r.json()).then(info => {
  els.systemStatus.textContent = 'System Online';
  els.deviceBadge.textContent = info.device;
}).catch(() => els.systemStatus.textContent = 'System Offline');

els.threshold.addEventListener('input', () => els.thresholdValue.value = Number(els.threshold.value).toFixed(2));
els.targetDrop.addEventListener('click', () => els.targetInput.click());
els.changeTarget.addEventListener('click', () => els.targetInput.click());
els.targetInput.addEventListener('change', () => uploadTarget(els.targetInput.files[0]));
els.targetDrop.addEventListener('dragover', e => { e.preventDefault(); els.targetDrop.classList.add('drag'); });
els.targetDrop.addEventListener('dragleave', () => els.targetDrop.classList.remove('drag'));
els.targetDrop.addEventListener('drop', e => { e.preventDefault(); els.targetDrop.classList.remove('drag'); uploadTarget(e.dataTransfer.files[0]); });

async function uploadTarget(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return toast('Vui lòng chọn một tệp ảnh.', true);
  let targetReady = false;
  stopSearch();
  if (state.targetObjectUrl) URL.revokeObjectURL(state.targetObjectUrl);
  state.targetObjectUrl = URL.createObjectURL(file);
  els.targetPreview.src = state.targetObjectUrl;
  els.targetPreview.hidden = false;
  els.targetPlaceholder.hidden = true;
  els.targetBadge.hidden = true;
  els.loadingState.hidden = false;
  const data = new FormData(); data.append('file', file);
  try {
    const response = await fetch('/api/target', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Không thể xử lý ảnh');
    if (state.sessionId) fetch(`/api/session/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});
    state.sessionId = result.session_id;
    resetResults();
    els.targetBadge.hidden = false;
    els.targetBadge.textContent = result.person_detected ? '✓ Person detected' : '✓ Using full image';
    els.bestPhoto.src = result.preview;
    els.bestPhoto.style.display = 'block';
    els.bestPlaceholder.style.display = 'none';
    targetReady = true;
    toast('Đã trích xuất đặc trưng người cần tìm.');
  } catch (error) {
    els.targetPreview.hidden = true;
    els.targetPlaceholder.hidden = false;
    toast(error.message, true);
  }
  finally { els.loadingState.hidden = true; }
  // Camera is the default source: start it as soon as a target is ready.
  // The Camera tab remains available for retrying after a denied permission.
  if (targetReady) await startCamera();
}

els.cameraTab.addEventListener('click', startCamera);
els.videoTab.addEventListener('click', () => {
  if (!state.sessionId) return toast('Hãy tải ảnh người cần tìm trước.', true);
  els.videoInput.click();
});
els.videoInput.addEventListener('change', () => startVideo(els.videoInput.files[0]));
els.cameraSelect.addEventListener('change', () => { if (state.source === 'camera') startCamera(); });

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const selected = els.cameraSelect.value;
    els.cameraSelect.innerHTML = cameras.map((d, i) => `<option value="${d.deviceId}">${escapeHtml(d.label || `Camera ${i + 1}`)}</option>`).join('');
    if (selected && cameras.some(c => c.deviceId === selected)) els.cameraSelect.value = selected;
  } catch (_) {}
}

async function startCamera() {
  if (!state.sessionId) return toast('Hãy tải ảnh người cần tìm trước.', true);
  const requestId = ++state.sourceRequest;
  state.source = 'switching';
  state.running = false; state.busy = false;
  state.socket?.close(); state.socket = null;
  stopMediaOnly();
  try {
    const deviceId = els.cameraSelect.value;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } }, audio: false
    });
    if (requestId !== state.sourceRequest) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    state.stream = stream;
    els.sourceVideo.srcObject = stream;
    els.sourceVideo.removeAttribute('src');
    await els.sourceVideo.play();
    await listCameras();
    state.cameraName = els.cameraSelect.options[els.cameraSelect.selectedIndex]?.text || 'Camera';
    activateSource('camera');
  } catch (error) {
    if (requestId === state.sourceRequest) stopSearch();
    toast(`Không mở được camera: ${error.message}`, true);
  }
}

async function startVideo(file) {
  if (!file) return;
  const requestId = ++state.sourceRequest;
  state.source = 'switching';
  state.running = false; state.busy = false;
  state.socket?.close(); state.socket = null;
  stopMediaOnly();
  els.sourceVideo.srcObject = null;
  els.sourceVideo.src = URL.createObjectURL(file);
  els.sourceVideo.loop = false;
  try {
    await els.sourceVideo.play();
    if (requestId !== state.sourceRequest) return;
    state.cameraName = file.name;
    activateSource('video');
  } catch (error) {
    if (requestId === state.sourceRequest) stopSearch();
    toast(`Không mở được video: ${error.message}`, true);
  }
}

function activateSource(source) {
  state.source = source; state.paused = false; state.running = true;
  els.stage.classList.remove('empty'); els.emptyState.hidden = true;
  els.sourceBadge.hidden = false; els.deviceBadge.hidden = false;
  els.sourceBadge.textContent = source.toUpperCase();
  els.cameraTab.classList.toggle('active', source === 'camera');
  els.videoTab.classList.toggle('active', source === 'video');
  els.stopBtn.disabled = els.pauseBtn.disabled = els.shotBtn.disabled = false;
  els.pauseBtn.innerHTML = 'Ⅱ &nbsp; Pause';
  connectSocket();
}

function connectSocket() {
  if (state.socket) state.socket.close();
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  state.socket = new WebSocket(`${protocol}://${location.host}/ws/search/${state.sessionId}`);
  state.socket.onopen = () => { els.loadingState.hidden = false; els.liveText.textContent = 'Starting'; };
  state.socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === 'ready') {
      els.loadingState.hidden = true; state.busy = false; els.liveText.textContent = 'Live';
      if (state.source === 'video' && els.sourceVideo.paused && !state.paused) els.sourceVideo.play();
      document.querySelector('.live-dot').classList.add('on'); requestAnimationFrame(frameLoop); return;
    }
    if (message.type === 'error') { state.busy = false; els.loadingState.hidden = true; toast(message.message, true); return; }
    if (message.type === 'result') { state.busy = false; updateResult(message); }
  };
  state.socket.onerror = () => { state.busy = false; els.loadingState.hidden = true; toast('Mất kết nối với máy chủ AI.', true); };
}

function frameLoop(now) {
  if (!state.running) return;
  if (!state.paused && !state.busy && state.socket?.readyState === WebSocket.OPEN && now - state.lastSend >= 65 && els.sourceVideo.readyState >= 2) {
    const vw = els.sourceVideo.videoWidth, vh = els.sourceVideo.videoHeight;
    const scale = Math.min(1, 1280 / Math.max(vw, vh));
    const w = Math.max(2, Math.round(vw * scale / 2) * 2), h = Math.max(2, Math.round(vh * scale / 2) * 2);
    els.captureCanvas.width = w; els.captureCanvas.height = h;
    captureCtx.drawImage(els.sourceVideo, 0, 0, w, h);
    const data = els.captureCanvas.toDataURL('image/jpeg', .78);
    state.busy = true; state.lastSend = now;
    state.socket.send(JSON.stringify({ type:'frame', data, threshold:Number(els.threshold.value) }));
  }
  requestAnimationFrame(frameLoop);
}

function updateResult(message) {
  state.boxes = message.boxes;
  drawBoxes(message.width, message.height);
  const stats = message.stats;
  els.peopleCount.textContent = stats.people;
  els.tracksCount.textContent = stats.tracks;
  els.fpsValue.textContent = stats.fps.toFixed(1);
  els.statFps.textContent = stats.fps.toFixed(1);
  els.deviceBadge.textContent = stats.device;
  if (message.best) updateBest(message.best);
  appendHistory(message);
}

function drawBoxes(width, height) {
  els.overlay.width = width; els.overlay.height = height;
  overlayCtx.clearRect(0, 0, width, height);
  state.boxes.forEach(box => {
    const [x1,y1,x2,y2] = box.bbox, w = x2-x1, h = y2-y1;
    const color = box.matched ? '#f1263f' : '#8fa0b4';
    overlayCtx.strokeStyle = color; overlayCtx.lineWidth = Math.max(2, width / 500); overlayCtx.strokeRect(x1,y1,w,h);
    const label = `ID: ${box.track_id}   ${(box.similarity*100).toFixed(1)}%`;
    overlayCtx.font = `600 ${Math.max(12, width / 75)}px Inter, sans-serif`;
    const tw = overlayCtx.measureText(label).width + 16, th = Math.max(24, height/23);
    overlayCtx.fillStyle = color; overlayCtx.fillRect(x1, Math.max(0, y1-th), tw, th);
    overlayCtx.fillStyle = '#fff'; overlayCtx.fillText(label, x1+8, Math.max(16, y1-th/3));
    if (box.matched) {
      overlayCtx.fillStyle = color; overlayCtx.fillRect(x1, y2, Math.min(92, w), 25);
      overlayCtx.fillStyle = '#fff'; overlayCtx.fillText('TARGET', x1+8, y2+18);
    }
  });
}

function updateBest(best) {
  state.best = best;
  const pct = `${(best.similarity * 100).toFixed(1)}%`;
  els.bestScore.textContent = pct; els.statSimilarity.textContent = pct;
  els.bestTrack.textContent = best.track_id; els.bestFrame.textContent = best.frame;
  els.bestTime.textContent = new Date(best.time).toLocaleString('vi-VN');
  els.bestCamera.textContent = state.cameraName;
  els.matchBadge.textContent = best.matched ? 'MATCHED' : 'TRACKING';
  els.matchBadge.classList.toggle('waiting', !best.matched);
  if (best.crop) { els.bestPhoto.src = best.crop; els.bestPhoto.style.display = 'block'; els.bestPlaceholder.style.display = 'none'; }
}

function resetResults() {
  state.best = null;
  state.boxes = [];
  state.history = [];
  state.lastHistory.clear();
  els.bestScore.textContent = '—';
  els.statSimilarity.textContent = '—';
  els.bestTrack.textContent = els.bestTime.textContent = els.bestCamera.textContent = els.bestFrame.textContent = '—';
  els.matchBadge.textContent = 'WAITING';
  els.matchBadge.classList.add('waiting');
  els.peopleCount.textContent = els.tracksCount.textContent = '0';
  els.fpsValue.textContent = els.statFps.textContent = '0.0';
  renderHistory();
}

function appendHistory(message) {
  const now = Date.now();
  for (const box of [...message.boxes].sort((a,b) => b.similarity-a.similarity)) {
    const previous = state.lastHistory.get(box.track_id) || 0;
    if (now - previous < (box.matched ? 850 : 2200)) continue;
    state.lastHistory.set(box.track_id, now);
    state.history.unshift({
      time: new Date(), trackId: box.track_id, similarity: box.similarity,
      matched: box.matched,
      snapshot: cropSnapshot(box.bbox, message.width, message.height),
    });
  }
  state.history = state.history.slice(0, 100);
  renderHistory();
}

function cropSnapshot(bbox, frameWidth, frameHeight) {
  if (!els.captureCanvas.width || !els.captureCanvas.height) return '';
  const [x1, y1, x2, y2] = bbox;
  const padX = Math.round((x2 - x1) * .04);
  const padY = Math.round((y2 - y1) * .03);
  const sx = Math.max(0, x1 - padX);
  const sy = Math.max(0, y1 - padY);
  const sw = Math.max(1, Math.min(frameWidth, x2 + padX) - sx);
  const sh = Math.max(1, Math.min(frameHeight, y2 + padY) - sy);
  const scale = Math.min(1, 180 / Math.max(sw, sh));
  const thumb = document.createElement('canvas');
  thumb.width = Math.max(1, Math.round(sw * scale));
  thumb.height = Math.max(1, Math.round(sh * scale));
  thumb.getContext('2d').drawImage(els.captureCanvas, sx, sy, sw, sh, 0, 0, thumb.width, thumb.height);
  return thumb.toDataURL('image/jpeg', .76);
}

function renderHistory() {
  const search = els.historySearch.value.trim().toLowerCase(), status = els.statusFilter.value;
  const rows = state.history.filter(row => (!search || String(row.trackId).includes(search)) && (status === 'all' || (status === 'matched') === row.matched));
  if (!rows.length) { els.historyBody.innerHTML = '<tr class="empty-row"><td colspan="6">Không có kết quả phù hợp.</td></tr>'; return; }
  els.historyBody.innerHTML = rows.map((row, i) => `<tr>
    <td>${i+1}</td><td>${row.time.toLocaleString('vi-VN')}</td><td>${row.trackId}</td>
    <td class="${row.matched?'sim-match':''}">${(row.similarity*100).toFixed(1)}%</td>
    <td><span class="status ${row.matched?'matched':'tracking'}">${row.matched?'MATCHED':'Tracking'}</span></td>
    <td>${row.snapshot?`<img class="thumb" src="${row.snapshot}" alt="Track ${row.trackId}">`:'—'}</td></tr>`).join('');
}
els.historySearch.addEventListener('input', renderHistory);
els.statusFilter.addEventListener('change', renderHistory);

els.pauseBtn.addEventListener('click', () => {
  state.paused = !state.paused;
  if (state.paused) els.sourceVideo.pause(); else els.sourceVideo.play();
  els.pauseBtn.innerHTML = state.paused ? '▶ &nbsp; Resume' : 'Ⅱ &nbsp; Pause';
  els.liveText.textContent = state.paused ? 'Paused' : 'Live';
});
els.stopBtn.addEventListener('click', stopSearch);
els.fullscreen.addEventListener('click', () => els.stage.requestFullscreen?.());
els.shotBtn.addEventListener('click', takeScreenshot);
els.sourceVideo.addEventListener('ended', () => {
  // Removing an old video source can dispatch `ended` after a camera switch.
  // Only a genuinely active file-video is allowed to stop the processing loop.
  if (state.source !== 'video') return;
  state.running = false;
  els.liveText.textContent = 'Ended';
  document.querySelector('.live-dot').classList.remove('on');
});

function takeScreenshot() {
  if (!els.overlay.width) return;
  const c = document.createElement('canvas'); c.width = els.overlay.width; c.height = els.overlay.height;
  const ctx = c.getContext('2d'); ctx.drawImage(els.sourceVideo, 0, 0, c.width, c.height); ctx.drawImage(els.overlay, 0, 0);
  const a = document.createElement('a'); a.download = `person-search-${Date.now()}.jpg`; a.href = c.toDataURL('image/jpeg', .92); a.click();
}

function stopMediaOnly() {
  state.stream?.getTracks().forEach(track => track.stop()); state.stream = null;
  if (els.sourceVideo.src?.startsWith('blob:')) URL.revokeObjectURL(els.sourceVideo.src);
  els.sourceVideo.pause(); els.sourceVideo.srcObject = null; els.sourceVideo.removeAttribute('src');
}
function stopSearch() {
  state.sourceRequest += 1;
  state.running = false; state.busy = false; state.source = null; state.boxes = [];
  state.socket?.close(); state.socket = null; stopMediaOnly();
  overlayCtx.clearRect(0,0,els.overlay.width,els.overlay.height);
  els.stage.classList.add('empty'); els.emptyState.hidden = false; els.loadingState.hidden = true;
  els.sourceBadge.hidden = els.deviceBadge.hidden = true;
  els.stopBtn.disabled = els.pauseBtn.disabled = els.shotBtn.disabled = true;
  els.liveText.textContent = 'Idle'; document.querySelector('.live-dot').classList.remove('on');
}
function escapeHtml(value) { const d=document.createElement('div'); d.textContent=value; return d.innerHTML; }
