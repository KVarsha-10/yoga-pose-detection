(() => {
  const $ = (id) => document.getElementById(id);
  const setup = $('setup');
  const practice = $('practice');
  const library = $('poseLibrary');
  const selected = $('selectedSequence');
  let sequence = [];
  let lastPromptId = 0;
  let lastPrompt = '';
  let pollHandle = null;
  let toastHandle = null;
  let voices = [];

  const voiceEnabled = () => $('voiceEnabled').checked && 'speechSynthesis' in window;
  function refreshVoices() { voices = window.speechSynthesis?.getVoices() || []; }
  if ('speechSynthesis' in window) {
    refreshVoices();
    window.speechSynthesis.addEventListener('voiceschanged', refreshVoices);
  }
  function speak(text, interrupt = true) {
    if (!text || !voiceEnabled()) return;
    const synth = window.speechSynthesis;
    if (interrupt) synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-IN';
    utterance.rate = 0.92;
    utterance.pitch = 1;
    const preferred = voices.find(v => /^en-IN/i.test(v.lang)) || voices.find(v => /^en/i.test(v.lang));
    if (preferred) utterance.voice = preferred;
    synth.speak(utterance);
  }
  function toast(message) {
    const el = $('toast'); el.textContent = message; el.classList.add('show');
    clearTimeout(toastHandle); toastHandle = setTimeout(() => el.classList.remove('show'), 3200);
  }
  function drawSequence() {
    $('sequenceCount').textContent = `${sequence.length} selected`;
    if (!sequence.length) {
      selected.innerHTML = '<div class="empty-sequence">Select poses from the library<br>to build your flow.</div>';
      return;
    }
    selected.replaceChildren();
    sequence.forEach((pose, i) => {
      const row = document.createElement('div'); row.className = 'selected-item';
      const number = document.createElement('span'); number.className = 'index'; number.textContent = String(i + 1).padStart(2, '0');
      const name = document.createElement('span'); name.className = 'pose-name'; name.textContent = pose;
      row.append(number, name);
      [['↑', 'Move up', -1], ['↓', 'Move down', 1], ['×', 'Remove pose', 0]].forEach(([symbol, title, delta]) => {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = symbol; button.title = title;
        button.addEventListener('click', () => {
          if (delta === 0) sequence.splice(i, 1);
          else { const to = i + delta; if (to < 0 || to >= sequence.length) return; [sequence[i], sequence[to]] = [sequence[to], sequence[i]]; }
          drawSequence();
        }); row.append(button);
      });
      selected.append(row);
    });
  }
  function addPose(pose) {
    if (!pose || sequence.includes(pose)) return;
    sequence.push(pose); drawSequence();
  }
  library.addEventListener('click', event => {
    const button = event.target.closest('[data-pose]');
    if (!button) return;
    addPose(button.dataset.pose); button.classList.add('selected');
  });
  $('addPose').addEventListener('click', () => {
    const button = library.querySelector('.pose-option:not(.hidden)');
    if (button) { addPose(button.dataset.pose); button.classList.add('selected'); }
  });
  $('addAll').addEventListener('click', () => { window.YOGA_POSES.forEach(addPose); });
  $('poseSearch').addEventListener('input', event => {
    const query = event.target.value.trim().toLocaleLowerCase();
    library.querySelectorAll('.pose-option').forEach(button => {
      button.classList.toggle('hidden', !button.dataset.pose.toLocaleLowerCase().includes(query));
    });
  });
  $('restEnabled').addEventListener('change', event => {
    $('restSeconds').disabled = !event.target.checked;
    $('restSeconds').parentElement.style.opacity = event.target.checked ? '1' : '.45';
  });

  async function post(path, body = {}) {
    const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The request could not be completed.');
    return result;
  }
  $('startButton').addEventListener('click', async () => {
    if (!sequence.length) { toast('Add at least one pose to your sequence first.'); return; }
    const hold = Number($('holdSeconds').value);
    const rest = $('restEnabled').checked ? Number($('restSeconds').value) : 0;
    if (!Number.isInteger(hold) || hold < 1 || hold > 120 || !Number.isInteger(rest) || rest < 0 || rest > 120) {
      toast('Enter whole seconds. Hold: 1–120; rest: 0–120.'); return;
    }
    const button = $('startButton'); button.disabled = true; button.firstChild.textContent = 'Starting camera… ';
    // Speak immediately from the button gesture so browser autoplay policies permit later cues.
    if (voiceEnabled()) { window.speechSynthesis.cancel(); speak('Starting your yoga practice.'); }
    try {
      const state = await post('/api/start', { sequence, hold_seconds: hold, rest_seconds: rest });
      setup.classList.add('hidden'); practice.classList.remove('hidden');
      $('cameraFeed').src = `/video_feed?ts=${Date.now()}`;
      $('cameraFeed').classList.remove('hidden'); $('cameraPlaceholder').classList.remove('hidden');
      lastPromptId = 0; updateState(state); clearInterval(pollHandle); pollHandle = setInterval(pollState, 400);
    } catch (error) {
      toast(error.message);
      if (voiceEnabled()) speak(error.message);
    } finally { button.disabled = false; button.innerHTML = 'Start my practice <span>→</span>'; }
  });
  async function pollState() {
    try { const response = await fetch('/api/state', { cache: 'no-store' }); updateState(await response.json()); }
    catch (_) { $('cameraStatus').textContent = 'Connection interrupted'; }
  }
  function updateState(state) {
    if (state.prompt_id && state.prompt_id > lastPromptId) {
      lastPromptId = state.prompt_id; lastPrompt = state.prompt || '';
      if (lastPrompt) speak(lastPrompt);
    }
    $('targetPose').textContent = state.target || (state.complete ? 'Flow complete' : '—');
    $('practiceTitle').textContent = state.complete ? 'Beautifully done.' : (state.paused ? 'Take your time.' : 'Find your space');
    $('progress').textContent = `Step ${state.total ? Math.min(state.step, state.total) : 0} of ${state.total || 0}`;
    $('stepNumber').textContent = String(state.step || 1).padStart(2, '0');
    $('timer').innerHTML = `${String(state.remaining ?? state.hold_seconds ?? 0).padStart(2, '0')}<span>s</span>`;
    $('feedback').textContent = state.feedback || 'Move into the target pose to begin.';
    $('detectedText').textContent = state.detected || 'Waiting for camera';
    $('cameraStatus').textContent = state.active ? (state.paused ? 'Practice paused' : 'Camera connected') : 'Session ended';
    $('modeBadge').textContent = ({holding: 'HOLD STEADY', rest: 'REST BREAK', paused: 'PAUSED', complete: 'COMPLETE', waiting: 'GET READY'})[state.mode] || 'GET READY';
    $('timerLabel').textContent = state.mode === 'rest' ? 'REST TIMER' : 'HOLD TIMER';
    $('pauseButton').innerHTML = state.paused ? '▶ <span>Resume</span>' : 'Ⅱ <span>Pause</span>';
    if (state.active && !state.complete && $('cameraFeed').naturalWidth) $('cameraPlaceholder').classList.add('hidden');
    if (state.target) setReference(state.target);
    if (state.complete) { $('cameraStatus').textContent = 'Sequence complete'; clearInterval(pollHandle); }
  }
  let referenceTarget = '';
  function setReference(pose) {
    if (pose === referenceTarget) return;
    referenceTarget = pose;
    const image = $('referenceImage'); const placeholder = $('referencePlaceholder');
    image.onload = () => { image.classList.remove('hidden'); placeholder.classList.add('hidden'); };
    image.onerror = () => { image.classList.add('hidden'); placeholder.classList.remove('hidden'); };
    image.src = `/reference/${encodeURIComponent(pose)}?ts=${Date.now()}`;
  }
  $('cameraFeed').addEventListener('load', () => $('cameraPlaceholder').classList.add('hidden'));
  $('cameraFeed').addEventListener('error', () => { $('cameraPlaceholder').classList.remove('hidden'); $('cameraPlaceholder').querySelector('p').textContent = 'Camera preview is unavailable.'; });
  $('pauseButton').addEventListener('click', async () => {
    try { updateState(await post('/api/pause', { paused: $('pauseButton').textContent.toLowerCase().includes('resume') ? false : true })); }
    catch (e) { toast(e.message); }
  });
  $('skipButton').addEventListener('click', async () => {
    try { updateState(await post('/api/skip')); }
    catch (e) { toast(e.message); }
  });
  $('repeatButton').addEventListener('click', () => {
    if (lastPrompt) speak(lastPrompt);
    else if ($('targetPose').textContent !== '—') speak(`Current pose: ${$('targetPose').textContent}. Hold for ${$('holdSeconds').value} seconds.`);
  });
  $('endButton').addEventListener('click', async () => {
    clearInterval(pollHandle);
    try { await post('/api/stop'); } catch (_) {}
    window.speechSynthesis?.cancel();
    $('cameraFeed').src = ''; practice.classList.add('hidden'); setup.classList.remove('hidden');
    $('cameraPlaceholder').classList.remove('hidden'); $('cameraPlaceholder').querySelector('p').textContent = 'Starting camera and pose detection…';
  });
  $('voiceEnabled').addEventListener('change', event => {
    if (!event.target.checked) window.speechSynthesis?.cancel();
    else speak('Voice guidance is on.');
  });
  window.addEventListener('beforeunload', () => { clearInterval(pollHandle); navigator.sendBeacon('/api/stop'); });
  drawSequence();
})();
