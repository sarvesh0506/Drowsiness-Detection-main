// App State variables
let currentStreamMode = 'backend'; // 'backend' or 'client'
let isCameraActive = true;
let isMuted = false;
let pollingInterval = null;
let clientProcessingInterval = null;
let webcamStream = null;
let biometricChart = null;

// Metrics History for charts
const MAX_CHART_POINTS = 40;
const chartData = {
    labels: [],
    ear: [],
    mar: [],
    fatigue: []
};

// Target DOM Elements
const connectionBadge = document.getElementById('connectionBadge');
const statusModeText = document.getElementById('statusModeText');
const backendStreamImg = document.getElementById('backendStreamImg');
const clientWebcamVideo = document.getElementById('clientWebcamVideo');
const clientProcessedCanvas = document.getElementById('clientProcessedCanvas');
const streamFallback = document.getElementById('streamFallback');
const toggleCamBtn = document.getElementById('toggleCamBtn');
const toggleCamText = document.getElementById('toggleCamText');
const toggleMuteBtn = document.getElementById('toggleMuteBtn');
const toggleMuteText = document.getElementById('toggleMuteText');
const streamFps = document.getElementById('streamFps');
const streamLatency = document.getElementById('streamLatency');

// Gauges & Stats Elements
const statusStateCard = document.getElementById('statusStateCard');
const statusMainText = document.getElementById('statusMainText');
const statusDescText = document.getElementById('statusDescText');
const fatigueGaugeCircle = document.getElementById('fatigueGaugeCircle');
const earGaugeCircle = document.getElementById('earGaugeCircle');
const fatigueValText = document.getElementById('fatigueVal');
const earValText = document.getElementById('earVal');
const statMarText = document.getElementById('statMar');
const statBlinksText = document.getElementById('statBlinks');
const statBlinkRateText = document.getElementById('statBlinkRate');
const statYawnsText = document.getElementById('statYawns');
const statHeadPoseText = document.getElementById('statHeadPose');
const timelineContainer = document.getElementById('timelineContainer');

// Settings modal inputs
const earThresholdRange = document.getElementById('earThresholdRange');
const earThresholdVal = document.getElementById('earThresholdVal');
const marThresholdRange = document.getElementById('marThresholdRange');
const marThresholdVal = document.getElementById('marThresholdVal');
const closedFramesRange = document.getElementById('closedFramesRange');
const closedFramesVal = document.getElementById('closedFramesVal');
const yawnFramesRange = document.getElementById('yawnFramesRange');
const yawnFramesVal = document.getElementById('yawnFramesVal');
const muteBuzzerSwitch = document.getElementById('muteBuzzerSwitch');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const clearLogsBtn = document.getElementById('clearLogsBtn');

// Stream Mode Toggles
const modeBackendBtn = document.getElementById('modeBackendBtn');
const modeClientBtn = document.getElementById('modeClientBtn');

let lastLoggedStatus = "Alert";

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    loadSettings();
    startApp();
    setupEventListeners();
});

function setupEventListeners() {
    // Mode toggling
    modeBackendBtn.addEventListener('click', () => switchStreamMode('backend'));
    modeClientBtn.addEventListener('click', () => switchStreamMode('client'));
    
    // Controls
    toggleCamBtn.addEventListener('click', toggleCamera);
    toggleMuteBtn.addEventListener('click', toggleMuteState);
    clearLogsBtn.addEventListener('click', () => {
        timelineContainer.innerHTML = '';
        appendLog('info', 'Audit timeline cleared.');
    });
    
    // Settings range indicators syncing
    earThresholdRange.addEventListener('input', (e) => earThresholdVal.textContent = parseFloat(e.target.value).toFixed(2));
    marThresholdRange.addEventListener('input', (e) => marThresholdVal.textContent = parseFloat(e.target.value).toFixed(2));
    closedFramesRange.addEventListener('input', (e) => closedFramesVal.textContent = e.target.value);
    yawnFramesRange.addEventListener('input', (e) => yawnFramesVal.textContent = e.target.value);
    
    // Save Settings
    saveSettingsBtn.addEventListener('click', saveSettings);
}

function startApp() {
    if (currentStreamMode === 'backend') {
        startBackendPolling();
    } else {
        startClientStreaming();
    }
}

// --- Settings Management ---
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();
        if (response.ok) {
            // Populate modal values
            earThresholdRange.value = data.ear_threshold;
            earThresholdVal.textContent = parseFloat(data.ear_threshold).toFixed(2);
            marThresholdRange.value = data.mar_threshold;
            marThresholdVal.textContent = parseFloat(data.mar_threshold).toFixed(2);
            closedFramesRange.value = data.closed_frames_threshold;
            closedFramesVal.textContent = data.closed_frames_threshold;
            yawnFramesRange.value = data.yawn_frames_threshold;
            yawnFramesVal.textContent = data.yawn_frames_threshold;
            
            muteBuzzerSwitch.checked = data.alarm_muted;
            isMuted = data.alarm_muted;
            updateMuteButtonVisuals();
        }
    } catch (e) {
        console.error("Error loading configurations:", e);
        appendLog('warning', 'Failed to retrieve configuration thresholds from server.');
    }
}

async function saveSettings() {
    const payload = {
        ear_threshold: parseFloat(earThresholdRange.value),
        mar_threshold: parseFloat(marThresholdRange.value),
        closed_frames: parseInt(closedFramesRange.value),
        yawn_frames: parseInt(yawnFramesRange.value),
        alarm_muted: muteBuzzerSwitch.checked
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (data.success) {
            isMuted = payload.alarm_muted;
            updateMuteButtonVisuals();
            appendLog('info', `Thresholds updated: EAR=${payload.ear_threshold}, MAR=${payload.mar_threshold}, Closed Frames=${payload.closed_frames}`);
            bootstrap.Modal.getInstance(document.getElementById('settingsModal')).hide();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (e) {
        console.error("Error saving configurations:", e);
        appendLog('danger', 'Failed to push configuration updates to server.');
    }
}

// --- UI Rendering Helpers ---
function updateGauge(circleElement, value, max) {
    const radius = 42;
    const circumference = 2 * Math.PI * radius; // 263.89
    const percentage = Math.max(0.0, Math.min(1.0, value / max));
    const offset = circumference - (percentage * circumference);
    circleElement.style.strokeDashoffset = offset;
}

function updateMuteButtonVisuals() {
    if (isMuted) {
        toggleMuteBtn.classList.remove('btn-glass');
        toggleMuteBtn.classList.add('btn-danger');
        toggleMuteText.textContent = "UNMUTE BUZZER";
        muteBuzzerSwitch.checked = true;
    } else {
        toggleMuteBtn.classList.remove('btn-danger');
        toggleMuteBtn.classList.add('btn-glass');
        toggleMuteText.textContent = "MUTE BUZZER";
        muteBuzzerSwitch.checked = false;
    }
}

function updateStatusDisplay(status, fatigue) {
    // Reset classes
    statusStateCard.classList.remove('card-glow-normal', 'card-glow-warning', 'card-glow-danger');
    statusMainText.classList.remove('text-glow', 'warning-glow', 'danger-glow');
    
    // Status text values
    statusMainText.textContent = status;
    
    const dot = connectionBadge.querySelector('.status-dot');
    
    if (status === 'Drowsy') {
        statusStateCard.classList.add('card-glow-danger');
        statusMainText.classList.add('danger-glow');
        statusDescText.textContent = "CRITICAL WARNING: Micro-sleep behaviors detected!";
        
        dot.className = "status-dot red";
        
        if (lastLoggedStatus !== "Drowsy") {
            appendLog('danger', `Drowsiness Alarm Triggered (Fatigue Score: ${fatigue.toFixed(1)}%)`);
            lastLoggedStatus = "Drowsy";
        }
    } else if (status === 'Warning' || status === 'Distracted') {
        statusStateCard.classList.add('card-glow-warning');
        statusMainText.classList.add('warning-glow');
        
        dot.className = "status-dot orange";
        
        if (status === 'Distracted') {
            statusDescText.textContent = "WARNING: Driver looking away or nodding head.";
            if (lastLoggedStatus !== "Distracted") {
                appendLog('warning', "Driver Distraction Detected - check eyes on road.");
                lastLoggedStatus = "Distracted";
            }
        } else {
            statusDescText.textContent = "WARNING: High drowsiness indicators detected.";
            if (lastLoggedStatus !== "Warning") {
                appendLog('warning', "Driver Warning: Fatigue accumulating.");
                lastLoggedStatus = "Warning";
            }
        }
    } else if (status === 'No Face Detected') {
        statusStateCard.classList.add('card-glow-normal');
        statusMainText.classList.add('text-glow');
        statusDescText.textContent = "Searching for driver's face mesh...";
        
        dot.className = "status-dot orange";
        
        if (lastLoggedStatus !== "No Face") {
            appendLog('info', "Webcam Face connection lost.");
            lastLoggedStatus = "No Face";
        }
    } else {
        statusStateCard.classList.add('card-glow-normal');
        statusMainText.classList.add('text-glow');
        statusDescText.textContent = "Driver shows normal attentive behaviors.";
        
        dot.className = "status-dot green";
        
        if (lastLoggedStatus !== "Alert" && lastLoggedStatus !== "No Face") {
            appendLog('info', "Driver recovered to Alert status.");
            lastLoggedStatus = "Alert";
        }
    }
}

function appendLog(level, message) {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    
    const logItem = document.createElement('div');
    logItem.className = `log-item ${level}`;
    logItem.innerHTML = `
        <span class="log-time font-orbitron">${timeStr}</span>
        <span class="log-msg">${message}</span>
    `;
    
    timelineContainer.appendChild(logItem);
    timelineContainer.scrollTop = timelineContainer.scrollHeight;
}

// --- Charting ---
function initChart() {
    const ctx = document.getElementById('biometricChart').getContext('2d');
    
    biometricChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Eye Ratio (EAR)',
                    data: [],
                    borderColor: '#00e5ff',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y1'
                },
                {
                    label: 'Mouth Ratio (MAR)',
                    data: [],
                    borderColor: '#d500f9',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    yAxisID: 'y1'
                },
                {
                    label: 'Fatigue Score %',
                    data: [],
                    borderColor: '#ff1744',
                    backgroundColor: 'rgba(255, 23, 68, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    yAxisID: 'y2'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#a0a0a0',
                        font: { family: 'Orbitron', size: 10 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { display: false }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0.0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#a0a0a0', font: { family: 'Orbitron' } }
                },
                y2: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#a0a0a0', font: { family: 'Orbitron' } }
                }
            }
        }
    });
}

function updateChart(ear, mar, fatigue) {
    if (!biometricChart) return;
    
    const now = new Date();
    const label = now.toTimeString().split(' ')[0];
    
    chartData.labels.push(label);
    chartData.ear.push(ear);
    chartData.mar.push(mar);
    chartData.fatigue.push(fatigue);
    
    if (chartData.labels.length > MAX_CHART_POINTS) {
        chartData.labels.shift();
        chartData.ear.shift();
        chartData.mar.shift();
        chartData.fatigue.shift();
    }
    
    biometricChart.data.labels = chartData.labels;
    biometricChart.data.datasets[0].data = chartData.ear;
    biometricChart.data.datasets[1].data = chartData.mar;
    biometricChart.data.datasets[2].data = chartData.fatigue;
    
    // Update chart without jarring transitions
    biometricChart.update('none');
}

// --- Stream Management ---
function switchStreamMode(mode) {
    if (currentStreamMode === mode) return;
    
    appendLog('info', `Switching to ${mode} mode...`);
    currentStreamMode = mode;
    
    // Stop current operations
    stopBackendPolling();
    stopClientStreaming();
    
    if (mode === 'backend') {
        // Visual button states
        modeBackendBtn.classList.add('active');
        modeClientBtn.classList.remove('active');
        
        statusModeText.textContent = "Backend Stream";
        
        // Hide client canvas, show backend img
        clientWebcamVideo.classList.add('d-none');
        clientProcessedCanvas.classList.add('d-none');
        backendStreamImg.classList.remove('d-none');
        streamFallback.classList.add('d-none');
        
        if (isCameraActive) {
            backendStreamImg.src = "/video_feed";
            startBackendPolling();
        }
    } else {
        // Visual button states
        modeClientBtn.classList.add('active');
        modeBackendBtn.classList.remove('active');
        
        statusModeText.textContent = "Client Processing";
        
        // Hide backend img, show client canvas
        backendStreamImg.classList.add('d-none');
        clientProcessedCanvas.classList.remove('d-none');
        streamFallback.classList.add('d-none');
        
        if (isCameraActive) {
            startClientStreaming();
        }
    }
}

function toggleCamera() {
    isCameraActive = !isCameraActive;
    
    if (isCameraActive) {
        toggleCamBtn.classList.remove('btn-danger');
        toggleCamBtn.classList.add('btn-glass');
        toggleCamText.textContent = "STOP CAMERA";
        
        appendLog('info', "Starting driver camera capture...");
        if (currentStreamMode === 'backend') {
            backendStreamImg.src = "/video_feed";
            backendStreamImg.classList.remove('d-none');
            streamFallback.classList.add('d-none');
            startBackendPolling();
        } else {
            startClientStreaming();
        }
    } else {
        toggleCamBtn.classList.remove('btn-glass');
        toggleCamBtn.classList.add('btn-danger');
        toggleCamText.textContent = "START CAMERA";
        
        appendLog('info', "Stopping driver camera capture.");
        
        // Stop pipelines
        stopBackendPolling();
        stopClientStreaming();
        
        // UI resetting
        backendStreamImg.src = "";
        backendStreamImg.classList.add('d-none');
        clientProcessedCanvas.classList.add('d-none');
        streamFallback.classList.remove('d-none');
        
        // Clear telemetry numbers
        streamFps.textContent = "--";
        streamLatency.textContent = "0ms";
        
        // Trigger REST call to stop background alarms
        fetch('/api/alarm/stop', { method: 'POST' }).catch(() => {});
    }
}

async function toggleMuteState() {
    const targetState = !isMuted;
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ alarm_muted: targetState })
        });
        const data = await response.json();
        if (data.success) {
            isMuted = targetState;
            updateMuteButtonVisuals();
            appendLog('info', isMuted ? 'Alarms MUTED.' : 'Alarms UNMUTED.');
        }
    } catch (e) {
        console.error("Mute state toggle error:", e);
    }
}

// --- Backend Mode Polling ---
function startBackendPolling() {
    stopBackendPolling();
    // Poll telemetry data every 150ms
    pollingInterval = setInterval(fetchBackendStatus, 150);
    appendLog('info', "Backend status telemetry polling connected.");
}

function stopBackendPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function fetchBackendStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        if (response.ok) {
            // Update dashboard statistics
            streamFps.textContent = data.fps.toFixed(1);
            streamLatency.textContent = "0ms (local)";
            
            earValText.textContent = data.ear.toFixed(3);
            fatigueValText.textContent = Math.round(data.fatigue_score);
            statMarText.textContent = data.mar.toFixed(3);
            statBlinksText.textContent = data.blink_count;
            statBlinkRateText.textContent = data.blink_rate;
            statYawnsText.textContent = data.yawn_count;
            
            const poseStr = `P: ${data.pitch.toFixed(1)}° | Y: ${data.yaw.toFixed(1)}° | R: ${data.roll.toFixed(1)}°`;
            statHeadPoseText.textContent = poseStr;
            
            // Sync Gauges
            updateGauge(fatigueGaugeCircle, data.fatigue_score, 100);
            updateGauge(earGaugeCircle, data.ear, 0.45); // Max EAR gauge boundary
            
            updateStatusDisplay(data.status, data.fatigue_score);
            updateChart(data.ear, data.mar, data.fatigue_score);
        }
    } catch (e) {
        console.error("Error fetching status:", e);
        stopBackendPolling();
        handleStreamError();
    }
}

// --- Client Mode Streaming ---
async function startClientStreaming() {
    stopClientStreaming();
    
    // Acquire webcam
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: { ideal: 30 } },
            audio: false
        });
        
        clientWebcamVideo.srcObject = webcamStream;
        clientWebcamVideo.classList.remove('d-none');
        clientProcessedCanvas.classList.remove('d-none');
        streamFallback.classList.add('d-none');
        
        // Hidden capture elements
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 480;
        const ctx = offscreenCanvas.getContext('2d');
        const renderCtx = clientProcessedCanvas.getContext('2d');
        
        appendLog('info', "Client browser webcam activated.");
        
        // Loop frame submissions at 12 FPS to optimize bandwidth/CPU
        clientProcessingInterval = setInterval(async () => {
            if (clientWebcamVideo.readyState === clientWebcamVideo.HAVE_ENOUGH_DATA) {
                // Capture video frame to offscreen canvas
                ctx.drawImage(clientWebcamVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
                const base64Img = offscreenCanvas.toDataURL('image/jpeg', 0.6); // 60% compression quality
                
                const t1 = performance.now();
                
                // POST to API
                try {
                    const response = await fetch('/api/process_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Img })
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        const t2 = performance.now();
                        const latency = Math.round(t2 - t1);
                        streamLatency.textContent = `${latency}ms`;
                        streamFps.textContent = "12.0 (Target)";
                        
                        // Render processed image with annotations to layout canvas
                        const processedImg = new Image();
                        processedImg.onload = () => {
                            clientProcessedCanvas.width = processedImg.width;
                            clientProcessedCanvas.height = processedImg.height;
                            renderCtx.drawImage(processedImg, 0, 0);
                        };
                        processedImg.src = data.processed_image;
                        
                        // Sync telemetry metrics
                        earValText.textContent = data.ear.toFixed(3);
                        fatigueValText.textContent = Math.round(data.fatigue_score);
                        statMarText.textContent = data.mar.toFixed(3);
                        statBlinksText.textContent = data.blink_count;
                        statBlinkRateText.textContent = data.blink_rate;
                        statYawnsText.textContent = data.yawn_count;
                        
                        const poseStr = `P: ${data.pitch.toFixed(1)}° | Y: ${data.yaw.toFixed(1)}° | R: ${data.roll.toFixed(1)}°`;
                        statHeadPoseText.textContent = poseStr;
                        
                        // Update Gauges
                        updateGauge(fatigueGaugeCircle, data.fatigue_score, 100);
                        updateGauge(earGaugeCircle, data.ear, 0.45);
                        
                        updateStatusDisplay(data.status, data.fatigue_score);
                        updateChart(data.ear, data.mar, data.fatigue_score);
                    }
                } catch (err) {
                    console.error("Frame processing POST error:", err);
                }
            }
        }, 83); // ~12 FPS
        
    } catch (e) {
        console.error("Webcam retrieval error:", e);
        appendLog('danger', `Camera access denied or unavailable: ${e.message}`);
        handleStreamError();
    }
}

function stopClientStreaming() {
    if (clientProcessingInterval) {
        clearInterval(clientProcessingInterval);
        clientProcessingInterval = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    clientWebcamVideo.srcObject = null;
    clientWebcamVideo.classList.add('d-none');
}

function handleStreamError() {
    backendStreamImg.classList.add('d-none');
    clientProcessedCanvas.classList.add('d-none');
    streamFallback.classList.remove('d-none');
    streamFps.textContent = "--";
    streamLatency.textContent = "0ms";
}
