// --- DOM Element Selection (updated) ---
const permissionScreen = document.getElementById('permission-screen');
const focusScreen = document.getElementById('focus-screen');
const alertScreen = document.getElementById('alert-screen');
const cameraFeed = document.getElementById('camera-feed');
const snapshotCanvas = document.getElementById('snapshot-canvas');
const userChoiceDiv = document.getElementById('user-choice');
const challengeSection = document.getElementById('challenge-section');
const challengeInput = document.getElementById('challenge-input');
const feedbackMessage = document.getElementById('feedback-message');
const alertMessage = document.getElementById('alert-message');
const breakSection = document.getElementById('break-section');
const breakTimeSelect = document.getElementById('break-time');
const breakFeedback = document.getElementById('break-feedback');

// Global variables
let detectionInterval;
const correctPuzzleAnswer = 21;

/**
/**
/**
 * 1. Requests Camera Permission and starts the monitoring process. (FINAL VERSION)
 */
async function startMonitoring() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraFeed.srcObject = stream;

        // Listener 1: Sets the canvas size as soon as metadata is known.
        cameraFeed.onloadedmetadata = () => {
            console.log("Metadata loaded. Video dimensions:", cameraFeed.videoWidth, "x", cameraFeed.videoHeight);
            snapshotCanvas.width = cameraFeed.videoWidth;
            snapshotCanvas.height = cameraFeed.videoHeight;
            cameraFeed.play();
        };

        // Listener 2: Switches screens and starts the loop as soon as the video plays.
        cameraFeed.onplaying = () => {
            console.log("Video is now playing.");
            permissionScreen.classList.add('hidden');
            focusScreen.classList.remove('hidden');

            // Start the detection loop
            if (detectionInterval) clearInterval(detectionInterval);
            detectionInterval = setInterval(sendFrameForDetection, 200);
        };

    } catch (err) {
        alert(`Camera access denied or failed: ${err.message}.`);
        console.error("Camera access error:", err);
    }
}/**
 * 2. Sends a single frame from the webcam to the Python backend.
 */
async function sendFrameForDetection() {
    console.log("Sending frame for detection...");
    if (cameraFeed.readyState < 2) return;

    const context = snapshotCanvas.getContext('2d');
    context.drawImage(cameraFeed, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
    await new Promise(requestAnimationFrame);

    const imageData = snapshotCanvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('http://127.0.0.1:5000/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        const result = await response.json();
        console.log("Response from backend:", result);

        if (result.is_drowsy) {
            triggerDrowsinessAlert();
        }
    } catch (error) {
        console.error("Error sending frame:", error);
    }
}

/**
 * 3. Triggers the drowsiness alert UI.
 */
function triggerDrowsinessAlert() {
    clearInterval(detectionInterval);

    focusScreen.classList.add('hidden');
    alertScreen.classList.remove('hidden');

    userChoiceDiv.classList.remove('hidden');
    challengeSection.classList.add('hidden');
    breakSection.classList.add('hidden');
    feedbackMessage.classList.add('hidden');
    breakFeedback.classList.add('hidden');
    challengeInput.value = '';

    alertMessage.textContent = "😴 Hey there! Looks like you're sleepy.";
}

/**
 * 4. Handles the user's initial choice after the alert.
 */
function handleUserChoice(choice) {
    userChoiceDiv.classList.add('hidden');

    if (choice === 'break') {
        alertMessage.textContent = "👍 Good choice! Time for a short rest.";
        breakSection.classList.remove('hidden');
    } else if (choice === 'continue') {
        alertMessage.textContent = "🧠 Great! Complete this quick challenge to boost your focus.";
        challengeSection.classList.remove('hidden');
        challengeInput.focus();
    }
}

/**
 * 5. Logic to start the break timer/alarm.
 */
function startBreakAlarm() {
    const breakTime = parseInt(breakTimeSelect.value);
    const breakTimeMs = breakTime * 60 * 1000;

    breakFeedback.classList.remove('hidden');
    breakFeedback.textContent = `Alarm set for ${breakTime} minutes. Enjoy your break!`;

    breakSection.querySelector('select').classList.add('hidden');
    breakSection.querySelector('button').classList.add('hidden');

    setTimeout(() => {
        alertMessage.textContent = "⏰ Break time is OVER! Ready to focus?";
        const resumeButton = document.createElement('button');
        resumeButton.className = 'btn primary large';
        resumeButton.textContent = "Resume Focus";
        resumeButton.onclick = resetToFocus;
        breakSection.appendChild(resumeButton);
    }, breakTimeMs / 10); // DEMO: Shortened for quick testing
}

/**
 * 6. Logic to check the answer for the quick puzzle.
 */
function checkAnswer() {
    const userAnswer = parseInt(challengeInput.value);
    feedbackMessage.classList.remove('hidden');

    if (userAnswer === correctPuzzleAnswer) {
        feedbackMessage.textContent = "✅ Correct! Welcome back to the zone.";
        setTimeout(resetToFocus, 2000);
    } else {
        feedbackMessage.textContent = "❌ Incorrect. Try again or take a break.";
        challengeInput.value = '';
    }
}

/**
 * 7. Resets the UI back to the Focus Screen.
 */
function resetToFocus() {
    alertScreen.classList.add('hidden');
    focusScreen.classList.remove('hidden');

    detectionInterval = setInterval(sendFrameForDetection, 200);

    breakSection.querySelector('select').classList.remove('hidden');
    breakSection.querySelector('button').classList.remove('hidden');

    const resumeButton = breakSection.querySelector('.large');
    if (resumeButton) {
        breakSection.removeChild(resumeButton);
    }
}

