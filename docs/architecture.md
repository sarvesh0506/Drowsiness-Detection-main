# Technical Architecture & System Diagrams

This document contains visual diagrams mapping out the design, workflow, interaction sequences, and deployment topography of the **Driver Drowsiness Detection System**.

---

## 1. System Architecture Diagram
Describes how the client interface, Flask web server, and asynchronous background engines interact.

```mermaid
graph TB
    subgraph Client [Client-Side Browser Interface]
        UI[Futuristic Web Dashboard]
        CamCap[HTML5 Camera Capturer]
        C2D[2D Canvas Overlay Renderer]
        CJS[Chart.js Telemetry Graphing]
    end

    subgraph Server [Flask Web Server Backend]
        App[app.py: Flask Factory/Routing]
        CamEng[camera.py: VideoCamera Core Engine]
        Util[utils.py: Facial Metrics Algebra]
        Alarm[alarm.py: Threaded Alarm Manager]
        CNN[TensorFlow Eye Classifier Model]
        Logs[Rotating Application Logger]
    end

    %% Client inputs
    CamCap -- Base64 Frames --> App
    App -- Render HTML/Assets --> UI
    
    %% Backend pipeline
    App -- Stream Request --> CamEng
    CamEng -- Process Landmarks --> Util
    CamEng -- Classify Crop --> CNN
    CamEng -- Update Fatigue Score --> Alarm
    CamEng -- Logging --> Logs
    
    %% Outputs
    CamEng -- MJPEG WebStream / REST Status JSON --> App
    App -- API Response Payload --> UI
    UI -- Extract Telemetry --> CJS
    UI -- Render Annotations --> C2D
```

---

## 2. Dynamic Workflow Diagram
Details the pipeline sequence for each frame captured by the system.

```mermaid
graph TD
    A[Capture Video Frame] --> B{Stream Mode?}
    B -- Local --> C[Read OpenCV cv2.VideoCapture]
    B -- Cloud --> D[Receive Base64 String via REST API]
    C --> E[Preprocess Frame to RGB]
    D --> E
    E --> F[Execute MediaPipe Face Mesh]
    F --> G{Face Detected?}
    
    G -- No --> H[Set Status: No Face Detected]
    G -- Yes --> I[Extract Left/Right Eye & Lip Contour Points]
    
    I --> J[Calculate Eye Aspect Ratio EAR]
    I --> K[Calculate Mouth Aspect Ratio MAR]
    I --> L[Solve Head Pose PNP - Pitch/Yaw/Roll]
    
    J --> M{TensorFlow CNN Classifier Loaded?}
    M -- Yes --> N[Crop Eyes + Predict via CNN]
    M -- No --> O[Fallback: Geometric-only Estimation]
    
    N --> P[Weighted State Decision: Open vs. Closed]
    O --> P
    
    P --> Q[Execute Fatigue State-Machine Counters]
    K --> Q
    L --> Q
    
    Q --> R[Calculate Driver Fatigue Score 0-100%]
    R --> S{Fatigue Score > 70%?}
    
    S -- Yes --> T[Trigger Background Audio Alarm & Status: Drowsy]
    S -- No --> U{Distracted Pose Threshold Exceeded?}
    
    U -- Yes --> V[Status: Distracted & Increment Fatigue]
    U -- No --> W[Status: Alert / Warning & Mute Alarms]
    
    T --> X[Render Landmark Overlays & Project Pose Vector]
    V --> X
    W --> X
    X --> Y[Return Annotated Frame & JSON Payload]
```

---

## 3. Interaction Sequence Diagram
Depicts the step-by-step transaction between client browser interactions, API routing, processing modules, and deep learning components.

```mermaid
sequenceDiagram
    autonumber
    actor Driver as Driver / User
    participant Browser as Web Browser Client
    participant App as Flask Router (app.py)
    participant Cam as Camera Engine (camera.py)
    participant TF as TensorFlow Predictor
    participant Alarm as Alarm Thread (alarm.py)

    Driver->>Browser: Open Dashboard UI
    Browser->>App: Request index.html / GET API config
    App-->>Browser: Return HTML Page & Threshold settings
    
    rect rgb(15, 30, 50)
        Note over Browser, Alarm: Real-time Frame Analysis Loop (12 FPS)
        Browser->>Browser: Capture webcam raw image
        Browser->>App: POST /api/process_frame (base64 JSON)
        App->>Cam: process_base64_frame(image)
        Cam->>Cam: Extract landmarks (Face Mesh)
        Cam->>TF: Predict crop matrices (Closed/Open)
        TF-->>Cam: Return float confidence scores
        Cam->>Cam: Calculate EAR, MAR, Head Pose angles
        Cam->>Cam: Increment fatigue state metrics
        
        alt Drowsy detected (>15 consec frames)
            Cam->>Alarm: trigger()
            Alarm->>Alarm: Play alert buzzer (thread loop)
        else Alert / Recovered
            Cam->>Alarm: stop()
            Alarm->>Alarm: Halt audio playback
        end
        
        Cam-->>App: Return annotated frame data & metrics JSON
        App-->>Browser: Return 200 OK Response
        Browser->>Browser: Render output canvas & update Chart.js
    end
```

---

## 4. Decision Flowchart
Logic flow for calculating fatigue scores based on sensor inputs.

```mermaid
flowchart TD
    Start([Frame Metrics Ingestion]) --> CheckClosed{Is eye closed?}
    
    CheckClosed -- Yes --> ConsecClosed[Increment Closed Frame Counter]
    ConsecClosed --> ClosedThreshold{Consec frames > Threshold?}
    ClosedThreshold -- Yes --> AlertDrowsy[State: Drowsy | Fatigue +4.0/frame | Alert Level: Critical]
    ClosedThreshold -- No --> CheckYawn{Is MAR > Mouth Threshold?}
    
    CheckClosed -- No --> ResetClosed[Reset Closed Frame Counter]
    ResetClosed --> CheckBlink{Was closed count > 0?}
    CheckBlink -- Yes --> AddBlink[Increment Blink Count]
    CheckBlink -- No --> CheckYawn
    AddBlink --> CheckYawn
    
    CheckYawn -- Yes --> ConsecYawn[Increment Yawn Frame Counter]
    ConsecYawn --> YawnThreshold{Consec frames > Yawn Threshold?}
    YawnThreshold -- Yes --> YawnAlert[Increment Yawn Count | Fatigue +15.0 once]
    YawnThreshold -- No --> CheckDistract{Is Pitch/Yaw out of bounds?}
    YawnAlert --> CheckDistract
    
    CheckYawn -- No --> ResetYawn[Reset Yawn Frame Counter]
    ResetYawn --> CheckDistract
    
    CheckDistract -- Yes --> ConsecDistract[Increment Distracted Frame Counter]
    ConsecDistract --> DistractThreshold{Consec frames > Distract Limit?}
    DistractThreshold -- Yes --> AlertDistracted[State: Distracted | Fatigue +0.8/frame | Alert Level: Warning]
    DistractThreshold -- No --> CalcScore
    
    CheckDistract -- No --> ResetDistract[Reset Distracted Frame Counter]
    ResetDistract --> AlertRecover[Fatigue -0.2/frame | Alert Level: Normal]
    AlertRecover --> CalcScore
    
    AlertDrowsy --> CalcScore
    AlertDistracted --> CalcScore
    
    CalcScore[Update Final Fatigue Score 0-100%] --> CheckAlarm{Fatigue > 70%?}
    CheckAlarm -- Yes --> SoundAlarm[Trigger Threaded Alarm sound]
    CheckAlarm -- No --> StopAlarm[Stop Alarm Sound]
    
    SoundAlarm --> End([End Frame Process])
    StopAlarm --> End
```

---

## 5. System Entity-Relationship (ER) Diagram
Defines the state data structures managed during a telemetry monitoring session.

```mermaid
erDiagram
    SESSION ||--o{ TELEMETRY : records
    SESSION ||--o{ ALERTS : triggers
    SESSION ||--|| SETTINGS : configures

    SESSION {
        string session_id PK
        timestamp start_time
        string driver_id
        float average_fps
    }

    TELEMETRY {
        timestamp recorded_at PK
        float ear_val
        float mar_val
        float pitch_deg
        float yaw_deg
        float fatigue_pct
        int cumulative_blinks
        int cumulative_yawns
        string status_label
    }

    ALERTS {
        timestamp triggered_at PK
        string alert_type
        float peak_fatigue_pct
        boolean was_muted
    }

    SETTINGS {
        float ear_threshold
        float mar_threshold
        int closed_frames_limit
        int yawn_frames_limit
        boolean mute_active
    }
```

---

## 6. Use Case Diagram
Maps out the system boundary actor activities.

```mermaid
left to right direction
actor Driver
actor FleetAdmin as Fleet Safety Manager

rectangle "AI Drowsiness Detection System" {
    usecase UC1 as "View Live Video Overlay"
    usecase UC2 as "Monitor Telemetry Gauges (EAR, MAR)"
    usecase UC3 as "Configure Detection Thresholds"
    usecase UC4 as "Trigger Auto Buzzer Alert"
    usecase UC5 as "Review Historical Charts"
    usecase UC6 as "Audit Safety Alert Timeline Logs"
}

Driver --> UC1
Driver --> UC2
Driver --> UC4
Driver --> UC5

UC3 --> FleetAdmin
UC6 --> FleetAdmin
UC5 --> FleetAdmin
```

---

## 7. Deployment Topography Diagram
Shows local and cloud execution hosting structures.

```mermaid
graph TD
    subgraph Local [Edge/Desktop Execution]
        HDWebcam[Hardware Webcam] --> |OpenCV cv2.VideoCapture| EdgeApp[PyFlask Application]
        EdgeApp --> |Render local loop| Monitor[Driver Dashboard Monitor]
        EdgeApp --> |Local Audio Hardware| Speaker[Buzzer Beep Alarm]
    end

    subgraph ContainerDeployment [Containerized Cloud Deployments]
        direction TB
        Browser[Client Web Browser] -->|HTML5 WebRTC Webcam Capture| RESTApi[Gunicorn WSGI Server]
        RESTApi -->|Dockerized Virtual App| FlaskBack[Flask Application Engine]
        FlaskBack -->|CPU/GPU Inference| TFKeras[TensorFlow Keras CNN Model]
        RESTApi -- JSON Metrics Payload --> Browser
        Browser -->|Web Audio API Synth Beeps| ClientSpeaker[Driver Browser Speaker]
    end

    %% External database/log forward
    FlaskBack --> |Host Mounted Volume| LogVol[logs/app.log file]
```

---

## 8. Workspace Folder Tree
The physical layout of files created for this project.

```
Driver-Drowsiness-Detection/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── data/
│   ├── train/
│   │   ├── closed_eyes/
│   │   └── open_eyes/
│   ├── val/
│   │   ├── closed_eyes/
│   │   └── open_eyes/
│   └── test/
│       ├── closed_eyes/
│       └── open_eyes/
├── docs/
│   └── architecture.md
├── logs/
│   └── app.log
├── models/
│   └── eye_classifier.h5
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── audio/
│       └── alert.mp3
├── templates/
│   └── index.html
├── tests/
│   └── test_app.py
├── training/
│   ├── dataset_prep.py
│   ├── evaluate.py
│   └── train.py
├── .gitignore
├── app.py
├── alarm.py
├── camera.py
├── config.py
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── Procfile
├── requirements.txt
├── runtime.txt
├── utils.py
└── README.md
```
