## Overview
SmartVision is a fully offline, edge-computing application that provides real-time obstacle detection and bilingual (English & Tamil) audio guidance for visually impaired users. Built with YOLOv8n for vision and local TTS, it requires zero internet connectivity.

**Problem it solves:** Current accessibility tools rely on cloud APIs, network latency, and internet connectivity. SmartVision works 100% offline on any machine.


# Bilingual Smart Vision Assist System 

A fully offline, edge-computing prototype application designed to assist visually impaired users. It runs on a local machine using a webcam feed (or a built-in virtual camera stream simulator if no webcam is available) and processes frames in real time through YOLOv8n to detect indoor obstacles, estimate distance, divide the spatial field into zones, and provide bilingual (English & Tamil) text-to-speech descriptions.

---

## Key Features

1. **100% Offline Edge Computing**: Leverages a local lightweight YOLOv8n model and system-level TTS engines (`pyttsx3`) without relying on any external internet or cloud-based APIs.
2. **Horizontal Spatial Zoning**: Automatically segments the 640x480 video frame into 3 vertical zones: **Left / இடது**, **Center / மையம்**, and **Right / வலது** using bounding box midpoints.
3. **Proximity & Distance Proximity Hack**: Calculates the ratio of the obstacle's bounding box height to the total frame height. Box height $\ge$ 45% is classified as **Close (அருகில்)**, otherwise it is **Far (தொலைவில்)**.
4. **Bilingual Translation Logic**:
   - **English (Format: `[Object], [Position], [Distance]`)**: e.g., *"Chair, Left, Close"*
   - **Tamil (Format: `[Position] [Distance] [Object]`)**: e.g., *"இடது அருகில் நாற்காலி"* (Phonetic fallback: *"Idathu Arugil Naarkali"*)
   - *Fallback Mechanism*: If a YOLO object is not translated in the dictionary, it speaks the English name of the object while preserving Tamil spatial/distance cues.
5. **Multi-Threaded Throttled Audio**: 
   - Uses a background queue and thread to run `pyttsx3`, preventing camera stream stuttering or freezing.
   - Throttles output to speak only once every **3 to 4 seconds** (customizable) for the highest-confidence object currently in view.
6. **Virtual Hardware Simulator Mode**: If a physical camera is not connected, the application automatically launches a simulated wireframe object moving across the screen, allowing full system testing headlessly or locally without a camera.
7. **Interactive Controls**:
   - **Spacebar**: Toggle Voice Assistance (Muted / Running) with instantaneous audio cues.
   - **L Key**: Swap Active Language (English <-> Tamil) with instantaneous bilingual confirmation cues.
   - **ESC Key**: Clean shutdown and release of system resources.

---

## Project Structure

```
SmartVision/
│
├── app.py                 # Main entrypoint, handles video loop, input keys, and HUD rendering
├── audio_engine.py        # Background thread-safe audio worker (COM & pyttsx3 queue management)
├── vision_processor.py    # YOLOv8 inference wrapper, zoning logic, and translation dictionaries
├── requirements.txt       # Python package dependencies
├── verify_system.py       # Automated unit test suite to verify model & TTS threads offline
└── README.md              # Documentation
```

---

##  Prerequisites
- Python 3.8 or higher
- Webcam (or will use virtual simulator mode)
- 500MB disk space (for YOLOv8 model download)
- Windows/Mac/Linux with audio drivers enabled

##  Setup & Installation

1. **Clone or copy** this directory to your local machine.
2. Open a terminal inside the `SmartVision` directory and run:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the automated verification test suite to pre-download the YOLOv8 model and verify that your system audio drivers work:
   ```bash
   python verify_system.py
   ```

---

## How to Run

Launch the main application:
```bash
python app.py
```

### Keyboard Simulation Controls:
- **`[SPACEBAR]`**: Toggles audio feedback ON/OFF (Muted / Running).
- **`[L]`**: Swaps system language between English and Tamil.
- **`[ESC]`**: Cleanly terminates the program.

---

##  Heads-Up Display (HUD) Interface
- **Left Panel**: Displays current **System Status** (`RUNNING`/`MUTED`), **Active Language** (`ENGLISH`/`TAMIL`), and keyboard shortcut reminders.
- **Right Panel**: Displays the **last spoken description target** and a dynamic, graphical progress bar representing the **TTS Cooldown Timer** (cooldown is active when orange, ready to speak when green).
- **Grid separators**: Shows boundaries for the Left, Center, and Right zones.
- **Bounding Boxes**: Color-coded boxes around detected items (**Green for Close**, **Blue for Far**) with corner brackets and details.
##  Troubleshooting

**Q: YOLOv8 model download fails**
A: Run `python verify_system.py` first — it pre-downloads the model. If it still fails, check your internet and disk space.

**Q: TTS not working / no sound**
A: Make sure your system audio drivers are enabled. On Windows, check Volume mixer. On Mac/Linux, check system audio settings.

**Q: Webcam not detected**
A: The app automatically launches Virtual Simulator Mode. If you have a camera connected, make sure no other app is using it.

**Q: Performance is slow**
A: YOLOv8n is lightweight, but older machines may see frame drops. Lower your webcam resolution or increase the throttle delay in `audio_engine.py`.