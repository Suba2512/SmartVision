m r# SmartVision – Offline Bilingual Vision Assist System

## Overview

**SmartVision** is a fully offline, edge-AI assistive application that provides real-time obstacle detection and bilingual (English & Tamil) voice guidance for visually impaired users. Powered by **YOLOv8n** and an offline **Text-to-Speech (TTS)** engine, SmartVision performs all processing locally without requiring an internet connection, cloud APIs, or external servers.

The system processes live video from a webcam and identifies surrounding obstacles, estimates their relative proximity, determines their spatial location, and announces the information through natural voice feedback in either English or Tamil.

---

## Problem Statement

Many existing visual assistance solutions depend on cloud services, resulting in:

* Internet dependency
* High latency
* Privacy concerns
* Reduced reliability in rural or low-connectivity areas
* Increased operational cost

SmartVision eliminates these limitations by performing all computer vision and speech synthesis locally on the device, ensuring fast, private, and reliable assistance anywhere.

---

## Solution

SmartVision combines lightweight computer vision with offline speech synthesis to create an intelligent navigation assistant capable of:

* Detecting common indoor obstacles
* Estimating relative distance
* Identifying obstacle position
* Providing bilingual audio guidance
* Operating completely offline

---

# Key Features

## 100% Offline Edge Computing

* Runs entirely on the local machine
* No cloud services
* No internet required
* Lightweight YOLOv8n object detector
* Offline pyttsx3 speech engine

---

## Real-Time Object Detection

Detects common indoor objects including:

* Chair
* Person
* Table
* Door
* Bottle
* Backpack
* Laptop
* Cup
* Keyboard
* Cell Phone

and all other objects supported by the YOLOv8n model.

---

## Intelligent Spatial Zoning

The video frame (640 × 480) is divided into three navigation zones.

* Left (இடது)
* Center (மையம்)
* Right (வலது)

The object's midpoint determines its position.

Example:

```
Chair → Left
Bottle → Center
Person → Right
```

---

## Relative Distance Estimation

Distance is estimated using the object's bounding-box height relative to the frame.

Bounding Box Height ≥ 45%

→ Close (அருகில்)

Bounding Box Height < 45%

→ Far (தொலைவில்)

This lightweight heuristic avoids the need for depth sensors while providing practical proximity information.

---

## Bilingual Voice Assistance

### English Format

```
<Object>, <Position>, <Distance>
```

Example:

```
Chair, Left, Close
```

### Tamil Format

```
<Position> <Distance> <Object>
```

Example:

```
இடது அருகில் நாற்காலி
```

If a Tamil translation is unavailable, SmartVision automatically falls back to the English object name while preserving Tamil location and distance cues.

Example:

```
இடது அருகில் Laptop
```

---

## Thread-Safe Audio Engine

A dedicated background thread handles speech generation to prevent interruptions in video processing.

Features include:

* Non-blocking audio queue
* Smooth video rendering
* Configurable speech interval (default: 3–4 seconds)
* Announces only the highest-confidence detected object

---

## Virtual Camera Simulator

If no webcam is detected, SmartVision automatically switches to Simulator Mode.

The simulator:

* Generates moving virtual obstacles
* Enables testing without physical hardware
* Supports demonstrations and development
* Allows automated verification

---

## Interactive Controls

| Key   | Function                          |
| ----- | --------------------------------- |
| Space | Toggle Voice Assistance           |
| L     | Switch Language (English ⇄ Tamil) |
| ESC   | Exit Application                  |

---

# Project Structure

```
SmartVision/
│
├── app.py
│   Main application entry point
│
├── vision_processor.py
│   YOLO inference, zoning, translations, distance estimation
│
├── audio_engine.py
│   Background speech engine using pyttsx3
│
├── verify_system.py
│   Offline verification and diagnostics
│
├── requirements.txt
│   Project dependencies
│
└── README.md
    Documentation
```

---

# System Requirements

* Python 3.8+
* Webcam (optional)
* Windows / Linux / macOS
* 500 MB free disk space
* Audio output device

---

# Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

Verify your system and download the YOLOv8n model:

```bash
python verify_system.py
```

---

# Running the Application

Start SmartVision:

```bash
python app.py
```

Keyboard shortcuts:

```
SPACE  → Voice ON/OFF

L      → Switch Language

ESC    → Exit
```

---

# Heads-Up Display (HUD)

The interface includes:

### Left Panel

* System Status
* Active Language
* Keyboard Controls

### Center View

* Live camera feed
* Detection boxes
* Left–Center–Right zone guides

### Right Panel

* Last spoken message
* Speech cooldown timer
* Voice status

Bounding box colors:

* 🟢 Green → Close Object
* 🔵 Blue → Far Object

---

# Verification

Before first use, execute:

```bash
python verify_system.py
```

The verification process checks:

* YOLO model availability
* Camera detection
* Offline speech engine
* Background audio thread
* Translation dictionaries

---

# Troubleshooting

### YOLO Model Not Downloading

Run:

```bash
python verify_system.py
```

Ensure an internet connection is available for the initial model download and verify sufficient disk space.

---

### No Audio Output

* Check system volume.
* Ensure audio drivers are installed.
* Verify pyttsx3 is functioning correctly.

---

### Webcam Not Detected

SmartVision automatically enters **Virtual Simulator Mode**.

If a webcam is connected, ensure no other application is currently using it.

---

### Low Performance

For older systems:

* Reduce webcam resolution.
* Increase the speech cooldown interval.
* Close unnecessary background applications.

---

# Future Enhancements

* Outdoor navigation support
* Staircase and pothole detection
* Currency recognition
* OCR-based text reading
* Face recognition for known contacts
* GPS integration
* Voice command interface
* Depth estimation using monocular vision
* Mobile deployment on Android devices
* Raspberry Pi implementation with wearable camera

---

# Technologies Used

* Python
* YOLOv8n
* OpenCV
* pyttsx3
* NumPy
* Multithreading
* Edge Computing
* Computer Vision

---

# Impact

SmartVision demonstrates that affordable, privacy-preserving, and intelligent assistive technology can operate entirely offline. By combining lightweight AI with bilingual accessibility, it offers an inclusive solution for visually impaired individuals in environments where internet connectivity is limited or unavailable.
