# Inter-Camera Person Tracking & Re-Identification

## 📌 Overview

This project implements an **intelligent multi-camera person tracking and re-identification system** based on Computer Vision and Deep Learning.

The system detects and tracks people across multiple cameras while maintaining a **consistent global identity** when the same person moves from one camera to another.

The pipeline combines **YOLOv8 for person detection, BoT-SORT for multi-object tracking, OSNet for person re-identification**, and appearance-based features such as **HSV color histograms** for cross-camera identity association.

> **Note:** This repository contains a portfolio version of the project. Real surveillance videos, sensitive data, company-specific resources, and proprietary model weights are not included.

---

## 🎯 Objectives

The main objectives of the project are:

* Detect people in video streams.
* Track people within each camera.
* Assign persistent local tracking IDs.
* Extract visual features for person re-identification.
* Associate identities across different cameras.
* Maintain a global identity for the same person across camera transitions.
* Handle temporary tracking loss and re-identification.
* Record tracking events and system results.

---

## 🧠 System Architecture

```text
                  ┌──────────────────┐
                  │     Camera 1     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     YOLOv8       │
                  │ Person Detection │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    BoT-SORT      │
                  │ Local Tracking   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Feature Extract. │
                  │      OSNet       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Global ID Manager│
                  └────────┬─────────┘
                           │
                 Cross-Camera Matching
                           │
                           ▼
                  ┌──────────────────┐
                  │     Camera 2     │
                  └──────────────────┘
```

---

## 🔄 Processing Pipeline

### 1. Person Detection

**YOLOv8** is used to detect people in each video stream.

```text
Video Frame
     ↓
YOLOv8
     ↓
Person Bounding Boxes
```

### 2. Local Multi-Object Tracking

**BoT-SORT** associates detections between consecutive frames and assigns local tracking IDs.

```text
Frame t     → Person ID 1
Frame t+1   → Person ID 1
Frame t+2   → Person ID 1
```

This allows the system to maintain a stable identity while a person remains inside the same camera.

### 3. Person Re-Identification

When a person moves between cameras, local tracking IDs are not sufficient.

The system therefore extracts visual representations using **OSNet**, a deep learning model designed for person re-identification.

These features are used to compare a person observed by one camera with candidates detected by another camera.

### 4. Appearance-Based Matching

The cross-camera association combines several visual and spatial cues:

* Re-ID feature similarity
* HSV color histogram similarity
* Bounding-box / spatial information
* IoU-based matching when applicable
* Temporal constraints
* Camera transition zones

The combination of these features helps reduce incorrect identity associations.

### 5. Global Identity Management

A **Global ID Manager** maintains a consistent identity across cameras.

For example:

```text
Camera 1
Person → Local ID 7
          ↓
     Camera transition
          ↓
Camera 2
Person → Local ID 3

Global Identity → Person G12
```

The system can therefore recognize that different local tracking IDs may correspond to the same person.

---

## 🚦 Camera Zones

The system supports predefined zones for camera transitions and overlap areas.

Example:

```text
CAM1_ENTRY_PORTES
CAM1_OVERLAP_LINE
CAM2_ENTRY_SORTIE
CAM2_OVERLAP_LINE
```

These zones provide contextual information for cross-camera association and help constrain possible identity transitions.

---

## 🧩 Main Components

| Component                   | Technology / Method        |
| --------------------------- | -------------------------- |
| Person Detection            | YOLOv8                     |
| Multi-Object Tracking       | BoT-SORT                   |
| Person Re-Identification    | OSNet                      |
| Appearance Features         | HSV Color Histogram        |
| Global Identity Association | Custom Global ID Manager   |
| Camera Zones                | OpenCV / Custom Zone Logic |
| Gender Classification       | EfficientNet               |
| Programming Language        | Python                     |
| Deep Learning               | PyTorch                    |
| Computer Vision             | OpenCV                     |

---

## 📁 Project Structure

```text
inter-camera-person-tracking/
│
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── tracker.py
│   ├── global_tracker.py
│   ├── reid_manager.py
│   ├── gender_detector.py
│   ├── color_detector.py
│   ├── camera_thread.py
│   ├── zones.py
│   ├── logger.py
│   └── utils.py
│
├── configs/
│   └── custom_tracker_config.yaml
│
├── scripts/
│   └── define_zones.py
│
└── assets/
    ├── architecture.png
    └── zones_result.png
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/inter-camera-person-tracking.git

cd inter-camera-person-tracking
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main application:

```bash
python main.py
```

Before running the complete pipeline, configure the required:

* camera/video sources;
* detection parameters;
* tracking thresholds;
* Re-ID configuration;
* transition zones.

---

## 📊 Cross-Camera Identity Association

The identity association process is based on a combination of multiple signals rather than a single similarity score.

Conceptually:

```text
                 Re-ID Similarity
                        │
                        ▼
HSV Appearance ──► Matching Engine ◄── Spatial Information
                        │
                        ▼
                 Temporal Constraints
                        │
                        ▼
                  Global Identity
```

This approach is designed to improve identity continuity when people move between camera views.

---

## 📸 Results

### Camera Zones

![Camera Zones](assets/zones_result.png)

### System Architecture

![System Architecture](assets/architecture.png)

> Demo videos and surveillance data are intentionally not included in this repository.

---

## 🔒 Data & Privacy

This project was developed in the context of intelligent video surveillance.

For privacy and confidentiality reasons, this repository does **not** contain:

* real surveillance videos;
* personal images or identifiable individuals;
* hospital/company data;
* detection logs containing sensitive information;
* proprietary datasets;
* confidential configuration files;
* proprietary model weights.

Only non-sensitive source code, documentation, configuration examples and illustrative assets are provided.

---

## 🛠️ Technologies

```text
Python
PyTorch
OpenCV
YOLOv8
BoT-SORT
OSNet
Deep Learning
Computer Vision
Multi-Object Tracking
Person Re-Identification
Feature Matching
```

---

## 🚀 Possible Improvements

Future improvements could include:

* Camera topology modeling
* Improved cross-camera Re-ID
* Automatic camera transition learning
* Real-time performance optimization
* GPU inference optimization
* Tracking quality evaluation using standard MOT metrics
* Re-ID evaluation using Rank-1 / mAP
* Deployment through an API or web dashboard
* Integration with an MLOps monitoring pipeline

---

## 👩‍💻 Author

**Kaoutar Adlani**

Ingénieure d'État en Intelligence Artificielle

**Focus:** Computer Vision · Deep Learning · Machine Learning · AI

---

## 📄 License

This project is intended for educational and portfolio purposes.

Please verify the licensing terms of third-party models and libraries before redistributing their weights or derivative components.
