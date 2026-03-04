# VisionGuard: AI-Powered Semantic Surveillance

VisionGuard is a sophisticated Smart CCTV prototype designed to overcome the limitations of traditional motion-based security systems. By implementing a Multimodal AI architecture, the system performs real-time semantic analysis of video streams, allowing it to identify specific human behaviors, safety emergencies, and security threats with high precision.

## Project Overview

Traditional surveillance systems rely on pixel-level motion detection, which frequently triggers false alarms due to environmental factors like shadows, animals, or weather. VisionGuard solves this by using a "Semantic Brain" based on OpenAI's CLIP (Contrastive Language-Image Pretraining). 

The core innovation lies in the system's ability to "read" a scene. Instead of looking for moving pixels, it measures the mathematical similarity between the live video feed and natural language descriptions of dangerous or noteworthy situations. This allows for a proactive security model that understands context, such as the difference between someone sitting down and someone collapsing in a health crisis.

---

## Technical Architecture

The VisionGuard pipeline is built for efficiency and modularity:

1. **Video Processing Layer**: 
   - Utilizes OpenCV for low-latency frame ingestion from local hardware or remote IP cameras.
   - Implements a configurable frame sampling logic (defaulting to 1.5s intervals) to optimize GPU/CPU utilization while maintaining security coverage.

2. **Multimodal AI Engine**:
   - **Model**: Uses a Vision Transformer (ViT-B/32) CLIP model.
   - **Feature Extraction**: Converts both video frames (images) and scenario descriptions (text) into 512-dimensional vector representations known as "embeddings."
   - **Zero-Shot Classifier**: Operates as a zero-shot learner, meaning it can detect complex scenarios (like "climbing a fence") without ever being explicitly trained on a specific dataset of people climbing fences.

3. **Inference & Matching Logic**:
   - Computes the Cosine Similarity between the live image embedding and a bank of pre-computed text embeddings.
   - Implements robust L2 normalization to ensure similarity scores remain accurate across varying lighting conditions and camera angles.
   - Triggers alerts based on a per-scenario thresholding system defined in the configuration.

4. **Integration Layer**:
   - A modern Streamlit-based dashboard providing real-time visual feedback, alert panels, and persistent event logging for audit trails.

---

## Detection Capabilities

The system is pre-loaded with over 15 high-impact security and safety scenarios:

- **Emergency Response**: Sudden collapses, slip-and-falls, or individuals lying motionless.
- **Intrusion Detection**: Fence/wall climbing, forced entry attempts, or individuals wearing masks in sensitive areas.
- **Weapon Detection**: Identification of firearms or large bladed weapons in threatening postures.
- **Public Safety**: Fire and smoke identification, crowd gathering, or unattended baggage in public corridors.
- **Behavioral Analysis**: Suspicious pacing, physical altercations, or vandalism in progress.

---

## Development Stack

- **Frameworks**: PyTorch, HuggingFace Transformers
- **AI Models**: OpenAI CLIP (ViT-B/32)
- **Tooling**: OpenCV, Pillow, NumPy, Scikit-Learn
- **Dashboard**: Streamlit

---

## Setup and Installation

### Dependencies
Ensure you have Python 3.8+ installed. Install the required AI and vision libraries:

```bash
pip install -r requirements.txt
```

### Execution
To start the core detection engine via command line:
```bash
python detection_pipeline.py
```

To launch the monitoring dashboard:
```bash
streamlit run dashboard.py
```

---

## Application and Future Scope
VisionGuard is designed for high-stakes environments such as hospitals (fall detection), airports (unattended baggage), and industrial sites (safety compliance). Future iterations could include temporal analysis (LSTM/Transformers) to understand actions over time, such as distinguishing between a friendly hug and a struggle.
