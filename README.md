# Realtime Cell Agglutination Analyzer

A system-oriented demo for real-time blood cell agglutination analysis using simulated microscope image streaming.

This project focuses on building an end-to-end real-time inference pipeline rather than maximizing model accuracy.

Static microscope images are used to simulate live streaming input, followed by frame-based inference, visualization overlay, latency monitoring, and queue-based processing.

The goal is to demonstrate practical system design for microscope image analysis applications.

---

## Project Goal

Build a demo system that simulates real-time microscope image input and performs blood cell agglutination analysis with:

- fake streaming from static images
- frame-by-frame inference
- real-time prediction overlay
- producer-consumer pipeline design
- queue and multi-thread processing
- latency monitoring
- frame dropping strategy
- demo recording for portfolio presentation

This project is designed for software engineering and AI system demonstration in job interviews.

---

## MVP Direction

### Classification-first

The first version uses image-level classification.

**Input:**  
Microscope image frame

**Output:**  
Normal / Agglutination prediction

**Overlay Display:**

- prediction result
- confidence score
- inference latency
- FPS
- queue size
- dropped frame count

### Detection as Optional Upgrade

If bounding box annotations are verified to be usable, the system can be upgraded to object detection for more detailed localization.

---

## Project Priority

Priority order:

- System demo > Model accuracy
- Working pipeline > Complex UI
- Classification-first > Detection-first

The main objective is to build a complete and explainable end-to-end system.

---

## Planned Pipeline

Image Source  
→ Fake Stream  
→ Frame Queue  
→ Inference Worker  
→ Post-processing  
→ Visualization Overlay  
→ Display / Demo Recording

---

## Repository Structure

```text
realtime-cell-agglutination-analyzer/

├── data/
│   ├── raw/
│   ├── samples/
│   └── processed/
│
├── src/
│   ├── streaming/
│   ├── inference/
│   ├── visualization/
│   ├── pipeline/
│   └── utils/
│
├── scripts/
│   ├── check_dataset.py
│   ├── train_baseline.py
│   └── run_demo.py
│
├── config/
│   └── default.yaml
│
├── notebooks/
│
├── README.md
├── requirements.txt
└── main.py
```

## Data

This repository does not include raw image data.

For local development, place image files under:

```text
data/raw/
  normal/
  agglutination/
```

The expected task is binary image classification:

- normal
- agglutination

The dataset checking script can be used to validate local image files before training or running the demo.

## Environment Setup

This project is developed with Anaconda on Windows and targets GPU inference/training with an NVIDIA RTX 4070 SUPER.

### 1. Create Conda Environment

```bash
conda create -n rca python=3.10 -y
conda activate rca
```

### 2. Upgrade Pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 5. Verify GPU Avalibility

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If torch.cuda.is_available() returns False, check that the NVIDIA driver is installed correctly.

### 6. Run Dataset Check

```bash
python scripts/check_dataset.py --image-dir data/raw --sample-dir data/samples
```

## Notes

- [Dataset Checking](docs/dataset_checking_20260429.md)