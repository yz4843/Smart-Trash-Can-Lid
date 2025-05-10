# Smart Trash Can Lid with AI-Powered Recyclable Waste Classification and Voice Interaction

This project implements a local, AI-powered smart trash can assistant using a Raspberry Pi 5. It classifies waste using image recognition and can interact with users via voice when uncertain, combining real-time image classification, speech-to-text transcription, and large language model reasoning — all fully offline.

---

## System Requirements

- **Platform**: Raspberry Pi 5
- **OS**: Raspberry Pi OS Desktop 64-bit (Bookworm)
- **Python**: 3.11+ (recommended to use virtual environment)

---

## Hardware Components

- [Adafruit BrainCraft HAT](https://www.adafruit.com/product/4374)
- [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/)
- [Mono Enclosed Speaker - 1W 8 Ohm (x2)](https://www.adafruit.com/product/5986)
- 1x Servo Motor (any model compatible with 5V PWM)
- Custom 3D-printed or DIY mounts for camera and servo (optional)

> Follow the official BrainCraft HAT setup guide to configure hardware:  
> https://learn.adafruit.com/adafruit-braincraft-hat-easy-machine-learning-for-raspberry-pi

---

## Environment Setup

### 1. Install Ollama (with sudo)

Install Ollama and download the required model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### 2. Create and Activate Python Virtual Environment

```bash
python3 -m venv ~/env
source ~/env/bin/activate
```

### 3. Install Python Dependencies

Install required packages in your virtual environment:

```bash
pip install numpy pygame sounddevice opencv-python scipy whisper
pip install adafruit-blinka adafruit-circuitpython-dotstar adafruit-circuitpython-motor
pip install tflite-runtime requests
```

> If using `whisper` with CPU:  
> `pip install git+https://github.com/openai/whisper.git`

---

## Project Structure

```
project/
├── models/
│   ├── garbage_classificatio.tflite
│   └── labels.txt
├── main.py
├── run.sh
├── bchatsplash.bmp
├── README.md
└── ...
```

---

## How to Run

### 1. Edit `run.sh`

Make sure the following variables are updated to your environment:

```bash
MODEL_NAME="llama3.2:1b"
PROJECT_DIR="$HOME/project"
VENV_ACTIVATE="$HOME/env/bin/activate"
MODEL_PATH="$HOME/project/models/garbage_classificatio.tflite"
LABELS_PATH="$HOME/project/models/labels.txt"
```

### 2. Launch the Smart Trash Can

```bash
chmod +x run.sh
./run.sh
```

The script will:
- Start the Ollama model server in the background.
- Activate your Python environment.
- Run `main.py` with the specified model and label file.

---

## Models Used

- **Image Classification**: MobileNetV2 (TFLite, 160x160 input, 0.75 width)
- **Speech Recognition**: Whisper (tiny.en)
- **Language Model**: LLaMA 3.2 (1B), via Ollama
- **TTS**: pico2wave (via subprocess, no GPU required)

---

## Features

- Real-time image classification
- Voice-based clarification for low-confidence predictions
- Local LLM-powered conversation and classification refinement
- Servo-controlled lid with LED indicator feedback

---

## Future Improvements

- Add support for more waste categories (e.g., compostable, hazardous)
- Enable multilingual interactions
- Explore gamification features for community engagement
- Open-source hardware and 3D files for reproducibility

---

## License

MIT License. See `main.py` for SPDX headers.
