
# 🎬 Bollywood Celebrity Look-Alike Finder

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.4.0-red)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5.0-orange)](https://www.tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-yellow)](https://github.com/ultralytics/yolov8)

## 🌟 Find Your Bollywood Celebrity Twin 🌟

Upload your image and discover which Bollywood celebrity you resemble the most! Using state-of-the-art face detection and feature extraction techniques, this app provides a fun and accurate matching experience.

![Celebrity Look-Alike Demo](demo_image.gif)

---

### 📜 Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [How It Works](#how-it-works)
5. [License](#license)

---

### ✨ Features
- **Bollywood Celebrity Matching**: Matches your image to a Bollywood star from our dataset.
- **State-of-the-Art Models**: Uses ResNet50 for feature extraction and MTCNN for face detection.
- **Streamlit Web Interface**: Simple and interactive interface built with Streamlit.
- **High Accuracy**: Calculates similarity with cosine similarity for precision.

---

### ⚙️ Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/YourUsername/celebrity-lookalike.git
cd celebrity-lookalike
pip install -r requirements.txt
```

**Requirements**:
- Python 3.7+
- Streamlit
- TensorFlow
- OpenCV
- MTCNN
- NumPy
- PIL

---

### 🚀 Usage

Run the app using Streamlit:

```bash
streamlit run app.py
```

#### Instructions
1. Upload a clear photo with only one face.
2. Wait for the app to process your image.
3. See your Bollywood celebrity match appear on the screen!

---

### 🧠 How It Works
The process includes:
1. **Face Detection**: The app uses MTCNN to detect faces in the uploaded image.
2. **Feature Extraction**: ResNet50 (without the top layer) is used to extract features from the face.
3. **Cosine Similarity**: The extracted features are compared with the celebrity dataset using cosine similarity to find the closest match.

#### Model Structure
- **ResNet50**: Used to extract facial features.
- **MTCNN**: Multi-task Cascaded Convolutional Networks for accurate face detection.

![Model Flowchart](model_flowchart.png)

---

### 🎥 Demo Video
[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

---

### 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

### 📬 Contributing
We welcome contributions! Feel free to submit pull requests to help improve this project.
```

### Instructions:
- Replace placeholders like `YOUR_VIDEO_ID`, `https://github.com/YourUsername/celebrity-lookalike.git`, and images like `demo_image.gif` and `model_flowchart.png` with your actual content.
- Add any specific usage notes or troubleshooting if needed.
  
This README format provides a clean, professional look for your project and includes visuals, badges, and organized sections to make it user-friendly and informative!