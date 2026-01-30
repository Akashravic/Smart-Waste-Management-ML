# ♻️ AI-Powered Urban Waste Classifier
**An Automated Segregation System for Self-Sustainable Smart Cities**

---

## 📌 Project Overview
Proper waste segregation is a cornerstone of sustainable urban development. This project utilizes **Deep Learning** and **Computer Vision** to automate the identification of household waste, categorizing materials to reduce recycling contamination.

The system was developed using **Transfer Learning** on the **InceptionV3** architecture, achieving significant improvements in generalization compared to baseline lightweight models.



---

## 🚀 Key Features
* **Multi-Scale Detection:** Uses Inception modules to identify waste items of various shapes and sizes.
* **Fine-Tuned Accuracy:** Optimized through partial unfreezing of deep layers to specialize in waste textures.
* **Interactive UI:** A real-time web dashboard for image uploading and classification.
* **Smart City Logic:** Integrated decision-making to label items as "Recyclable" or "General Trash."

---

## 📊 Technical Benchmarks
I performed architecture benchmarking to determine the best model for this use case.

| Metric | MobileNetV2 (Baseline) | InceptionV3 (Final) |
| :--- | :--- | :--- |
| **Training Accuracy** | 80.2% | 75.9% |
| **Validation Accuracy** | 47.9% | **70.0%** |
| **Key Advantage** | Resource Efficient | **Superior Generalization** |



---

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Deep Learning:** TensorFlow 2.15+, Keras (Legacy Support via `tf-keras`)
* **Architecture:** InceptionV3
* **Frontend:** Streamlit
* **Deployment:** Virtual Environments (venv)


