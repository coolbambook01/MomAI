# 🎨 MomAI: Multimodal Literacy Engine & Font Synthesis (Phase 1)

**MomAI** is an interactive educational tool designed to help children bridge the gap between visual letters and spoken language. This repository currently houses **Phase 1** of the project: a high-accuracy neural character classifier.

## 🌟 The Vision
The ultimate goal of this project is to develop a **Custom Font Generator**. While this current iteration identifies letters, the next phase will involve using these classifications to feed a Generative Adversarial Network (GAN) or Variational Autoencoder (VAE) to synthesize entirely new typography based on a user's handwriting style.

---

## 🧠 Technical Implementation (Phase 1: Classifier)

The model was developed and trained in `nn.ipynb` using the following pipeline:

### 1. Dataset
- **Source:** Sourced from **Kaggle** (Handwritten/Printed Letter Dataset).
- **Structure:** Processed via `ImageFolder` with a folder-per-letter class architecture.

### 2. Architecture & Training
- **Core Model:** **ResNet18** (Transfer Learning).
- **Optimization:** Replaced the final Fully Connected (FC) layer to support a 26-way classification (A-Z).
- **Hardware Acceleration:** Leveraged **Apple Silicon (MPS)** for high-speed GPU training.
- **Pre-processing:** - Resized to `224x224`.
  - Grayscale conversion (3-channel mapped).
  - **Augmentation:** Implemented `RandomInvert(p=0.5)` to improve model robustness against different ink/background contrasts.
- **Hyperparameters:** Trained for `7` epochs using `Adam` optimizer ($lr=0.001$) and `CrossEntropyLoss`.

### 3. Evaluation Metrics
The model was evaluated on a 20% held-out test set using:
- **Accuracy:** Overall correctness of predictions.
- **F1-Score (Weighted):** To ensure balanced performance across all 26 classes.

---

## 🚀 Features (The App)
- **Visual Intelligence:** Uses the trained `my_model_checkpoint.pth` to identify letters via webcam or upload.
- **Multimodal Feedback:** Integrated **gTTS (Google Text-to-Speech)** to play audio: *"M is for Monkey!"*
- **Kid-Friendly UX:** Built with **Streamlit** for a simple, "Capture & Reveal" interaction.

---

## 🛠 Setup & Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/coolbambook01/MomAI.git](https://github.com/coolbambook01/MomAI.git)
