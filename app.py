import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Set device to MPS for your Mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 1. Model Loading Function
@st.cache_resource
def load_font_model(checkpoint_path):
    # Replicate your ResNet18 architecture
    model = models.resnet18(weights=None)
    num_letters = 26
    model.fc = nn.Linear(model.fc.in_features, num_letters)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# 2. Match your specific Training Transforms
# We skip RandomInvert here to keep predictions stable
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🖋️ Font Style Capture")

try:
    model = load_font_model('my_model_checkpoint.pth')
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.success(f"Brain active on {device}!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload your photo of R B G A S W Q", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- IMAGE SEGMENTATION ---
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu to find letters automatically
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = sorted([c for c in contours if cv2.contourArea(c) > 50], 
                            key=lambda c: cv2.boundingRect(c)[0])

    if valid_contours:
        cols = st.columns(len(valid_contours))
        
        for i, cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # Crop with 10px padding for better ResNet context
            roi = img_array[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
            
            # --- MODEL INFERENCE ---
            roi_pil = Image.fromarray(roi)
            input_tensor = predict_transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                prediction = class_names[predicted_idx.item()]

            with cols[i]:
                st.image(roi, use_container_width=True)
                st.markdown(f"**{prediction}**")
                st.checkbox("Yes", key=f"c_{i}", value=True)

    if st.button("Generate My Font"):
        st.balloons()
        st.write("Starting Phase 2: Style Interpolation...")