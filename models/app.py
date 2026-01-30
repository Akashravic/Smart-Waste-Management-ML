# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import os

# # Set page config
# st.set_page_config(page_title="Smart City Waste Classifier", layout="centered")

# # 1. Load the Model (Cached so it only loads once)
# @st.cache_resource
# def load_my_model():
#     model = tf.keras.models.load_model('waste_classifier_finetuned.h5')
#     return model

# model = load_my_model()
# class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# # 2. UI Layout
# st.title("♻️ Smart Waste Management System")
# st.write("Upload an image of waste to identify its material for proper recycling.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Uploaded Image', use_column_width=True)
    
#     # 3. Predict Button
#     if st.button('Classify Waste'):
#         with st.spinner('Analyzing material...'):
#             # Preprocessing
#             img = img.resize((224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array /= 255.0

#             # Prediction
#             predictions = model.predict(img_array)[0]
#             result_index = np.argmax(predictions)
#             result_label = class_names[result_index]
#             confidence = predictions[result_index]

#             # 4. Display Results
#             st.success(f"**Result: {result_label.upper()}** (Confidence: {confidence*100:.2f}%)")
            
#             # Show breakdown using a progress bar or chart
#             st.write("### Material Breakdown:")
#             for i in range(len(class_names)):
#                 st.write(f"{class_names[i].capitalize()}")
#                 st.progress(float(predictions[i]))

# # Add a footer about your Smart City Project
# st.sidebar.info("This module is part of the 'Self-Sustainable Smart Cities' initiative, focusing on automated waste segregation.")\
    
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Page styling
st.set_page_config(page_title="Smart City Waste AI", page_icon="♻️")

# 1. Load Model with Modern Caching
@st.cache_resource
def load_my_model():
    # Note: Use the fine-tuned model for best results
    return tf.keras.models.load_model('waste_classifier_finetuned.h5')

model = load_my_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 2. UI Header
st.title("♻️ Smart Waste Classifier")
st.markdown("### Goal: Automated Urban Waste Segregation")
st.write("Upload a photo of an item to determine its material and recycling category.")

# 3. Sidebar Information
st.sidebar.header("Project Specs")
st.sidebar.write("**Model:** InceptionV3")
st.sidebar.write("**Accuracy:** ~70% (Fine-tuned)")
st.sidebar.write("**Environment:** Python 3.10")

# 4. File Uploader
uploaded_file = st.file_uploader("Upload trash image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Selected Image", use_container_width=True)
    
    if st.button("Identify Material"):
        with st.spinner("Processing..."):
            # Image Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run Prediction
            preds = model.predict(img_array)[0]
            idx = np.argmax(preds)
            label = class_names[idx]
            conf = preds[idx] * 100
            
            # Display Main Result
            st.subheader(f"Prediction: {label.upper()}")
            st.metric(label="Confidence Score", value=f"{conf:.2f}%")
            
            # Smart City Insight
            if label in ['cardboard', 'paper', 'glass', 'metal', 'plastic']:
                st.success("✅ This item is RECYCLABLE.")
            else:
                st.warning("⚠️ This item belongs in GENERAL TRASH.")

            # 5. Probability Visualization
            st.write("---")
            st.write("### Detection Probability")
            cols = st.columns(len(class_names))
            for i, col in enumerate(cols):
                col.write(f"**{class_names[i].capitalize()}**")
                col.write(f"{preds[i]*100:.1f}%")