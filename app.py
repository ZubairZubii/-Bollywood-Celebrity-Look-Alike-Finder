import os
import pickle
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from mtcnn import MTCNN

# Set page configurations
st.set_page_config(page_title="Celebrity Look-Alike", page_icon="🎬")
st.title("🌟 Discover Your Bollywood Celebrity Look-Alike 🌟")
st.write("Upload a clear image of yourself to find out which Bollywood celebrity you resemble!")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.info("1. Upload a clear photo with only one face.\n"
                "2. Wait a moment for processing.\n"
                "3. See which celebrity you look like!\n")

# Load models and data
feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

# Image Upload
uploaded_file = st.file_uploader("📷 Upload an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image with modern style
    st.markdown("### Your Uploaded Image vs Celebrity Match")
    
    # Use columns to align images side by side
    col1, col2 = st.columns(2, gap="small")

    # Display uploaded image in left column
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=200, use_column_width=True)
    
    # Convert uploaded image to array
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Detect face
    results = detector.detect_faces(image_array)
    if results:
        x, y, width, height = results[0]['box']
        face = image_array[y:y + height, x:x + width]

        # Processing message and progress bar
        with st.spinner('Finding your celebrity match...'):
            # Extract and preprocess face features
            face_image = Image.fromarray(face).resize((224, 224))
            face_array = np.asarray(face_image).astype('float32')
            expanded_img = np.expand_dims(face_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img)
            result = model.predict(preprocessed_img).flatten()

            # Compute similarity with celebrity images
            similarity = [cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0]
                          for i in range(len(feature_list))]
            index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]

        # Display matched celebrity image in right column
        with col2:
            st.image(filenames[index_pos], caption="Celebrity Match!", width=200, use_column_width=True)
        st.success("Match Found!")
    else:
        st.error("No face detected in the uploaded image. Please try again with a clearer image.")
else:
    st.info("Please upload an image to start the matching process.")
