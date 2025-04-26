import streamlit as st
from fastai.vision.all import *
import pathlib
import PIL
from PIL import Image
import platform

# Set title and description
st.title("Age and Gender Classifier")
st.write("Upload a face image to classify the person's age group (young/middle/old) and gender")

# Temporary fix for Linux systems
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Load the model
@st.cache_resource
def load_model():
    try:
        return load_learner('age_gender_model.pkl')
    except Exception as e:
        st.error(f"Model file not found. Please ensure 'age_gender_model.pkl' is in the same directory. Error: {str(e)}")
        return None

learn = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and learn is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    try:
        # Convert to fastai image and predict
        img = PILImage.create(uploaded_file)
        pred, pred_idx, probs = learn.predict(img)
        
        # Display prediction
        gender, age = pred.split('_')
        gender = "Male" if gender == "male" else "Female"
        
        st.subheader("Prediction Results:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Gender", value=gender)
            
        with col2:
            st.metric(label="Age Group", value=age.capitalize())
        
        # Show confidence
        st.write(f"Confidence: {probs[pred_idx]*100:.1f}%")
        
        # Show probabilities for all classes
        st.subheader("Probabilities for each class:")
        for idx, prob in enumerate(probs):
            st.write(f"{learn.dls.vocab[idx]}: {prob*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add some instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a clear frontal face image
2. The model will predict gender and age group
3. Results will show with confidence percentages

Note: Works best with cropped face images.
""")

# Add model info
st.sidebar.header("Model Information")
st.sidebar.write("""
- Model: ResNet34
- Classes: 6 (male/female × young/middle/old)
- Training data: Balanced UTKFace subset (1000 images per class)
""")
