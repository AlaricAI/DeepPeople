import streamlit as st
from fastai.vision.all import *
import pathlib
import PIL
import platform
import requests
import os

# Sarlavha
st.title("Yosh va Jinsni Aniqlash Modeli")
st.write("Rasm yuklang, model shaxsning yoshi (yosh/o'rta/qari) va jinsini aniqlaydi")

# Modelni yuklab olish funksiyasi
@st.cache_resource
def load_model():
    try:
        # Agar model fayli yo'q bo'lsa, internetdan yuklab olamiz
        model_url = "https://drive.google.com/file/d/1q5VXxhwe8QwhQfOA-qLXUWmpkar_ZKjt/view?usp=sharing"
        model_path = "age_gender_model.pkl"
        
        if not os.path.exists(model_path):
            st.warning("Model yuklanmoqda...")
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        
        return load_learner(model_path)
    except Exception as e:
        st.error(f"Model yuklanmadi. Xato: {str(e)}")
        return None

# Linux tizimlari uchun
if platform.system() == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

# Modelni yuklash
learn = load_model()

# Rasm yuklash qismi
uploaded_file = st.file_uploader("Rasm tanlang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and learn is not None:
    try:
        # Rasmni ko'rsatish
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption='Yuklangan rasm', use_column_width=True)
        
        # Bashorat qilish
        img_fastai = PILImage.create(uploaded_file)
        pred, pred_idx, probs = learn.predict(img_fastai)
        
        # Natijalarni o'zbek tilida ko'rsatish
        jins, yosh = pred.split('_')
        jins_uz = "Erkak" if jins == "male" else "Ayol"
        yosh_uz = {
            'young': 'Yosh (18 dan kichik)',
            'middle': "O'rta yosh (18-45)",
            'old': 'Qari (45 dan katta)'
        }.get(yosh, yosh)
        
        st.subheader("Natijalar:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Jins", value=jins_uz)
            
        with col2:
            st.metric(label="Yosh guruhi", value=yosh_uz)
        
        st.write(f"Ishonchlilik darajasi: {probs[pred_idx]*100:.1f}%")
        
        st.subheader("Barcha guruhlar bo'yicha ehtimollar:")
        for idx, prob in enumerate(probs):
            label_uz = learn.dls.vocab[idx].replace('male', 'Erkak').replace('female', 'Ayol')
            st.write(f"{label_uz}: {prob*100:.1f}%")
            
    except Exception as e:
        st.error(f"Rasmni tahlil qilishda xato: {str(e)}")

# Qo'shimcha ma'lumotlar
st.sidebar.header("Ko'rsatmalar")
st.sidebar.write("""
1. Aniq yuz ko'rinadigan rasm yuklang
2. Model jins va yosh guruhini aniqlaydi
3. Natijalar ishonchlilik foizi bilan ko'rsatiladi

Eslatma: Yaxshiroq natija uchun faqat yuz bo'lgan rasmlardan foydalaning.
""")
