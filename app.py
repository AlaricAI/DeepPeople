import streamlit as st
from fastai.vision.all import *
import pathlib
import PIL
import platform
import plotly.express as px
import pandas as pd

# Sarlavha
st.title("Yosh va Jinsni Aniqlash Modeli")
st.write("Rasm yuklang, model shaxsning yoshi (yosh/o'rta/qari) va jinsini aniqlaydi")

# Linux tizimlari uchun
if platform.system() == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

# Modelni yuklash
@st.cache_resource
def load_model():
    try:
        return load_learner('age_gender_model.pkl')
    except Exception as e:
        st.error(f"Model yuklanmadi. Xato: {str(e)}")
        return None

learn = load_model()

# Rasm yuklash qismi
uploaded_file = st.file_uploader("Rasm tanlang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and learn is not None:
    try:
        # Rasmni ko'rsatish
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption='Yuklangan rasm', width=300)
        
        # Bashorat qilish
        img_fastai = PILImage.create(uploaded_file)
        pred, pred_idx, probs = learn.predict(img_fastai)
        
        # Natijalarni o'zbek tilida tayyorlash
        jins, yosh = pred.split('_')
        jins_uz = "Erkak" if jins == "male" else "Ayol"
        yosh_uz = {
            'young': 'Yosh (18 dan kichik)',
            'middle': "O'rta yosh (18-45)",
            'old': 'Qari (45 dan katta)'
        }.get(yosh, yosh)
        
        # Natijalarni ko'rsatish
        st.subheader("Asosiy natija:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Jins", value=jins_uz)
            
        with col2:
            st.metric(label="Yosh guruhi", value=yosh_uz)
        
        st.write(f"Ishonchlilik darajasi: {probs[pred_idx]*100:.1f}%")
        
        # Plotly vizualizatsiyasi
        st.subheader("Barcha guruhlar bo'yicha ehtimollar:")
        
        # Ma'lumotlarni tayyorlash
        categories = []
        probabilities = []
        
        for idx, (category, prob) in enumerate(zip(learn.dls.vocab, probs)):
            # Kategoriyalarni o'zbek tiliga tarjima qilish
            cat_uz = category.replace('male', 'Erkak').replace('female', 'Ayol')
            cat_uz = cat_uz.replace('young', 'yosh').replace('middle', "o'rta").replace('old', 'qari')
            categories.append(cat_uz)
            probabilities.append(prob.item() * 100)  # foizga o'tkazish
        
        # DataFrame yaratish
        df = pd.DataFrame({
            'Kategoriya': categories,
            'Ehtimollik (%)': probabilities
        })
        
        # Saralash
        df = df.sort_values('Ehtimollik (%)', ascending=False)
        
        # Plotly bar chart
        fig = px.bar(df, 
                     x='Kategoriya', 
                     y='Ehtimollik (%)',
                     color='Ehtimollik (%)',
                     color_continuous_scale='Bluered',
                     text='Ehtimollik (%)',
                     title='Barcha kategoriyalar bo\'yicha ehtimollar')
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_yaxes(range=[0, 100])  # 0-100% oralig'ini ko'rsatish
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        fig_pie = px.pie(df,
                         names='Kategoriya',
                         values='Ehtimollik (%)',
                         title='Ehtimollarning taqsimlanishi')
        
        st.plotly_chart(fig_pie, use_container_width=True)
            
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
