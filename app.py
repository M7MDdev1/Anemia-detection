import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# تحميل النموذج المحفوظ باستخدام joblib
model = joblib.load('svm_model.joblib')

# تحميل الـ scaler المحفوظ
scaler = joblib.load('scaler.joblib')

# إنشاء واجهة Streamlit
st.title("تنبؤ الأنيميا باستخدام نموذج تعلم الآلة")

# مدخلات المستخدم مع قيم افتراضية
gender = st.selectbox("الجنس:", [0, 1], format_func=lambda x: "أنثى" if x == 0 else "ذكر", index=1)  # 0: أنثى, 1: ذكر
hemoglobin = st.number_input("الهيموجلوبين:", min_value=0.0, value=13.0)  # القيمة الافتراضية للهيموجلوبين
mch = st.number_input("MCH:", min_value=0.0, value=25.0)  # القيمة الافتراضية لـ MCH
mchc = st.number_input("MCHC:", min_value=0.0, value=32.0)  # القيمة الافتراضية لـ MCHC
mcv = st.number_input("MCV:", min_value=0.0, value=85.0)  # القيمة الافتراضية لـ MCV

# عند الضغط على الزر، نقوم بالتنبؤ
if st.button("تنبؤ"):
    # تحويل المدخلات إلى DataFrame
    user_data = pd.DataFrame([[gender, hemoglobin, mch, mchc, mcv]], columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"])
    
    # تطبيع البيانات المدخلة
    user_data_scaled = scaler.transform(user_data)  # نستخدم transform بدلاً من fit_transform
    
    # التنبؤ
    prediction = model.predict(user_data_scaled)
    
    # عرض النتيجة
    if prediction == 0:
        st.success("✅ المريض لا يعاني من الأنيميا.")
    else:
        st.error("⚠️ المريض يُحتمل أن يكون مصابًا بالأنيميا.")
