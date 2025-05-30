import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# تحميل النموذج المحفوظ باستخدام joblib
model = joblib.load('svm_model.joblib')

# تحميل الـ scaler المحفوظ
scaler = joblib.load('scaler.joblib')

# إنشاء واجهة Streamlit
st.title("'SVM' التنبؤ بالانيميا باستخدام خوارزمية ال")

# مدخلات المستخدم مع قيم افتراضية
gender = st.selectbox("الجنس:", [0, 1], format_func=lambda x: "أنثى" if x == 0 else "ذكر", index=1)  # 0: أنثى, 1: ذكر
hemoglobin = st.number_input("الهيموجلوبين:", min_value=0.0, value=20.0)  # القيمة الافتراضية للهيموجلوبين
mchc = st.number_input("MCHC:", min_value=0.0, value=80.0)  # القيمة الافتراضية لـ MCHC

# عند الضغط على الزر، نقوم بالتنبؤ
if st.button("تنبؤ"):
    # تحويل المدخلات إلى DataFrame
    user_data = pd.DataFrame([[gender, hemoglobin, mchc]], columns=["Gender", "Hemoglobin",  "MCHC"])
    
    # تطبيع البيانات المدخلة
    user_data_scaled = scaler.transform(user_data)  # نستخدم transform بدلاً من fit_transform
    
    # التنبؤ
    prediction = model.predict(user_data_scaled)
    
    # عرض النتيجة
    if prediction == 0:
        st.success("✅ المريض لا يعاني من الأنيميا.")
    else:
        st.error("⚠️ المريض يُحتمل أن يكون مصابًا بالأنيميا.")
