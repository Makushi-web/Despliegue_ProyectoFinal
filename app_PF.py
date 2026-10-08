import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento Estudiantil",
    page_icon="🎓",
    layout="centered"
)

# Cargar los artefactos del modelo de forma segura
@st.cache_resource
def load_artifacts():
    model = joblib.load('optimized_boosting_model.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    le_internet = joblib.load('label_encoder_internet.joblib')
    le_passfail = joblib.load('label_encoder_passfail.joblib')
    return model, scaler, le_internet, le_passfail

try:
    model, scaler, le_internet, le_passfail = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}. Asegúrate de haber subido los archivos .joblib al repositorio.")
    st.stop()

st.title("🎓 Predicción de Rendimiento Escolar")
st.write("Introduce los datos del estudiante para evaluar si aprobará o no la materia.")

# Formulario de entrada
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Edad (Age)", min_value=15, max_value=22, value=17)
        study_hours = st.number_input("Horas de estudio semanales", min_value=0.0, max_value=40.0, value=10.0)
        failures = st.slider("Fallas previas (Failures)", min_value=0, max_value=3, value=0)
        absences = st.number_input("Ausencias (Absences)", min_value=0, max_value=100, value=2)
        internet = st.selectbox("¿Tiene acceso a Internet?", ["yes", "no"])
        free_time = st.slider("Tiempo libre después de clases (1-5)", 1, 5, 3)

    with col2:
        go_out = st.slider("Salidas con amigos (1-5)", 1, 5, 3)
        health = st.slider("Estado de salud actual (1-5)", 1, 5, 5)
        mother_edu = st.selectbox("Educación de la Madre (0-4)", [0, 1, 2, 3, 4], index=2)
        father_edu = st.selectbox("Educación del Padre (0-4)", [0, 1, 2, 3, 4], index=2)
        travel_time = st.slider("Tiempo de viaje a la escuela (1-4)", 1, 4, 1)

    submit_button = st.form_submit_button(label="Predecir Resultado")

if submit_button:
    # Transformar la variable de internet
    internet_encoded = le_internet.transform([internet])[0]

    # Crear el vector de entrada con las 11 características originales
    input_data = pd.DataFrame([{
        'Age': float(age),
        'StudyTime_hours_week': float(study_hours),
        'Failures': float(failures),
        'Absences': float(absences),
        'Internet': float(internet_encoded),
        'FreeTime': float(free_time),
        'GoOut': float(go_out),
        'Health': float(health),
        'MotherEducation': float(mother_edu),
        'FatherEducation': float(father_edu),
        'TravelTime': float(travel_time)
    }])

    # Escalar los datos usando el scaler cargado
    input_scaled = scaler.transform(input_data)

    # Realizar predicción de clase y probabilidad
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Decodificar el resultado para mostrarlo
    result_label = le_passfail.inverse_transform([prediction])[0]

    st.markdown("--- ")
    if prediction == 1:
        st.success(f"🎉 **Resultado: {result_label}** (Probabilidad de aprobar: {probability*100:.2f}%)")
    else:
        st.error(f"❌ **Resultado: {result_label}** (Probabilidad de aprobar: {probability*100:.2f}%)")
