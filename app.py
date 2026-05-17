import streamlit as st
import pandas as pd
import joblib

# 1. Cargar el modelo pre-entrenado
# Usamos Random Forest porque suele tener mejor rendimiento (F1-Score)
try:
    modelo = joblib.load('modelos/modelo_random_forest.pkl')
except Exception as e:
    st.error(f"Error al cargar el modelo. Verifica que la carpeta 'modelos' y el archivo existan. Detalle: {e}")

# 2. Encabezado de la aplicación (Requisito ISIL)
st.title("🚢 Predicción de Supervivencia en el Titanic")
st.markdown("### Desarrollado por: Joel Vargas Cano - Código ISIL: 61250273")
st.markdown("🔗 **[Enlace a mi cuaderno de Google COLAB] https://colab.research.google.com/drive/1IPJHSg7cI4dRbEOxF9DrVkzdQOiyAAFi?usp=sharing**")
st.markdown("---")

st.write("Ingrese los datos del pasajero para evaluar su probabilidad de supervivencia:")

# 3. Interfaz de entrada de datos (Features)
col1, col2 = st.columns(2)

with col1:
    # index=2 hace que por defecto aparezca seleccionada la 3ra clase (la más común)
    pclass = st.selectbox("Clase del Pasajero (Pclass)", [1, 2, 3], index=2, help="1 = Primera, 2 = Segunda, 3 = Tercera.")
    sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
    
    # step=1.0 para que sume de a 1 año entero. value=28.0 (mediana histórica)
    age = st.number_input("Edad", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
    
    # Tarifa máxima histórica fue aprox 512. step=5.0 para saltos más cómodos al usuario
    fare = st.number_input("Tarifa pagada (Fare)", min_value=0.0, max_value=600.0, value=15.0, step=5.0)

with col2:
    # Máximos históricos reales del dataset: SibSp=8, Parch=6
    sibsp = st.number_input("Hermanos/Cónyuges a bordo (SibSp)", min_value=0, max_value=8, value=0, step=1)
    parch = st.number_input("Padres/Hijos a bordo (Parch)", min_value=0, max_value=6, value=0, step=1)
    
    # Southampton fue el puerto donde embarcó la inmensa mayoría, lo ponemos primero
    embarked = st.selectbox("Puerto de Embarque", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])

# 4. Procesamiento de los datos ingresados
# Transformar las entradas para que coincidan exactamente con el preprocesamiento del COLAB
sex_num = 1 if sex == "Femenino" else 0
embarked_q = 1 if embarked == "Q (Queenstown)" else 0
embarked_s = 1 if embarked == "S (Southampton)" else 0

# Crear el DataFrame con la misma estructura de entrenamiento
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_num],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [embarked_q],
    'Embarked_S': [embarked_s]
})

# 5. Botón de Predicción
if st.button("🔮 Predecir Supervivencia"):
    # HOTFIX: Igualar los nombres de las columnas a los que el modelo espera internamente
    input_data.columns = modelo.feature_names_in_
    
    # Realizar la predicción
    prediccion = modelo.predict(input_data)
    
    st.markdown("---")
    if prediccion[0] == 1:
        st.success("✅ **Resultado:** Según el modelo, este pasajero **SOBREVIVIÓ** al naufragio.")
    else:
        st.error("⚠️ **Resultado:** Según el modelo, este pasajero **NO SOBREVIVIÓ** al naufragio.")
