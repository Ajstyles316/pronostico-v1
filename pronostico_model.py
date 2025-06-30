import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from datetime import datetime

# Variables globales para el modelo y scaler
model = None
scaler = None
label_encoder = None
model_loaded = False

def load_or_train_model():
    """
    Carga el modelo pre-entrenado si existe, o entrena uno nuevo si no existe
    """
    global model, scaler, label_encoder, model_loaded
    
    if model_loaded:
        return model, scaler, label_encoder
    
    model_path = os.path.join(os.path.dirname(__file__), 'modelo_pronostico.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_pronostico.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
    
    # Intentar cargar modelo existente
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("✅ Modelo cargado exitosamente")
            model_loaded = True
            return model, scaler, label_encoder
        except Exception as e:
            print(f"⚠️ Error al cargar modelo: {e}")
    
    # Si no existe, entrenar nuevo modelo
    print("🔄 Entrenando nuevo modelo...")
    return train_new_model()

def train_new_model():
    """
    Entrena un nuevo modelo y lo guarda
    """
    global model, scaler, label_encoder, model_loaded
    
    try:
        # Cargar datos
        csv_path = os.path.join(os.path.dirname(__file__), 'pronostico_maquinaria_1.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Limpiar datos - eliminar filas con valores NaN en las columnas necesarias
        df_clean = df.dropna(subset=['dias_desde_mantenimiento', 'recorrido', 'horas_op', 'prediccion_tipo'])
        
        # Preparar datos para entrenamiento
        X = df_clean[['dias_desde_mantenimiento', 'recorrido', 'horas_op']]
        y = df_clean['prediccion_tipo']
        
        # Convertir etiquetas categóricas a numéricas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entrenar modelo XGBoost simplificado
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_scaled, y_encoded)
        
        # Guardar modelo, scaler y encoder
        model_path = os.path.join(os.path.dirname(__file__), 'modelo_pronostico.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_pronostico.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("✅ Modelo entrenado y guardado exitosamente")
        print(f"📊 Datos de entrenamiento: {len(df_clean)} registros")
        print(f"🎯 Tipos de mantenimiento: {list(label_encoder.classes_)}")
        mapeo = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        print(f"🔢 Mapeo: {mapeo}")
        model_loaded = True
        return model, scaler, label_encoder
        
    except Exception as e:
        print(f"❌ Error al entrenar modelo: {e}")
        raise

def predecir_mantenimiento(datos):
    """
    Función principal para predecir mantenimiento
    """
    try:
        # Cargar modelo
        model, scaler, label_encoder = load_or_train_model()
        
        # Preparar datos de entrada con nombres de columnas
        X_input = pd.DataFrame([[
            datos.get('dias', 0),
            datos.get('recorrido', 0),
            datos.get('horas_op', 0)
        ]], columns=['dias_desde_mantenimiento', 'recorrido', 'horas_op'])
        
        # Escalar datos
        X_scaled = scaler.transform(X_input)
        
        # Predecir
        prediccion_encoded = model.predict(X_scaled)[0]
        probabilidades = model.predict_proba(X_scaled)[0]
        
        # Convertir predicción de vuelta a etiqueta original
        prediccion = label_encoder.inverse_transform([prediccion_encoded])[0]
        
        # Determinar riesgo basado en probabilidades
        max_prob = float(max(probabilidades))
        if max_prob > 0.8:
            riesgo = "ALTO"
        elif max_prob > 0.6:
            riesgo = "MEDIO"
        else:
            riesgo = "BAJO"
        
        return {
            "resultado": str(prediccion),
            "riesgo": riesgo,
            "probabilidad": round(max_prob * 100, 2),
            "fecha_prediccion": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return {
            "resultado": "ERROR",
            "riesgo": "DESCONOCIDO",
            "probabilidad": 0,
            "error": str(e)
        }

# Función para verificar si el modelo está listo
def verificar_modelo():
    """
    Verifica si el modelo está cargado y listo para usar
    """
    try:
        model, scaler, label_encoder = load_or_train_model()
        return True
    except:
        return False

if __name__ == "__main__":
    # Test de la función
    print("🧪 Probando función de pronóstico...")
    resultado = predecir_mantenimiento({
        'dias': 30,
        'recorrido': 1000,
        'horas_op': 150
    })
    print(f"Resultado: {resultado}") 