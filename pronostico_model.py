import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from datetime import datetime, timedelta

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

def calcular_fechas_futuras_mantenimiento(tipo_mantenimiento, dias_desde_mantenimiento, recorrido, horas_op):
    """
    Calcula fechas futuras de mantenimiento basadas en el tipo y parámetros actuales
    """
    try:
        fecha_actual = datetime.now()
        
        # Calcular fechas futuras según el tipo de mantenimiento
        if "preventivo" in str(tipo_mantenimiento).lower():
            # Mantenimiento preventivo: cada 30-90 días o 5000-10000 km
            if dias_desde_mantenimiento > 60:
                # Si ya pasaron muchos días, sugerir pronto
                dias_hasta_mantenimiento = 7
            elif dias_desde_mantenimiento > 30:
                dias_hasta_mantenimiento = 15
            else:
                dias_hasta_mantenimiento = 30
            
            # Ajustar por recorrido
            if recorrido > 8000:
                dias_hasta_mantenimiento = min(dias_hasta_mantenimiento, 7)
            elif recorrido > 5000:
                dias_hasta_mantenimiento = min(dias_hasta_mantenimiento, 15)
                
        elif "correctivo" in str(tipo_mantenimiento).lower():
            # Mantenimiento correctivo: urgente (1-7 días)
            if dias_desde_mantenimiento > 100:
                dias_hasta_mantenimiento = 1  # Muy urgente
            elif dias_desde_mantenimiento > 60:
                dias_hasta_mantenimiento = 3  # Urgente
            else:
                dias_hasta_mantenimiento = 7  # Pronto
        else:
            # Mantenimiento general: 30 días
            dias_hasta_mantenimiento = 30
        
        # Calcular fechas
        fecha_mantenimiento = fecha_actual + timedelta(days=dias_hasta_mantenimiento)
        fecha_recordatorio = fecha_mantenimiento - timedelta(days=3)
        
        # Determinar urgencia con parámetros específicos
        if dias_hasta_mantenimiento <= 1 or dias_desde_mantenimiento > 120 or recorrido > 15000 or horas_op > 3000:
            urgencia = "CRÍTICA"
        elif dias_hasta_mantenimiento <= 3 or dias_desde_mantenimiento > 90 or recorrido > 12000 or horas_op > 2500:
            urgencia = "ALTA"
        elif dias_hasta_mantenimiento <= 7 or dias_desde_mantenimiento > 60 or recorrido > 9000 or horas_op > 2000:
            urgencia = "MODERADA"
        elif dias_hasta_mantenimiento <= 15 or dias_desde_mantenimiento > 30 or recorrido > 6000 or horas_op > 1500:
            urgencia = "NORMAL"
        else:
            urgencia = "MÍNIMA"
        
        return {
            "fecha_mantenimiento": fecha_mantenimiento.strftime('%Y-%m-%d'),
            "fecha_recordatorio": fecha_recordatorio.strftime('%Y-%m-%d'),
            "dias_hasta_mantenimiento": dias_hasta_mantenimiento,
            "urgencia": urgencia
        }
        
    except Exception as e:
        print(f"Error calculando fechas futuras: {e}")
        return {
            "fecha_mantenimiento": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            "fecha_recordatorio": (datetime.now() + timedelta(days=27)).strftime('%Y-%m-%d'),
            "dias_hasta_mantenimiento": 30,
            "urgencia": "NORMAL"
        }

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
        # Determinar riesgo basado en probabilidades Y valores de entrada
        max_prob = float(max(probabilidades))
        
        # Obtener valores de entrada para análisis de riesgo
        dias_desde_mantenimiento = datos.get('dias', 0)
        recorrido = datos.get('recorrido', 0)
        horas_op = datos.get('horas_op', 0)
        
        # Lógica de riesgo mejorada que considera tanto la probabilidad como los valores
        riesgo_probabilidad = "BAJO"
        if max_prob > 0.8:
            riesgo_probabilidad = "ALTO"
        elif max_prob > 0.6:
            riesgo_probabilidad = "MEDIO"
        
        # Análisis de riesgo basado en valores de entrada
        riesgo_valores = "BAJO"
        
        # Criterios para riesgo alto basado en valores
        if (dias_desde_mantenimiento > 90 or 
            recorrido > 10000 or 
            horas_op > 2000 or
            (dias_desde_mantenimiento > 60 and recorrido > 8000) or
            (dias_desde_mantenimiento > 45 and horas_op > 1500)):
            riesgo_valores = "ALTO"
        elif (dias_desde_mantenimiento > 60 or 
              recorrido > 7000 or 
              horas_op > 1200 or
              (dias_desde_mantenimiento > 30 and recorrido > 5000)):
            riesgo_valores = "MEDIO"
        
        # Combinar ambos análisis de riesgo
        if riesgo_probabilidad == "ALTO" or riesgo_valores == "ALTO":
            riesgo = "ALTO"
        elif riesgo_probabilidad == "MEDIO" or riesgo_valores == "MEDIO":
            riesgo = "MEDIO"
        else:
            riesgo = "BAJO"
        
        # Debug: mostrar información del análisis de riesgo
        print(f"🔍 ANÁLISIS DE RIESGO:")
        print(f"   Valores de entrada: días={dias_desde_mantenimiento}, recorrido={recorrido}, horas_op={horas_op}")
        print(f"   Probabilidad máxima: {max_prob:.3f} ({max_prob*100:.1f}%)")
        print(f"   Riesgo por probabilidad: {riesgo_probabilidad}")
        print(f"   Riesgo por valores: {riesgo_valores}")
        print(f"   Riesgo final: {riesgo}")
        # Recomendaciones según tipo de mantenimiento (case-insensitive, flexible)
        prediccion_lower = str(prediccion).strip().lower()
        print(f"Predicción IA: '{prediccion}' (lower: '{prediccion_lower}')")
        if "correctivo" in prediccion_lower:
            recomendaciones = [
                'Diagnóstico preciso: uso de herramientas de diagnóstico o software.',
                'Inspección técnica detallada por un mecánico especializado.',
                'Reemplazo de partes dañadas: motores, correas, rodamientos, etc.',
                'Reparación estructural: soldaduras, enderezado de chasis, refuerzos.',
                'Análisis de causa raíz: documentar para evitar que se repita.',
                'Actualización del historial de la máquina.',
                'Medidas de seguridad post-reparación: pruebas antes de volver a operar.'
            ]
        elif "preventivo" in prediccion_lower:
            recomendaciones = [
                'Revisión periódica del equipo.',
                'Inspección visual de componentes.',
                'Verificación de ruidos anómalos, vibraciones o fugas.',
                'Lubricación regular de partes móviles.',
                'Cambio de filtros y fluidos según cronograma.',
                'Calibraciones y ajustes: sensores, frenos, presión hidráulica.',
                'Monitoreo de horas de uso y recorrido.',
                'Capacitación del operador y revisión diaria básica.',
                'Checklist preventiva y documentación en cada revisión.'
            ]
        else:
            recomendaciones = ['Consultar con el área de mantenimiento.']

        # --- Ajuste: calcular recorrido futuro como el máximo histórico + 10% ---
        try:
            # Intentar cargar el histórico desde el CSV
            csv_path = os.path.join(os.path.dirname(__file__), 'pronostico_maquinaria_1.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                max_recorrido = df['recorrido'].max()
                recorrido_futuro = max_recorrido * 1.10
            else:
                recorrido_futuro = float(datos.get('recorrido', 0)) * 1.10
        except Exception:
            recorrido_futuro = float(datos.get('recorrido', 0)) * 1.10

        # Calcular fechas futuras de mantenimiento
        fechas_calculadas = calcular_fechas_futuras_mantenimiento(
            prediccion, 
            datos.get('dias', 0), 
            datos.get('recorrido', 0), 
            datos.get('horas_op', 0)
        )

        return {
            "resultado": str(prediccion),
            "riesgo": riesgo,
            "probabilidad": round(max_prob * 100, 2),
            "fecha_prediccion": datetime.now().strftime('%Y-%m-%d'),
            "recomendaciones": recomendaciones,
            "recorrido": recorrido_futuro,
            "fecha_mantenimiento": fechas_calculadas["fecha_mantenimiento"],
            "fecha_recordatorio": fechas_calculadas["fecha_recordatorio"],
            "dias_hasta_mantenimiento": fechas_calculadas["dias_hasta_mantenimiento"],
            "urgencia": fechas_calculadas["urgencia"],
            "fecha_sugerida": fechas_calculadas["fecha_mantenimiento"]
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return {
            "resultado": "ERROR",
            "riesgo": "DESCONOCIDO",
            "probabilidad": 0,
            "error": str(e),
            "recomendaciones": ['Error al generar recomendaciones.'],
            "fecha_mantenimiento": None,
            "fecha_recordatorio": None,
            "dias_hasta_mantenimiento": None,
            "urgencia": None,
            "fecha_sugerida": None
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