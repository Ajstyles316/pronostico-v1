# Sistema de Pronóstico de Mantenimiento

## Descripción General

El sistema de pronóstico utiliza un modelo de Machine Learning (XGBoost) para predecir el tipo de mantenimiento que necesitará una maquinaria basándose en datos históricos y parámetros actuales.

## Parámetros de Entrada

El modelo utiliza **3 parámetros principales** para hacer sus predicciones:

1. **`dias_desde_mantenimiento`**: Días transcurridos desde el último mantenimiento
2. **`recorrido`**: Kilómetros totales recorridos por la maquinaria
3. **`horas_op`**: Horas totales de operación

## Tipos de Predicción

El modelo puede predecir dos tipos de mantenimiento:

- **Preventivo**: Mantenimiento programado para prevenir fallas
- **Correctivo**: Mantenimiento urgente para reparar fallas existentes

## Niveles de Riesgo

El riesgo se determina por la **probabilidad de confianza** del modelo:

### 🟢 RIESGO BAJO
- **Probabilidad**: < 40%
- **Significado**: El modelo tiene poca confianza en su predicción
- **Acción**: Revisar manualmente los parámetros

### 🟡 RIESGO MEDIO  
- **Probabilidad**: 40% - 70%
- **Significado**: El modelo tiene confianza moderada
- **Acción**: Seguir recomendaciones con precaución

### 🔴 RIESGO ALTO
- **Probabilidad**: > 70%
- **Significado**: El modelo tiene alta confianza en su predicción
- **Acción**: Seguir recomendaciones inmediatamente

## Cálculo de Fechas Futuras

### Mantenimiento Preventivo
- **Días desde mantenimiento > 60**: Mantenimiento en 7 días (urgente)
- **Días desde mantenimiento > 30**: Mantenimiento en 15 días
- **Días desde mantenimiento ≤ 30**: Mantenimiento en 30 días
- **Recorrido > 8000 km**: Reducir plazo a 7 días máximo
- **Recorrido > 5000 km**: Reducir plazo a 15 días máximo

### Mantenimiento Correctivo
- **Días desde mantenimiento > 100**: Mantenimiento en 1 día (muy urgente)
- **Días desde mantenimiento > 60**: Mantenimiento en 3 días (urgente)
- **Días desde mantenimiento ≤ 60**: Mantenimiento en 7 días

### Niveles de Urgencia
- **ALTA**: ≤ 7 días hasta mantenimiento
- **MEDIA**: 8-15 días hasta mantenimiento  
- **BAJA**: > 15 días hasta mantenimiento

## Datos Guardados en Base de Datos

Cada pronóstico guarda la siguiente información:

```json
{
  "placa": "EB-123",
  "fecha_asig": "2024-01-15",
  "horas_op": 150,
  "recorrido": 5000,
  "resultado": "Preventivo",
  "riesgo": "MEDIO",
  "probabilidad": 65.5,
  "fecha_prediccion": "2024-01-15T10:30:00",
  "fecha_sugerida": "2024-02-15T00:00:00",
  "fecha_mantenimiento": "2024-02-15T00:00:00",
  "fecha_recordatorio": "2024-02-12T00:00:00",
  "dias_hasta_mantenimiento": 31,
  "urgencia": "BAJA",
  "recomendaciones": ["Revisión periódica del equipo", "..."],
  "fechas_futuras": {
    "fecha_mantenimiento": "2024-02-15T00:00:00",
    "fecha_recordatorio": "2024-02-12T00:00:00", 
    "dias_hasta_mantenimiento": 31,
    "urgencia": "BAJA"
  }
}
```

## Recomendaciones por Tipo

### Mantenimiento Preventivo
- Revisión periódica del equipo
- Inspección visual de componentes
- Verificación de ruidos anómalos, vibraciones o fugas
- Lubricación regular de partes móviles
- Cambio de filtros y fluidos según cronograma
- Calibraciones y ajustes: sensores, frenos, presión hidráulica
- Monitoreo de horas de uso y recorrido
- Capacitación del operador y revisión diaria básica
- Checklist preventiva y documentación en cada revisión

### Mantenimiento Correctivo
- Diagnóstico preciso: uso de herramientas de diagnóstico o software
- Inspección técnica detallada por un mecánico especializado
- Reemplazo de partes dañadas: motores, correas, rodamientos, etc.
- Reparación estructural: soldaduras, enderezado de chasis, refuerzos
- Análisis de causa raíz: documentar para evitar que se repita
- Actualización del historial de la máquina
- Medidas de seguridad post-reparación: pruebas antes de volver a operar

## Archivos del Sistema

- `pronostico_model.py`: Modelo principal de predicción
- `modelo_pronostico.pkl`: Modelo entrenado (XGBoost)
- `scaler_pronostico.pkl`: Escalador de datos
- `label_encoder.pkl`: Codificador de etiquetas
- `pronostico_maquinaria_1.csv`: Datos de entrenamiento
- `pronostico_mantenimiento.ipynb`: Notebook de análisis y entrenamiento

## Uso del API

### Endpoint: POST /api/pronostico/

```json
{
  "placa": "EB-123",
  "fecha_asig": "2024-01-15",
  "horas_op": 150,
  "recorrido": 5000
}
```

### Respuesta:

```json
{
  "resultado": "Preventivo",
  "riesgo": "MEDIO", 
  "probabilidad": 65.5,
  "fecha_prediccion": "2024-01-15T10:30:00",
  "fecha_mantenimiento": "2024-02-15T00:00:00",
  "fecha_recordatorio": "2024-02-12T00:00:00",
  "dias_hasta_mantenimiento": 31,
  "urgencia": "BAJA",
  "recomendaciones": ["Revisión periódica del equipo", "..."],
  "fechas_futuras": {
    "fecha_mantenimiento": "2024-02-15T00:00:00",
    "fecha_recordatorio": "2024-02-12T00:00:00",
    "dias_hasta_mantenimiento": 31,
    "urgencia": "BAJA"
  }
}
``` 