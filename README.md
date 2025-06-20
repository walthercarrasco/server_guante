# Smart Glove Server con Análisis de Gestos

Servidor WebSocket para recibir datos del Smart Glove ESP32-S3 con análisis de lenguaje de señas en tiempo real.

## 🌟 Características

- **Servidor WebSocket Multi-Cliente**: Soporta ESP32 y aplicaciones móviles simultáneamente
- **Análisis de Gestos ML**: Detecta gestos del lenguaje de señas usando TensorFlow
- **Broadcast en Tiempo Real**: Envía gestos detectados a todas las apps móviles conectadas
- **Guardado de Datos**: Almacena todos los datos en formato CSV
- **Visualización Configurable**: Muestra solo gestos detectados o todos los datos

## 📋 Requisitos

- Python 3.8+
- TensorFlow
- Websockets
- NumPy
- Archivo `modelo.h5` (modelo entrenado)

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

En `main.py`, puedes configurar:

```python
# Servidor
HOST = "0.0.0.0"  # Escuchar en todas las interfaces
PORT = 8080

# Archivos
CSV_FILE = "smart_glove_data.csv"
LOG_FILE = "server.log"
MODEL_PATH = "modelo.h5"

# Visualización
SHOW_ONLY_GESTURES = True  # True: solo gestos, False: todos los datos
```

## 🎯 Gestos Disponibles

El modelo está entrenado para detectar:
- **BIEN** 👍
- **SI** ✓
- **HOLA** 👋
- **ADIOS** 👋

## 🏃 Uso

1. **Iniciar el servidor**:
```bash
python main.py
```

2. **Conectar el Smart Glove**:
   - Configura el ESP32-S3 para conectarse a `ws://[IP_SERVIDOR]:8080`

3. **Ver resultados**:
   - Los gestos detectados aparecerán en la consola
   - Todos los datos se guardan en `smart_glove_data.csv`

## 📊 Formato de Datos

El servidor espera datos JSON con esta estructura:

```json
{
    "timestamp": 123456789,
    "flex_sensors": [90.5, 85.2, 92.1, 88.3, 91.7],
    "mpu6050": {
        "roll": 12.5,
        "pitch": -5.3,
        "angle_gx": 0.8,
        "angle_gy": -1.2,
        "angle_gz": 2.1,
        "accel_x": 0.98,
        "accel_y": 0.05,
        "accel_z": 0.12,
        "gyro_x": 0.5,
        "gyro_y": -0.3,
        "gyro_z": 0.2
    }
}
```

## 📈 Análisis de Gestos

- **Buffer**: Necesita 68 muestras para comenzar predicciones
- **Frecuencia**: Analiza cada 0.8 segundos
- **Umbral**: Solo muestra gestos con confianza ≥ 0.7
- **CSV**: Incluye columnas `predicted_gesture` y `confidence`

## 🔧 Solución de Problemas

Si el modelo no carga:
- Verifica que `modelo.h5` existe en el directorio
- El servidor funcionará sin análisis de gestos

Si hay errores de codificación en Windows:
- El servidor maneja automáticamente UTF-8 para logs y consola

## 📱 Conexión Multi-Cliente

El servidor ahora soporta múltiples tipos de clientes:

### Tipos de Clientes:

1. **ESP32** - Envía datos de sensores
   - Se identifica automáticamente por estructura JSON
   - Recibe confirmación y análisis de gestos

2. **Aplicación Móvil** - Recibe gestos detectados
   - Debe identificarse con `{"client_type": "mobile"}`
   - Recibe broadcast de todos los gestos detectados
   - Puede solicitar el último gesto con `{"action": "get_last_gesture"}`

### Flujo de Comunicación:

```
ESP32 → Servidor → Modelo ML → Broadcast → Apps Móviles
         ↓
      Archivo CSV
```

### Estadísticas en Logs:

```
Clientes conectados - Total: 3, ESP32: 1, Móvil: 2
```

Consulta `API_MOBILE_APP.md` para detalles de implementación de la app móvil.
