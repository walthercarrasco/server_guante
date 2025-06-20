#!/usr/bin/env python3
"""
Servidor WebSocket para recibir datos del Smart Glove ESP32-S3
Guarda los datos en tiempo real, los muestra en consola y realiza análisis de gestos
"""

import asyncio
import websockets
import json
import logging
import csv
import os
from datetime import datetime
import threading
import time
import numpy as np
import tensorflow as tf
import joblib
from collections import deque

# Importar capas personalizadas
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Configuración
HOST = "0.0.0.0"  # Escuchar en todas las interfaces
PORT = 8080
CSV_FILE = "smart_glove_data.csv"
LOG_FILE = "server.log"

# Configuración del modelo
MODEL_PATH = "modelo.h5"
SCALER_PATH = None  # Cambiar si tienes un scaler guardado

# Configuración de visualización
SHOW_ONLY_GESTURES = True  # True: solo muestra gestos detectados, False: muestra todos los datos

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class FeatureWeightingLayer(Layer):
    def __init__(self, scaling_factors, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        self.scaling_factors = K.variable(scaling_factors, name='scaling_factors')
    
    def call(self, inputs):
        return inputs * self.scaling_factors
    
    def get_config(self):
        config = super(FeatureWeightingLayer, self).get_config()
        config.update({"scaling_factors": self.scaling_factors.numpy()})
        return config

class GestureAnalyzer:
    """Analizador de gestos para los datos del Smart Glove"""
    
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        # Configuración del modelo según el entrenamiento
        self.sequence_length = 68  # Longitud de secuencia del modelo entrenado
        self.feature_size = 10     # 5 flex + 2 roll/pitch + 3 gyro angles
        
        # Buffer para secuencias
        self.data_buffer = deque(maxlen=self.sequence_length)
        
        # Modelo y scaler
        self.model = None
        self.scaler = None
        
        # Mapeo de gestos según el entrenamiento
        self.gesture_map = {0: 'HOLA', 1: 'GRACIAS', 2: 'NO', 3: 'BIEN', 4: 'SI'}
        
        # Control de predicciones
        self.last_prediction = None
        self.last_analysis_time = 0
        self.analysis_interval = 0.8  # Analizar cada 800ms
        self.confidence_threshold = 0.7  # Umbral para mostrar predicción
        
        # Cargar modelo
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path=None):
        """Cargar modelo preentrenado con capas personalizadas"""
        try:
            # Verificar si el archivo del modelo existe
            if not os.path.exists(model_path):
                logging.warning(f"Archivo del modelo no encontrado: {model_path}")
                logging.info("El servidor funcionará sin análisis de gestos")
                return False
            
            # Definir objetos personalizados para cargar el modelo
            custom_objects = {
                'FeatureWeightingLayer': FeatureWeightingLayer
            }
            
            # Cargar modelo con objetos personalizados
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            logging.info(f"Modelo cargado desde: {model_path}")
            
            # Cargar scaler si existe
            if scaler_path and os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logging.info(f"Scaler cargado desde: {scaler_path}")
                except Exception as e:
                    logging.warning(f"No se pudo cargar el scaler: {e}")
                    self.scaler = None
            
            # Mostrar información del modelo
            logging.info(f"Forma de entrada del modelo: {self.model.input_shape}")
            logging.info(f"Forma de salida del modelo: {self.model.output_shape}")
            logging.info(f"Gestos disponibles: {list(self.gesture_map.values())}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error cargando modelo: {e}")
            logging.info("El servidor funcionará sin análisis de gestos")
            return False
    
    def extract_features_from_json(self, data):
        """Extraer características del JSON en el formato correcto para el modelo"""
        try:
            # Extraer datos del JSON
            flex_data = data.get('flex_sensors', [0, 0, 0, 0, 0])
            mpu_data = data.get('mpu6050', {})
            
            # Asegurar que tenemos 5 valores de flex
            while len(flex_data) < 5:
                flex_data.append(0)
            flex_data = flex_data[:5]  # Tomar solo los primeros 5
            
            # Extraer características en el orden correcto:
            # [angle1, angle2, angle3, angle4, angle5, rolldeg, pitchdeg, anglegx, anglegy, anglegz]
            features = [
                flex_data[0],  # angle1 (flex sensor 1)
                flex_data[1],  # angle2 (flex sensor 2)
                flex_data[2],  # angle3 (flex sensor 3)
                flex_data[3],  # angle4 (flex sensor 4)
                flex_data[4],  # angle5 (flex sensor 5)
                mpu_data.get('roll', 0),      # rolldeg
                mpu_data.get('pitch', 0),     # pitchdeg
                mpu_data.get('angle_gx', 0),  # anglegx
                mpu_data.get('angle_gy', 0),  # anglegy
                mpu_data.get('angle_gz', 0)   # anglegz
            ]
            
            return features
            
        except Exception as e:
            logging.error(f"Error extrayendo características: {e}")
            return [0.0] * self.feature_size
    
    def preprocess_data(self, sequence):
        """Preprocesar datos según el modelo entrenado"""
        # Convertir a numpy array
        sequence = np.array(sequence)
        
        # Ajustar longitud de secuencia si es necesario
        if len(sequence) < self.sequence_length:
            # Rellenar con ceros o repetir último valor
            padding_size = self.sequence_length - len(sequence)
            if len(sequence) > 0:
                # Repetir último valor
                last_value = sequence[-1]
                padding = np.tile(last_value, (padding_size, 1))
            else:
                # Rellenar con ceros
                padding = np.zeros((padding_size, self.feature_size))
            
            sequence = np.vstack([padding, sequence])
        
        elif len(sequence) > self.sequence_length:
            # Tomar los últimos valores
            sequence = sequence[-self.sequence_length:]
        
        # Aplicar normalización si existe scaler
        if self.scaler is not None:
            # Reshape para normalizar
            original_shape = sequence.shape
            sequence_flat = sequence.reshape(-1, self.feature_size)
            sequence_normalized = self.scaler.transform(sequence_flat)
            sequence = sequence_normalized.reshape(original_shape)
        
        # Reshape para el modelo (batch_size=1)
        return sequence.reshape(1, self.sequence_length, self.feature_size)
    
    def analyze_gesture(self):
        """Analizar gesto actual"""
        if not self.model or len(self.data_buffer) < self.sequence_length:
            return None, 0.0, None
        
        # Obtener secuencia actual
        current_sequence = list(self.data_buffer)
        
        # Preprocesar datos
        processed_data = self.preprocess_data(current_sequence)
        
        # Hacer predicción
        prediction = self.model.predict(processed_data, verbose=0)
        
        # Obtener resultado
        predicted_class = int(np.argmax(prediction[0]))  # Convertir a int estándar
        confidence = float(np.max(prediction[0]))  # Convertir a float estándar
        
        # Mapear a gesto
        gesture = self.gesture_map.get(predicted_class, f"UNKNOWN_{predicted_class}")
        
        return gesture, confidence, prediction[0]
    
    def process_data(self, json_data):
        """Procesar datos JSON y realizar análisis si es necesario"""
        if not self.model:
            return None, 0.0  # Sin modelo, no hay análisis
        
        # Extraer características
        features = self.extract_features_from_json(json_data)
        
        # Agregar al buffer
        self.data_buffer.append(features)
        
        # Verificar si es tiempo de analizar
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            if len(self.data_buffer) >= self.sequence_length:
                gesture, confidence, _ = self.analyze_gesture()
                
                if gesture and confidence > self.confidence_threshold:
                    # Solo retornar si la predicción es diferente a la anterior
                    if gesture != self.last_prediction:
                        self.last_prediction = gesture
                        self.last_analysis_time = current_time
                        return gesture, confidence
            
            self.last_analysis_time = current_time
        
        return None, 0.0  # Valores por defecto seguros

class SmartGloveServer:
    def __init__(self):
        self.connected_clients = set()
        self.esp32_clients = set()  # Clientes ESP32 que envían datos
        self.mobile_clients = set()  # Clientes móviles que reciben gestos
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.csv_writer = None
        self.csv_file_handle = None
        self.setup_csv_file()
        
        # Inicializar analizador de gestos
        self.gesture_analyzer = GestureAnalyzer()
        
        # Estadísticas de análisis
        self.total_predictions = 0
        self.successful_predictions = 0
        
        # Último gesto detectado para enviar a nuevos clientes
        self.last_detected_gesture = None
        self.last_gesture_confidence = 0.0
        self.last_gesture_time = None
        
    def setup_csv_file(self):
        """Configura el archivo CSV para guardar los datos"""
        file_exists = os.path.exists(CSV_FILE)
        
        self.csv_file_handle = open(CSV_FILE, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        
        # Escribir headers si el archivo es nuevo
        if not file_exists:
            headers = [
                'timestamp', 'datetime',
                'flex_sensor_1', 'flex_sensor_2', 'flex_sensor_3', 
                'flex_sensor_4', 'flex_sensor_5',
                'roll', 'pitch', 'angle_gx', 'angle_gy', 'angle_gz',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'predicted_gesture', 'confidence'
            ]
            self.csv_writer.writerow(headers)
            self.csv_file_handle.flush()
            logging.info(f"Archivo CSV creado: {CSV_FILE}")

    async def register_client(self, websocket, client_type="unknown"):
        """Registra un nuevo cliente según su tipo"""
        self.connected_clients.add(websocket)
        
        if client_type == "esp32":
            self.esp32_clients.add(websocket)
            client_type_str = "ESP32"
        elif client_type == "mobile":
            self.mobile_clients.add(websocket)
            client_type_str = "Aplicación Móvil"
        else:
            client_type_str = "Desconocido"
        
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logging.info(f"Cliente {client_type_str} conectado: {client_info}")
        logging.info(f"Clientes conectados - Total: {len(self.connected_clients)}, ESP32: {len(self.esp32_clients)}, Móvil: {len(self.mobile_clients)}")
        
        # Si es un cliente móvil y hay un gesto reciente, enviarlo
        if client_type == "mobile" and self.last_detected_gesture:
            await self.send_gesture_to_client(websocket, self.last_detected_gesture, self.last_gesture_confidence)
    
    async def unregister_client(self, websocket):
        """Desregistra un cliente"""
        self.connected_clients.discard(websocket)
        self.esp32_clients.discard(websocket)
        self.mobile_clients.discard(websocket)
        
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logging.info(f"Cliente desconectado: {client_info}")
        logging.info(f"Clientes conectados - Total: {len(self.connected_clients)}, ESP32: {len(self.esp32_clients)}, Móvil: {len(self.mobile_clients)}")

    def save_data_to_csv(self, data, predicted_gesture=None, confidence=0.0):
        """Guarda los datos en el archivo CSV incluyendo predicción"""
        try:
            current_time = datetime.now()
            
            # Extraer datos del JSON
            flex_data = data.get('flex_sensors', [0, 0, 0, 0, 0])
            mpu_data = data.get('mpu6050', {})
            timestamp = data.get('timestamp', 0)
            
            # Preparar fila para CSV (convertir valores NumPy a tipos estándar)
            row = [
                timestamp,
                current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                *[float(x) for x in flex_data],  # Convertir sensores flex a float estándar
                float(mpu_data.get('roll', 0)),
                float(mpu_data.get('pitch', 0)),
                float(mpu_data.get('angle_gx', 0)),
                float(mpu_data.get('angle_gy', 0)),
                float(mpu_data.get('angle_gz', 0)),
                float(mpu_data.get('accel_x', 0)),
                float(mpu_data.get('accel_y', 0)),
                float(mpu_data.get('accel_z', 0)),
                float(mpu_data.get('gyro_x', 0)),
                float(mpu_data.get('gyro_y', 0)),
                float(mpu_data.get('gyro_z', 0)),
                predicted_gesture or '',
                float(confidence) if confidence is not None else 0.0
            ]
            
            with self.buffer_lock:
                self.csv_writer.writerow(row)
                self.csv_file_handle.flush()
                
        except Exception as e:
            logging.error(f"Error guardando datos en CSV: {e}")
    
    def display_data(self, data, predicted_gesture=None, confidence=0.0):
        """Muestra los datos en consola de forma formateada incluyendo predicción"""
        try:
            # Modo: Solo mostrar gestos
            if SHOW_ONLY_GESTURES:
                # Solo mostrar cuando se detecta un gesto con confianza suficiente
                if predicted_gesture and confidence >= self.gesture_analyzer.confidence_threshold:
                    timestamp = data.get('timestamp', 0)
                    current_time = datetime.now().strftime('%H:%M:%S')
                    
                    # Mostrar el gesto detectado de forma simple
                    print(f"\n[{current_time}] 🎯 GESTO DETECTADO: {predicted_gesture} (Confianza: {confidence:.3f})")
                    
                    # Mostrar estadísticas ocasionalmente (cada 10 predicciones exitosas)
                    if self.successful_predictions % 10 == 0:
                        if self.total_predictions > 0:
                            success_rate = (self.successful_predictions / self.total_predictions) * 100
                            print(f"    📊 Estadísticas: {self.total_predictions} análisis, {success_rate:.1f}% exitosos")
                
                # Mostrar estado del buffer solo cuando está llenándose por primera vez
                elif self.gesture_analyzer.model and len(self.gesture_analyzer.data_buffer) < self.gesture_analyzer.sequence_length:
                    buffer_status = len(self.gesture_analyzer.data_buffer)
                    percentage = (buffer_status / self.gesture_analyzer.sequence_length) * 100
                    
                    # Mostrar progreso cada 10 muestras
                    if buffer_status % 10 == 0 or buffer_status == 1:
                        print(f"\r📈 Recopilando datos: {buffer_status}/{self.gesture_analyzer.sequence_length} muestras ({percentage:.0f}%)", end='', flush=True)
                    
                    # Limpiar la línea cuando se completa el buffer
                    if buffer_status == self.gesture_analyzer.sequence_length:
                        print("\r✅ Buffer completo. Iniciando análisis de gestos...                    ")
                        print("🎯 Los gestos detectados aparecerán aquí:")
                        print("-" * 50)
            
            # Modo: Mostrar todos los datos
            else:
                flex_data = data.get('flex_sensors', [])
                mpu_data = data.get('mpu6050', {})
                timestamp = data.get('timestamp', 0)
                
                print("\n" + "="*60)
                print(f"SMART GLOVE DATA - Timestamp: {timestamp}")
                if predicted_gesture:
                    print(f"🎯 GESTO DETECTADO: {predicted_gesture} (Confianza: {confidence:.3f})")
                print("="*60)
                
                # Mostrar sensores flex
                print("SENSORES FLEXIBLES:")
                finger_names = ["Pulgar", "Índice", "Corazón", "Anular", "Meñique"]
                for i, value in enumerate(flex_data):
                    finger = finger_names[i] if i < len(finger_names) else f"Sensor {i+1}"
                    print(f"  {finger:>8}: {value:>8.1f}")
                
                print("\nMPU6050 - ORIENTACIÓN:")
                print(f"  {'Roll':>8}: {mpu_data.get('roll', 0):>8.2f}°")
                print(f"  {'Pitch':>8}: {mpu_data.get('pitch', 0):>8.2f}°")
                
                print("\nMPU6050 - ÁNGULOS:")
                print(f"  {'Angle GX':>8}: {mpu_data.get('angle_gx', 0):>8.2f}°")
                print(f"  {'Angle GY':>8}: {mpu_data.get('angle_gy', 0):>8.2f}°")
                print(f"  {'Angle GZ':>8}: {mpu_data.get('angle_gz', 0):>8.2f}°")
            
        except Exception as e:
            logging.error(f"Error mostrando datos: {e}")
    
    async def send_gesture_to_client(self, websocket, gesture, confidence):
        """Envía un gesto detectado a un cliente específico"""
        try:
            message = {
                "type": "gesture_detected",
                "gesture": gesture,
                "confidence": float(confidence),
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logging.error(f"Error enviando gesto a cliente: {e}")
    
    async def broadcast_gesture_to_mobile_clients(self, gesture, confidence):
        """Envía el gesto detectado a todos los clientes móviles conectados"""
        if not self.mobile_clients:
            return
        
        # Actualizar último gesto detectado
        self.last_detected_gesture = gesture
        self.last_gesture_confidence = confidence
        self.last_gesture_time = time.time()
        
        # Crear lista de tareas de envío
        tasks = []
        disconnected_clients = []
        
        for client in self.mobile_clients:
            try:
                tasks.append(self.send_gesture_to_client(client, gesture, confidence))
            except Exception:
                disconnected_clients.append(client)
        
        # Ejecutar todas las tareas de envío en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Limpiar clientes desconectados
        for client in disconnected_clients:
            await self.unregister_client(client)

    async def handle_client_message(self, websocket):
        """Maneja los mensajes de los clientes"""
        client_type = None
        
        try:
            # Esperar primer mensaje para identificar tipo de cliente
            first_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            
            try:
                data = json.loads(first_message)
                
                # Identificar tipo de cliente
                if "client_type" in data:
                    client_type = data.get("client_type", "unknown")
                elif "flex_sensors" in data and "mpu6050" in data:
                    client_type = "esp32"
                else:
                    client_type = "mobile"
                
                # Registrar cliente según su tipo
                await self.register_client(websocket, client_type)
                
                # Si es un mensaje del ESP32 con datos, procesarlo
                if client_type == "esp32" and "flex_sensors" in data:
                    await self.process_esp32_data(websocket, data)
                
                # Si es un cliente móvil solicitando el último gesto
                elif client_type == "mobile" and data.get("action") == "get_last_gesture":
                    if self.last_detected_gesture:
                        await self.send_gesture_to_client(websocket, self.last_detected_gesture, self.last_gesture_confidence)
                
            except json.JSONDecodeError:
                logging.error(f"Primer mensaje no es JSON válido: {first_message}")
                await self.register_client(websocket, "unknown")
            
            # Procesar mensajes subsiguientes
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Mensajes del ESP32
                    if client_type == "esp32" and "flex_sensors" in data:
                        await self.process_esp32_data(websocket, data)
                    
                    # Mensajes de la app móvil
                    elif client_type == "mobile":
                        # Manejar keep-alive o solicitudes específicas
                        if data.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                        elif data.get("action") == "get_last_gesture":
                            if self.last_detected_gesture:
                                await self.send_gesture_to_client(websocket, self.last_detected_gesture, self.last_gesture_confidence)
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON: {e}")
                except Exception as e:
                    logging.error(f"Error procesando mensaje: {type(e).__name__}: {e}")
                    
        except asyncio.TimeoutError:
            logging.warning("Cliente no envió mensaje de identificación en 10 segundos")
        except websockets.exceptions.ConnectionClosed:
            logging.info("Cliente desconectado normalmente")
        except Exception as e:
            logging.error(f"Error en conexión con cliente: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def process_esp32_data(self, websocket, data):
        """Procesa datos recibidos del ESP32"""
        # Realizar análisis de gestos
        predicted_gesture, confidence = self.gesture_analyzer.process_data(data)
        
        # Actualizar estadísticas
        if self.gesture_analyzer.model and len(self.gesture_analyzer.data_buffer) >= self.gesture_analyzer.sequence_length:
            self.total_predictions += 1
            if predicted_gesture:
                self.successful_predictions += 1
                # Enviar gesto a todos los clientes móviles
                await self.broadcast_gesture_to_mobile_clients(predicted_gesture, confidence)
        
        # Guardar en CSV
        self.save_data_to_csv(data, predicted_gesture, confidence)
        
        # Mostrar en consola
        self.display_data(data, predicted_gesture, confidence)
        
        # Responder al ESP32 con información adicional
        response = {
            "status": "ok",
            "timestamp": time.time(),
            "gesture_analysis": {
                "predicted_gesture": predicted_gesture,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "buffer_size": len(self.gesture_analyzer.data_buffer),
                "model_ready": self.gesture_analyzer.model is not None
            }
        }
        
        try:
            await websocket.send(json.dumps(response))
        except Exception as e:
            logging.error(f"Error enviando respuesta a ESP32: {e}")

    async def start_server(self):
        """Inicia el servidor WebSocket"""
        logging.info(f"Iniciando servidor WebSocket en {HOST}:{PORT}")
        logging.info(f"Los datos se guardarán en: {CSV_FILE}")
        
        # Mostrar información del modelo
        if self.gesture_analyzer.model:
            logging.info("Análisis de gestos ACTIVADO")
            logging.info(f"Gestos disponibles: {list(self.gesture_analyzer.gesture_map.values())}")
            logging.info(f"Necesita {self.gesture_analyzer.sequence_length} muestras para predicción")
            logging.info(f"Frecuencia de análisis: cada {self.gesture_analyzer.analysis_interval} segundos")
            logging.info(f"Umbral de confianza: {self.gesture_analyzer.confidence_threshold}")
        else:
            logging.info("Análisis de gestos DESACTIVADO (modelo no disponible)")
        
        logging.info("Presiona Ctrl+C para detener el servidor")
        
        try:
            async with websockets.serve(
                self.handle_client_message, 
                HOST, 
                PORT,
                ping_interval=20,
                ping_timeout=10
            ):
                await asyncio.Future()  # Ejecutar indefinidamente
            
        except KeyboardInterrupt:
            logging.info("Servidor detenido por el usuario")
        except Exception as e:
            logging.error(f"Error en servidor: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos al cerrar"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
        
        # Mostrar estadísticas finales
        if self.total_predictions > 0:
            success_rate = (self.successful_predictions / self.total_predictions) * 100
            logging.info(f"Estadísticas finales: {self.total_predictions} análisis, {success_rate:.1f}% exitosos")
        
        logging.info("Servidor cerrado correctamente")

def main():
    """Función principal"""
    print("=" * 60)
    print("SERVIDOR WEBSOCKET PARA SMART GLOVE ESP32-S3")
    print("CON ANÁLISIS DE GESTOS EN TIEMPO REAL")
    print("=" * 60)
    print(f"Servidor WebSocket: ws://{HOST}:{PORT}")
    print(f"Archivo de datos: {CSV_FILE}")
    print(f"Archivo de log: {LOG_FILE}")
    print(f"Modelo ML: {MODEL_PATH}")
    print("=" * 60)
    print("\n📌 MODO DE VISUALIZACIÓN:")
    print("   Solo se mostrarán los gestos detectados con confianza >= 0.7")
    print("   Los datos completos se guardan en el archivo CSV")
    print("=" * 60)
    
    server = SmartGloveServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario")
    except Exception as e:
        logging.error(f"Error ejecutando servidor: {e}")

if __name__ == "__main__":
    main()