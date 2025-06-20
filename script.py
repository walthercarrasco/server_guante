import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import threading
import time
import sys
import platform

# Importar capas personalizadas
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

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

def list_available_ports():
    """Listar todos los puertos COM disponibles"""
    ports = serial.tools.list_ports.comports()
    available_ports = []
    
    print("\n🔍 Puertos COM disponibles:")
    print("=" * 50)
    
    if not ports:
        print("❌ No se encontraron puertos COM disponibles")
        return []
    
    for i, port in enumerate(ports, 1):
        print(f"{i}. {port.device}")
        print(f"   Descripción: {port.description}")
        print(f"   Fabricante: {port.manufacturer or 'Desconocido'}")
        print(f"   VID:PID: {port.vid}:{port.pid}" if port.vid and port.pid else "   VID:PID: No disponible")
        print("-" * 30)
        available_ports.append(port.device)
    
    return available_ports

def test_port_connection(port, baud_rate=115200, timeout=2):
    """Probar conexión a un puerto específico"""
    try:
        print(f"🔧 Probando conexión a {port}...")
        
        # Intentar abrir el puerto
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(1)  # Esperar un momento
        
        # Verificar si hay datos disponibles
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"✅ Conexión exitosa. Datos recibidos: {data[:50]}...")
        else:
            print(f"✅ Conexión exitosa. Puerto abierto correctamente.")
        
        ser.close()
        return True
        
    except serial.SerialException as e:
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            print(f"❌ Error de permisos en {port}: {e}")
            print("   💡 Soluciones posibles:")
            print("      - Cerrar Arduino IDE si está abierto")
            print("      - Cerrar cualquier monitor serie activo")
            print("      - Desconectar y reconectar el Arduino")
            print("      - Ejecutar como administrador")
        elif "FileNotFoundError" in str(e) or "could not open port" in str(e):
            print(f"❌ Puerto {port} no disponible: {e}")
        else:
            print(f"❌ Error en {port}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado en {port}: {e}")
        return False

def select_com_port():
    """Seleccionar puerto COM interactivamente"""
    available_ports = list_available_ports()
    
    if not available_ports:
        print("\n❌ No hay puertos COM disponibles.")
        print("💡 Verifica que:")
        print("   - El Arduino esté conectado")
        print("   - Los drivers estén instalados")
        print("   - El cable USB funcione correctamente")
        return None
    
    print(f"\n📋 Selecciona un puerto COM:")
    for i, port in enumerate(available_ports, 1):
        print(f"{i}. {port}")
    
    while True:
        try:
            choice = input(f"\nIngresa el número (1-{len(available_ports)}) o 'q' para salir: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            port_index = int(choice) - 1
            if 0 <= port_index < len(available_ports):
                selected_port = available_ports[port_index]
                
                # Probar la conexión
                if test_port_connection(selected_port):
                    print(f"✅ Puerto seleccionado: {selected_port}")
                    return selected_port
                else:
                    print(f"\n❌ No se pudo conectar a {selected_port}")
                    retry = input("¿Intentar con otro puerto? (s/n): ").strip().lower()
                    if retry != 's':
                        return None
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Por favor ingresa un número válido")
        except KeyboardInterrupt:
            print("\n\n👋 Operación cancelada")
            return None

def troubleshoot_connection_issues():
    """Guía de solución de problemas"""
    print("\n🔧 GUÍA DE SOLUCIÓN DE PROBLEMAS")
    print("=" * 50)
    
    print("\n1. 📱 VERIFICAR HARDWARE:")
    print("   ✓ Arduino conectado correctamente")
    print("   ✓ Cable USB funcionando")
    print("   ✓ LED de alimentación encendido")
    
    print("\n2. 💻 VERIFICAR SOFTWARE:")
    print("   ✓ Cerrar Arduino IDE")
    print("   ✓ Cerrar monitor serie")
    print("   ✓ Cerrar otras aplicaciones que usen el puerto")
    
    print("\n3. 🔄 REINICIAR CONEXIÓN:")
    print("   ✓ Desconectar y reconectar Arduino")
    print("   ✓ Cambiar puerto USB")
    print("   ✓ Reiniciar el script")
    
    print("\n4. 🛡️ PERMISOS:")
    if platform.system() == "Windows":
        print("   ✓ Ejecutar como administrador")
        print("   ✓ Verificar drivers del dispositivo")
    else:
        print("   ✓ Agregar usuario al grupo dialout: sudo usermod -a -G dialout $USER")
        print("   ✓ Reiniciar sesión después del comando anterior")
    
    print("\n5. 🔍 VERIFICAR DISPOSITIVO:")
    print("   ✓ Administrador de dispositivos (Windows)")
    print("   ✓ Buscar dispositivos con errores")

class SignLanguageAnalyzer:
    def __init__(self, com_port=None, baud_rate=115200, 
                 model_path='modelo.h5', scaler_path=None):
        self.com_port = com_port
        self.baud_rate = baud_rate
        
        # Configuración del modelo según el entrenamiento
        self.sequence_length = 68  # Longitud de secuencia del modelo entrenado
        self.feature_size = 10     # 5 angles + 2 roll/pitch + 3 gyro angles
        
        # Buffer para secuencias
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.raw_data = deque(maxlen=100)
        
        # Modelo y scaler
        self.model = None
        self.scaler = None
        
        # Mapeo de gestos según el entrenamiento
        self.gesture_map = {0: 'HOLA', 1: 'BIEN', 2: 'ADIOS', 3: 'SI'}
        
        # Control de hilos
        self.running = False
        self.serial_thread = None
        self.analysis_thread = None
        self.serial_connection = None
        
        # Cargar modelo
        self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path=None):
        """Cargar modelo preentrenado con capas personalizadas"""
        try:
            # Definir objetos personalizados para cargar el modelo
            custom_objects = {
                'FeatureWeightingLayer': FeatureWeightingLayer
            }
            
            # Cargar modelo con objetos personalizados
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Modelo cargado desde: {model_path}")
            
            # Cargar scaler si existe
            if scaler_path:
                try:
                    self.scaler = joblib.load(scaler_path)
                    print(f"Scaler cargado desde: {scaler_path}")
                except:
                    print("No se pudo cargar el scaler, usando datos sin normalizar")
                    self.scaler = None
            
            # Mostrar información del modelo
            print(f"Forma de entrada del modelo: {self.model.input_shape}")
            print(f"Forma de salida del modelo: {self.model.output_shape}")
            
            # Mostrar resumen del modelo
            print("\nArquitectura del modelo:")
            self.model.summary()
            
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print("Asegúrate de que el archivo del modelo existe y contiene las capas personalizadas correctas")
            return False
    
    def connect_arduino(self):
        """Conectar con Arduino con selección automática de puerto"""
        # Si no se especificó puerto, seleccionar automáticamente
        if not self.com_port:
            print("🔍 No se especificó puerto COM. Iniciando selección automática...")
            self.com_port = select_com_port()
            
            if not self.com_port:
                print("❌ No se seleccionó ningún puerto")
                return False
        
        # Intentar conectar al puerto seleccionado
        try:
            print(f"🔌 Conectando a {self.com_port}...")
            self.serial_connection = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"✅ Conectado a Arduino en {self.com_port}")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Error conectando a Arduino: {e}")
            
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                print("\n🚨 ERROR DE PERMISOS DETECTADO")
                troubleshoot_connection_issues()
                
                # Ofrecer seleccionar otro puerto
                retry = input("\n¿Intentar con otro puerto? (s/n): ").strip().lower()
                if retry == 's':
                    self.com_port = select_com_port()
                    if self.com_port:
                        return self.connect_arduino()  # Recursión para intentar de nuevo
            
            return False
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False
    
    def read_sensor_data(self):
        """Leer datos del Arduino en formato CSV directo"""
        while self.running:
            try:
                if self.serial_connection and self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Procesar datos en formato CSV (como en test_data_collector.py)
                    if line and ',' in line and not line.startswith('angle1'):  # Evitar header
                        try:
                            # Dividir la línea por comas
                            parts = line.split(',')
                            
                            # Verificar que tenemos al menos 10 valores (sin timestamp)
                            if len(parts) >= 10:
                                # Extraer las primeras 10 características en el orden correcto:
                                # ['angle1', 'angle2', 'angle3', 'angle4', 'angle5', 
                                #  'rolldeg', 'pitchdeg', 'anglegx', 'anglegy', 'anglegz']
                                features = []
                                
                                for i in range(10):
                                    try:
                                        value = float(parts[i])
                                        features.append(value)
                                    except (ValueError, IndexError):
                                        features.append(0.0)  # Valor por defecto si hay error
                                
                                # Asegurar que tenemos exactamente 10 características
                                if len(features) == self.feature_size:
                                    # Agregar al buffer
                                    self.data_buffer.append(features)
                                    
                                    # Guardar datos completos para debug
                                    self.raw_data.append({
                                        'timestamp': time.time(),
                                        'features': features,
                                        'original_line': line
                                    })
                                    
                                    # # Mostrar datos cada 20 muestras para no saturar la consola
                                    # if len(self.raw_data) % 20 == 0:
                                    #     print(f"📊 Datos recibidos: {[round(x,1) for x in features[:10]]}... (muestra {len(self.raw_data)})")
                            
                        except Exception as e:
                            # Continuar si hay error en una línea específica
                            continue
                            
            except Exception as e:
                print(f"Error leyendo datos: {e}")
                time.sleep(0.1)
    
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
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Mapear a gesto
        gesture = self.gesture_map.get(predicted_class, f"UNKNOWN_{predicted_class}")
        
        return gesture, confidence, prediction[0]
    
    def continuous_analysis(self):
        """Análisis continuo de gestos"""
        last_analysis_time = 0
        last_display_time = 0
        analysis_interval = 0.8  # Analizar cada 800ms
        display_interval = 5.0   # Mostrar estado cada 5 segundos
        
        confidence_threshold = 0.7  # Umbral para mostrar predicción
        last_prediction = None  # Variable para almacenar la última predicción
        
        print("🔄 Iniciando análisis continuo...")
        print(f"📊 Esperando {self.sequence_length} muestras para comenzar predicciones...")
        print("🎯 Las letras predichas aparecerán aquí:")
        print("-" * 50)
        
        while self.running:
            current_time = time.time()
            
            # Mostrar estado del buffer periódicamente
            if current_time - last_display_time >= display_interval:
                buffer_size = len(self.data_buffer)
                data_count = len(self.raw_data)
                
                if buffer_size < self.sequence_length:
                    print(f"\n📈 Estado: Recopilando datos {buffer_size}/{self.sequence_length} muestras")
                else:
                    pass
                    #print(f"\n📊 Estado: Analizando gestos... (Total datos: {data_count})")
                
                last_display_time = current_time
            
            # Realizar análisis si tenemos suficientes datos
            if current_time - last_analysis_time >= analysis_interval:
                if len(self.data_buffer) >= self.sequence_length:
                    gesture, confidence, full_prediction = self.analyze_gesture()
                    
                    if gesture and confidence > confidence_threshold:
                        # Solo mostrar si la predicción es diferente a la anterior
                        if gesture != last_prediction:
                            print(f"{gesture}", end="  ", flush=True)
                            last_prediction = gesture
                
                last_analysis_time = current_time
            
            time.sleep(0.1)
    
    def start_analysis(self):
        """Iniciar análisis en tiempo real"""
        if not self.model:
            print("❌ Error: No se pudo cargar el modelo")
            return False
        
        if not self.connect_arduino():
            print("❌ Error: No se pudo conectar con Arduino")
            return False
        
        print("\n🚀 INICIANDO ANÁLISIS EN TIEMPO REAL")
        print("=" * 60)
        print(f"🎯 Gestos disponibles: {list(self.gesture_map.values())}")
        print(f"📏 Necesita {self.sequence_length} muestras para cada predicción")
        print("⚡ Frecuencia de análisis: cada 0.8 segundos")
        print("🎚️ Umbral de confianza: 0.7 (solo muestra predicciones seguras)")
        print("\n💡 Las letras aparecerán en la consola cuando se detecten gestos")
        print("💡 Presiona Ctrl+C para detener el análisis")
        print("=" * 60)
        
        self.running = True
        
        # Iniciar hilos
        self.serial_thread = threading.Thread(target=self.read_sensor_data, daemon=True)
        self.analysis_thread = threading.Thread(target=self.continuous_analysis, daemon=True)
        
        self.serial_thread.start()
        self.analysis_thread.start()
        
        try:
            # Mantener el programa corriendo y mostrar estadísticas
            start_time = time.time()
            while True:
                time.sleep(1)
                
                # Verificar que los hilos sigan corriendo
                if not self.serial_thread.is_alive():
                    print("⚠️ Hilo de lectura serial se detuvo")
                    break
                if not self.analysis_thread.is_alive():
                    print("⚠️ Hilo de análisis se detuvo")
                    break
        
        except KeyboardInterrupt:
            print("\n\n🛑 Deteniendo análisis por solicitud del usuario...")
            self.stop_analysis()
    
    def stop_analysis(self):
        """Detener análisis"""
        print("\n🛑 Deteniendo análisis...")
        self.running = False
        
        if self.serial_thread:
            self.serial_thread.join()
        if self.analysis_thread:
            self.analysis_thread.join()
        if self.serial_connection:
            self.serial_connection.close()
        
        print("✅ Análisis detenido")
    
    def test_data_reception(self):
        """Probar recepción de datos sin análisis de modelo"""
        print("\n🔍 MODO PRUEBA - RECEPCIÓN DE DATOS")
        print("=" * 50)
        print("Este modo solo muestra los datos recibidos del Arduino")
        print("Útil para verificar que los datos lleguen correctamente")
        print("Presiona Ctrl+C para detener")
        print("=" * 50)
        
        if not self.connect_arduino():
            return
        
        self.running = True
        samples_received = 0
        start_time = time.time()
        
        try:
            while self.running:
                if self.serial_connection and self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line and ',' in line and not line.startswith('angle1'):
                        parts = line.split(',')
                        if len(parts) >= 10:
                            samples_received += 1
                            
                            # Mostrar cada 10 muestras
                            if samples_received % 10 == 0:
                                elapsed = time.time() - start_time
                                rate = samples_received / elapsed if elapsed > 0 else 0
                                
                                print(f"\n📊 Muestra #{samples_received} (Frecuencia: {rate:.1f} Hz)")
                                print(f"   Datos: {line}")
                                print(f"   Valores: {[round(float(x), 2) for x in parts[:10]]}")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\n\n✅ Prueba completada:")
            elapsed = time.time() - start_time
            print(f"   ⏱️ Tiempo: {elapsed:.1f} segundos")
            print(f"   📊 Muestras: {samples_received}")
            if elapsed > 0:
                print(f"   📈 Frecuencia promedio: {samples_received/elapsed:.1f} Hz")
        finally:
            self.stop_analysis()

    def test_single_prediction(self):
        """Hacer una sola predicción para pruebas"""
        print("\n🎯 MODO PRUEBA - PREDICCIÓN ÚNICA")
        print("=" * 50)
        
        if not self.model:
            print("❌ Error: Modelo no cargado")
            return
            
        if not self.connect_arduino():
            return
        
        self.running = True
        self.serial_thread = threading.Thread(target=self.read_sensor_data, daemon=True)
        self.serial_thread.start()
        
        print(f"📊 Esperando {self.sequence_length} muestras para hacer predicción...")
        
        # Esperar hasta tener suficientes datos
        while len(self.data_buffer) < self.sequence_length:
            current_samples = len(self.data_buffer)
            print(f"📈 Recopilando datos... {current_samples}/{self.sequence_length} ({current_samples/self.sequence_length*100:.1f}%)")
            time.sleep(2)
        
        # Hacer predicción
        print("\n🔄 Realizando predicción...")
        gesture, confidence, full_prediction = self.analyze_gesture()
        
        print(f"\n🎯 RESULTADO:")
        print("=" * 30)
        if confidence > 0.7:
            print(f"   🔤 LETRA PREDICHA: {gesture}")
            print(f"   📊 Confianza: {confidence:.3f}")
        else:
            print(f"   ❓ Predicción incierta: {gesture}")
            print(f"   📊 Confianza baja: {confidence:.3f}")
            print("   💡 Intenta hacer el gesto más claro")
        
        print("\n✅ Predicción completada")
        self.stop_analysis()

# Función principal
def main():
    print("🤖 ANALIZADOR DE LENGUAJE DE SEÑAS")
    print("=" * 50)
    
    # Crear analizador sin puerto específico (selección automática)
    analyzer = SignLanguageAnalyzer(
        com_port=None,  # Selección automática
        model_path='./modelo.h5',  # Cambiar por la ruta de tu modelo
        scaler_path=None  # Cambiar por la ruta de tu scaler (opcional)
    )
    
    if not analyzer.model:
        print("❌ No se pudo cargar el modelo. Verifica la ruta del archivo.")
        return
    
    print("\n📋 Opciones disponibles:")
    print("1. 🔄 Análisis continuo en tiempo real (requiere modelo)")
    print("2. 🎯 Predicción única de prueba (requiere modelo)")
    print("3. 📊 Probar recepción de datos (sin modelo)")
    print("4. 🔍 Solo probar conexión COM")
    print("5. 🛠️ Mostrar guía de solución de problemas")
    
    choice = input("\nSelecciona una opción (1-5): ").strip()
    
    if choice == '1':
        if analyzer.model:
            analyzer.start_analysis()
        else:
            print("❌ Esta opción requiere que el modelo esté cargado")
    elif choice == '2':
        if analyzer.model:
            analyzer.test_single_prediction()
        else:
            print("❌ Esta opción requiere que el modelo esté cargado")
    elif choice == '3':
        analyzer.test_data_reception()
    elif choice == '4':
        # Solo probar conexión
        if analyzer.connect_arduino():
            print("✅ Conexión exitosa!")
            analyzer.serial_connection.close()
        else:
            print("❌ No se pudo establecer conexión")
    elif choice == '5':
        troubleshoot_connection_issues()
    else:
        print("❌ Opción inválida")

if __name__ == "__main__":
    main()