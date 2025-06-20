#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// Configuración WiFi
const char* ssid = "WiFi Carrasco";
const char* password = "W1a6l2t0h0e4r";

// Configuración WebSocket
const char* websocket_server = "192.168.31.222"; // IP de tu servidor
const int websocket_port = 8080;

// Cliente WebSocket
WebSocketsClient webSocket;

// Configuración del MPU6050
MPU6050 mpu(0x69);

// Variables Globales para los Offsets Calculados Automáticamente
int16_t auto_ax_offset = 0;
int16_t auto_ay_offset = 0;
int16_t auto_az_offset = 0;
int16_t auto_gx_offset = 0;
int16_t auto_gy_offset = 0;
int16_t auto_gz_offset = 0;

// Constantes para Sensores Flexibles
const int flexPins[] = {4, 5, 6, 7, 15};
const int numFlexSensors = 5;
const float VCC_FLEX = 3.3;
const float R_DIV_FLEX = 10000.0;
const float FLEX_FLAT_RESISTANCES[numFlexSensors] = {
  25000.0, 25000.0, 25000.0, 25000.0, 25000.0
};
const float FLEX_BEND_RESISTANCES[numFlexSensors] = {
  50000.0, 50000.0, 50000.0, 50000.0, 50000.0
};

// Variables para MPU6050
int16_t ax_raw, ay_raw, az_raw;
int16_t gx_raw, gy_raw, gz_raw;
int16_t ax_cal, ay_cal, az_cal;
int16_t gx_cal, gy_cal, gz_cal;
float accel_x_g, accel_y_g, accel_z_g;
float gyro_x_dps, gyro_y_dps, gyro_z_dps;
float roll_deg, pitch_deg;
float angle_gx_deg = 0.0, angle_gy_deg = 0.0, angle_gz_deg = 0.0;

// Variables para filtrado
float last_flex_values[numFlexSensors] = {0};
float last_ax = 0, last_ay = 0, last_az = 0;
float last_gx = 0, last_gy = 0, last_gz = 0;
const float alpha = 0.3;

unsigned long prev_time_mpu = 0;
float dt_mpu;

// Estado de conexión
bool wifiConnected = false;
bool websocketConnected = false;

// Flag para imprimir CSV
bool printCSV = true;  // Cambiar a false para desactivar impresión CSV
bool csvHeaderPrinted = false;

// Función para eventos WebSocket
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WebSocket] Desconectado");
      websocketConnected = false;
      break;
      
    case WStype_CONNECTED:
      Serial.printf("[WebSocket] Conectado a: %s\n", payload);
      websocketConnected = true;
      break;
      
    case WStype_TEXT:
      Serial.printf("[WebSocket] Mensaje recibido: %s\n", payload);
      break;
      
    case WStype_ERROR:
      Serial.println("[WebSocket] Error de conexión");
      websocketConnected = false;
      break;
      
    default:
      break;
  }
}

// Función para conectar WiFi
void connectWiFi() {
  Serial.println("Conectando a WiFi...");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println();
    Serial.println("WiFi conectado!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("Error: No se pudo conectar a WiFi");
    wifiConnected = false;
  }
}

// Función para calcular offsets automáticamente
void calcularOffsetsAutomaticosMPU6050() {
  long ax_sum = 0, ay_sum = 0, az_sum = 0;
  long gx_sum = 0, gy_sum = 0, gz_sum = 0;
  const int num_muestras_calibracion = 1000;

  Serial.println("\n=== INICIO DE CALIBRACIÓN MPU6050 ===");
  Serial.println("IMPORTANTE: Coloca el guante en una superficie plana");
  Serial.println("y mantenlo completamente inmóvil durante la calibración.");
  Serial.println("La calibración comenzará en 3 segundos...");
  
  for(int i = 3; i > 0; i--) {
    Serial.print(i);
    Serial.print("... ");
    delay(1000);
  }
  Serial.println("\nRecolectando muestras...");

  for (int i = 0; i < num_muestras_calibracion; i++) {
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
    
    ax_sum += ax_raw;
    ay_sum += ay_raw;
    az_sum += az_raw;
    gx_sum += gx_raw;
    gy_sum += gy_raw;
    gz_sum += gz_raw;

    if (i % (num_muestras_calibracion / 10) == 0) {
      Serial.print("Progreso: ");
      Serial.print((i * 100) / num_muestras_calibracion);
      Serial.println("%");
    }
    delay(2);
  }

  auto_ax_offset = ax_sum / num_muestras_calibracion;
  auto_ay_offset = ay_sum / num_muestras_calibracion;
  auto_az_offset = az_sum / num_muestras_calibracion;
  auto_gx_offset = gx_sum / num_muestras_calibracion;
  auto_gy_offset = gy_sum / num_muestras_calibracion;
  auto_gz_offset = gz_sum / num_muestras_calibracion;

  Serial.println("\n=== OFFSETS CALCULADOS ===");
  Serial.print("Acelerómetro X: "); Serial.println(auto_ax_offset);
  Serial.print("Acelerómetro Y: "); Serial.println(auto_ay_offset);
  Serial.print("Acelerómetro Z: "); Serial.println(auto_az_offset);
  Serial.print("Giroscopio X: "); Serial.println(auto_gx_offset);
  Serial.print("Giroscopio Y: "); Serial.println(auto_gy_offset);
  Serial.print("Giroscopio Z: "); Serial.println(auto_gz_offset);
  Serial.println("========================");
  
  Serial.println("\nCalibración completada. Iniciando en 2 segundos...");
  delay(2000);
}

// Función para imprimir header CSV
void printCSVHeader() {
  if (!csvHeaderPrinted && printCSV) {
    Serial.println("\n=== INICIO DE DATOS CSV ===");
    Serial.println("flex1,flex2,flex3,flex4,flex5,roll,pitch,angle_gx,angle_gy,angle_gz,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,timestamp");
    csvHeaderPrinted = true;
  }
}

// Función para imprimir datos en formato CSV
void printDataCSV(float flex_values[]) {
  if (!printCSV) return;
  
  // Imprimir header si no se ha hecho
  if (!csvHeaderPrinted) {
    printCSVHeader();
  }
  
  // Imprimir datos de sensores flex
  for (int i = 0; i < numFlexSensors; i++) {
    Serial.print(flex_values[i], 0);
    Serial.print(",");
  }
  
  // Imprimir datos MPU6050
  Serial.print(roll_deg, 2);
  Serial.print(",");
  Serial.print(pitch_deg, 2);
  Serial.print(",");
  Serial.print(angle_gx_deg, 2);
  Serial.print(",");
  Serial.print(angle_gy_deg, 2);
  Serial.print(",");
  Serial.print(angle_gz_deg, 2);
  Serial.print(",");
  Serial.print(accel_x_g, 3);
  Serial.print(",");
  Serial.print(accel_y_g, 3);
  Serial.print(",");
  Serial.print(accel_z_g, 3);
  Serial.print(",");
  Serial.print(gyro_x_dps, 2);
  Serial.print(",");
  Serial.print(gyro_y_dps, 2);
  Serial.print(",");
  Serial.print(gyro_z_dps, 2);
  Serial.print(",");
  Serial.println(millis());
}

void setup() {
  Wire.begin();
  Serial.begin(115200);
  delay(1000);

  Serial.println("Iniciando Smart Glove ESP32-S3 con WebSocket y salida CSV...");

  // Conectar WiFi
  connectWiFi();

  // Inicialización del MPU6050
  Serial.println("Iniciando MPU6050 en la direccion 0x69...");
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("FALLO en conexion MPU6050 en 0x69.");
    while (1);
  }
  Serial.println("Conexion MPU6050 en 0x69 exitosa.");

  Serial.println("Reiniciando MPU6050...");
  mpu.reset();
  delay(100);
  mpu.initialize();
  delay(100);

  Serial.println("Configurando rangos del MPU6050...");
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

  // Configuración Sensores Flexibles
  for (int i = 0; i < numFlexSensors; i++) {
    pinMode(flexPins[i], INPUT);
    analogReadResolution(12);
  }
  Serial.println("Sensores Flexibles configurados.");

  // Ejecutar Calibración Automática del MPU6050
  calcularOffsetsAutomaticosMPU6050();

  // Configurar WebSocket
  if (wifiConnected) {
    webSocket.begin(websocket_server, websocket_port, "/");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    Serial.println("WebSocket configurado.");
  }

  Serial.println("\nSetup completado. Iniciando lectura de datos...");
  Serial.println("Los datos se imprimirán en formato CSV y se enviarán por WebSocket.");
  prev_time_mpu = micros();
}

float readFlexSensor(int pin, int index) {
    int adc_flex = analogRead(pin);
    return adc_flex;
}

void filterMPUData() {
    accel_x_g = (alpha * accel_x_g) + ((1 - alpha) * last_ax);
    accel_y_g = (alpha * accel_y_g) + ((1 - alpha) * last_ay);
    accel_z_g = (alpha * accel_z_g) + ((1 - alpha) * last_az);
    
    gyro_x_dps = (alpha * gyro_x_dps) + ((1 - alpha) * last_gx);
    gyro_y_dps = (alpha * gyro_y_dps) + ((1 - alpha) * last_gy);
    gyro_z_dps = (alpha * gyro_z_dps) + ((1 - alpha) * last_gz);
    
    last_ax = accel_x_g;
    last_ay = accel_y_g;
    last_az = accel_z_g;
    last_gx = gyro_x_dps;
    last_gy = gyro_y_dps;
    last_gz = gyro_z_dps;
}

void sendDataToServer() {
  if (!websocketConnected) return;
  
  // Crear objeto JSON
  StaticJsonDocument<300> doc;
  
  // Datos de sensores flex
  JsonArray flexData = doc.createNestedArray("flex_sensors");
  for (int i = 0; i < numFlexSensors; i++) {
    float flexValue = readFlexSensor(flexPins[i], i);
    flexData.add(flexValue);
  }
  
  // Datos del MPU6050
  JsonObject mpu_data = doc.createNestedObject("mpu6050");
  mpu_data["roll"] = roll_deg;
  mpu_data["pitch"] = pitch_deg;
  mpu_data["angle_gx"] = angle_gx_deg;
  mpu_data["angle_gy"] = angle_gy_deg;
  mpu_data["angle_gz"] = angle_gz_deg;
  mpu_data["accel_x"] = accel_x_g;
  mpu_data["accel_y"] = accel_y_g;
  mpu_data["accel_z"] = accel_z_g;
  mpu_data["gyro_x"] = gyro_x_dps;
  mpu_data["gyro_y"] = gyro_y_dps;
  mpu_data["gyro_z"] = gyro_z_dps;
  
  // Timestamp
  doc["timestamp"] = millis();
  
  // Convertir a string y enviar
  String jsonString;
  serializeJson(doc, jsonString);
  webSocket.sendTXT(jsonString);
}

void loop() {
    static unsigned long lastMeasurement = 0;
    unsigned long currentMillis = millis();

    // Mantener conexión WebSocket
    webSocket.loop();
    
    // Verificar conexión WiFi
    if (WiFi.status() != WL_CONNECTED && wifiConnected) {
      Serial.println("WiFi desconectado. Intentando reconectar...");
      wifiConnected = false;
      websocketConnected = false;
      connectWiFi();
      if (wifiConnected) {
        webSocket.begin(websocket_server, websocket_port, "/");
      }
    }

    if (currentMillis - lastMeasurement >= 50) {  // 20Hz sampling rate
        lastMeasurement = currentMillis;

        // Array para almacenar valores de sensores flex
        float flex_values[numFlexSensors];
        
        // Leer sensores flex
        for (int i = 0; i < numFlexSensors; i++) {
            flex_values[i] = readFlexSensor(flexPins[i], i);
        }

        // Leer MPU6050
        mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

        // Calibración
        ax_cal = ax_raw - auto_ax_offset;
        ay_cal = ay_raw - auto_ay_offset;
        az_cal = az_raw - auto_az_offset;
        gx_cal = gx_raw - auto_gx_offset;
        gy_cal = gy_raw - auto_gy_offset;
        gz_cal = gz_raw - auto_gz_offset;

        // Convertir a unidades físicas
        accel_x_g = (float)ax_cal / 16384.0;
        accel_y_g = (float)ay_cal / 16384.0;
        accel_z_g = (float)az_cal / 16384.0;
        gyro_x_dps = (float)gx_cal / 131.0;
        gyro_y_dps = (float)gy_cal / 131.0;
        gyro_z_dps = (float)gz_cal / 131.0;

        // Aplicar filtros
        filterMPUData();

        // Calcular ángulos
        roll_deg = atan2(ay_cal, az_cal) * 180.0 / PI;
        float pitch_arg = (float)-ax_cal / sqrt((float)ay_cal * ay_cal + (float)az_cal * az_cal);
        pitch_arg = constrain(pitch_arg, -1.0, 1.0);
        pitch_deg = asin(pitch_arg) * 180.0 / PI;

        // Integrar datos del giroscopio
        unsigned long current_time_mpu = micros();
        dt_mpu = (float)(current_time_mpu - prev_time_mpu) / 1000000.0;
        prev_time_mpu = current_time_mpu;

        angle_gx_deg += gyro_x_dps * dt_mpu;
        angle_gy_deg += gyro_y_dps * dt_mpu;
        angle_gz_deg += gyro_z_dps * dt_mpu;

        // Imprimir datos en formato CSV
        printDataCSV(flex_values);
        
        // Enviar datos al servidor
        sendDataToServer();
    }
} 