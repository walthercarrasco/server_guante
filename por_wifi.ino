#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// Configuración WiFi
const char* ssid = "WiFi Carrasco";
const char* password = "W1a6l2t0h0e4r";

// Configuración WebSocket
const char* websocket_server = "192.168.31.222";
const int websocket_port = 8080;

// Hardware
WebSocketsClient webSocket;
MPU6050 mpu(0x69);

// Pines sensores flex
const int flexPins[] = {4, 5, 6, 7, 15};
const int numFlexSensors = 5;

// Offsets calibración MPU
int16_t ax_offset = 0, ay_offset = 0, az_offset = 0;
int16_t gx_offset = 0, gy_offset = 0, gz_offset = 0;

// Variables MPU
float roll = 0, pitch = 0;
float angleX = 0, angleY = 0, angleZ = 0;
unsigned long lastTime = 0;

// Estado conexión
bool wsConnected = false;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Configurar ADC
  analogReadResolution(12);
  for (int i = 0; i < numFlexSensors; i++) {
    pinMode(flexPins[i], INPUT);
  }
  
  // Inicializar MPU6050
  mpu.initialize();
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  
  // Conectar WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado");
  
  // Configurar WebSocket
  webSocket.begin(websocket_server, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  
  // Calibrar MPU
  calibrateMPU();
  
  lastTime = micros();
}

void loop() {
  static unsigned long lastSend = 0;
  unsigned long now = millis();
  
  webSocket.loop();
  
  // Enviar datos cada 50ms (20Hz)
  if (now - lastSend >= 50) {
    lastSend = now;
    
    // Leer sensores flex
    int flexValues[numFlexSensors];
    for (int i = 0; i < numFlexSensors; i++) {
      flexValues[i] = analogRead(flexPins[i]);
    }
    
    // Leer MPU6050
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Aplicar calibración
    ax -= ax_offset;
    ay -= ay_offset;
    az -= az_offset;
    gx -= gx_offset;
    gy -= gy_offset;
    gz -= gz_offset;
    
    // Calcular ángulos
    roll = atan2(ay, az) * 180.0 / PI;
    pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    
    // Integrar giroscopio
    float dt = (micros() - lastTime) / 1000000.0;
    lastTime = micros();
    angleX += (gx / 131.0) * dt;
    angleY += (gy / 131.0) * dt;
    angleZ += (gz / 131.0) * dt;
    
    // Enviar datos
    if (wsConnected) {
      sendData(flexValues, ax, ay, az, gx, gy, gz);
    }
    
    // Imprimir CSV
    printCSV(flexValues, ax, ay, az, gx, gy, gz);
  }
}

void calibrateMPU() {
  long axSum = 0, aySum = 0, azSum = 0;
  long gxSum = 0, gySum = 0, gzSum = 0;
  const int samples = 1000;
  
  Serial.println("Calibrando MPU6050...");
  
  for (int i = 0; i < samples; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    axSum += ax;
    aySum += ay;
    azSum += az;
    gxSum += gx;
    gySum += gy;
    gzSum += gz;
    
    delay(2);
  }
  
  ax_offset = axSum / samples;
  ay_offset = aySum / samples;
  az_offset = azSum / samples;
  gx_offset = gxSum / samples;
  gy_offset = gySum / samples;
  gz_offset = gzSum / samples;
  
  Serial.println("Calibración completada");
}

void sendData(int flex[], int16_t ax, int16_t ay, int16_t az, int16_t gx, int16_t gy, int16_t gz) {
  StaticJsonDocument<256> doc;
  
  JsonArray flexArray = doc.createNestedArray("flex");
  for (int i = 0; i < numFlexSensors; i++) {
    flexArray.add(flex[i]);
  }
  
  JsonObject mpuData = doc.createNestedObject("mpu");
  mpuData["roll"] = roll;
  mpuData["pitch"] = pitch;
  mpuData["angleX"] = angleX;
  mpuData["angleY"] = angleY;
  mpuData["angleZ"] = angleZ;
  mpuData["ax"] = ax / 16384.0;
  mpuData["ay"] = ay / 16384.0;
  mpuData["az"] = az / 16384.0;
  mpuData["gx"] = gx / 131.0;
  mpuData["gy"] = gy / 131.0;
  mpuData["gz"] = gz / 131.0;
  
  doc["time"] = millis();
  
  String json;
  serializeJson(doc, json);
  webSocket.sendTXT(json);
}

void printCSV(int flex[], int16_t ax, int16_t ay, int16_t az, int16_t gx, int16_t gy, int16_t gz) {
  for (int i = 0; i < numFlexSensors; i++) {
    Serial.print(flex[i]);
    Serial.print(",");
  }
  Serial.print(roll, 2);
  Serial.print(",");
  Serial.print(pitch, 2);
  Serial.print(",");
  Serial.print(angleX, 2);
  Serial.print(",");
  Serial.print(angleY, 2);
  Serial.print(",");
  Serial.print(angleZ, 2);
  Serial.print(",");
  Serial.print(ax / 16384.0, 3);
  Serial.print(",");
  Serial.print(ay / 16384.0, 3);
  Serial.print(",");
  Serial.print(az / 16384.0, 3);
  Serial.print(",");
  Serial.print(gx / 131.0, 2);
  Serial.print(",");
  Serial.print(gy / 131.0, 2);
  Serial.print(",");
  Serial.print(gz / 131.0, 2);
  Serial.print(",");
  Serial.println(millis());
}

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch(type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.println("WebSocket conectado");
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("WebSocket desconectado");
      break;
  }
} 