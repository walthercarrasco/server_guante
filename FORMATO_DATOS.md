# Formato Correcto de Datos - Smart Glove

## 📊 Estructura JSON Esperada

```json
{
    "flex_sensors": [2048, 1890, 2100, 1950, 2200],  // Valores ADC (0-4095)
    "mpu6050": {
        "roll": -12.5,      // Grados
        "pitch": 5.3,       // Grados
        "angle_gx": 0.8,    // Grados
        "angle_gy": -1.2,   // Grados
        "angle_gz": 2.1,    // Grados
        "accel_x": 0.02,    // g
        "accel_y": -0.98,   // g
        "accel_z": 0.15,    // g
        "gyro_x": 0.5,      // °/s
        "gyro_y": -0.3,     // °/s
        "gyro_z": 0.2       // °/s
    },
    "timestamp": 123456789
}
```

## 🎯 Rangos de Valores Esperados

| Sensor | Rango | Unidad |
|--------|-------|--------|
| flex_sensors[0-4] | 0 - 4095 | ADC (12-bit) |
| roll | -180 a 180 | grados |
| pitch | -90 a 90 | grados |
| angle_gx/gy/gz | -180 a 180 | grados |
| accel_x/y/z | -2 a 2 | g |
| gyro_x/y/z | -250 a 250 | °/s |

## ✅ Formato Correcto

El modelo fue entrenado con valores ADC directos del ESP32:

```json
"flex_sensors": [2048, 1890, 2100, 1950, 2200]  // ✓ Valores ADC correctos
```

## 📝 Notas Importantes

- Los sensores flex envían valores ADC de 12 bits (0-4095)
- NO es necesario convertir a ángulos
- El modelo aprende la relación entre valores ADC y gestos directamente
- Valores típicos:
  - Sensor recto: ~2048 (mitad del rango)
  - Sensor doblado: valores más altos o bajos dependiendo del circuito 