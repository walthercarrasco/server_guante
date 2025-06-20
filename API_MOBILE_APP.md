# API WebSocket para Aplicación Móvil

## 🔌 Conexión al Servidor

```javascript
const ws = new WebSocket('ws://192.168.1.19:8080');
```

## 📱 Protocolo de Comunicación

### 1. **Identificación del Cliente** (Primer mensaje)

La app móvil debe enviar como primer mensaje:

```json
{
    "client_type": "mobile"
}
```

O simplemente conectarse y el servidor la identificará automáticamente si no envía datos de sensores.

### 2. **Recepción de Gestos**

Cuando se detecta un gesto, la app recibirá:

```json
{
    "type": "gesture_detected",
    "gesture": "HOLA",
    "confidence": 0.892,
    "timestamp": 1234567890.123
}
```

### 3. **Solicitar Último Gesto**

Para obtener el último gesto detectado:

**Enviar:**
```json
{
    "action": "get_last_gesture"
}
```

**Recibir:** (si hay un gesto reciente)
```json
{
    "type": "gesture_detected",
    "gesture": "BIEN",
    "confidence": 0.945,
    "timestamp": 1234567890.123
}
```

### 4. **Keep-Alive** (Opcional)

Para mantener la conexión activa:

**Enviar:**
```json
{
    "type": "ping"
}
```

**Recibir:**
```json
{
    "type": "pong",
    "timestamp": 1234567890.123
}
```

## 💻 Ejemplo de Implementación

### JavaScript/React Native:

```javascript
class GestureClient {
    constructor(serverUrl) {
        this.ws = new WebSocket(serverUrl);
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.ws.onopen = () => {
            console.log('Conectado al servidor');
            // Identificarse como cliente móvil
            this.ws.send(JSON.stringify({
                client_type: "mobile"
            }));
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'gesture_detected') {
                this.onGestureDetected(data.gesture, data.confidence);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('Error WebSocket:', error);
        };
        
        this.ws.onclose = () => {
            console.log('Desconectado del servidor');
            // Implementar reconexión si es necesario
        };
    }
    
    onGestureDetected(gesture, confidence) {
        // Actualizar UI con el gesto detectado
        console.log(`Gesto: ${gesture} (${(confidence * 100).toFixed(1)}%)`);
    }
    
    requestLastGesture() {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                action: "get_last_gesture"
            }));
        }
    }
    
    sendPing() {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: "ping"
            }));
        }
    }
}

// Uso
const client = new GestureClient('ws://192.168.1.19:8080');
```

### Flutter/Dart:

```dart
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';

class GestureClient {
  late WebSocketChannel channel;
  
  void connect() {
    channel = WebSocketChannel.connect(
      Uri.parse('ws://192.168.1.19:8080'),
    );
    
    // Identificarse como móvil
    channel.sink.add(jsonEncode({
      'client_type': 'mobile'
    }));
    
    // Escuchar mensajes
    channel.stream.listen((message) {
      final data = jsonDecode(message);
      
      if (data['type'] == 'gesture_detected') {
        onGestureDetected(
          data['gesture'], 
          data['confidence']
        );
      }
    });
  }
  
  void onGestureDetected(String gesture, double confidence) {
    // Actualizar UI
    print('Gesto: $gesture (${(confidence * 100).toStringAsFixed(1)}%)');
  }
  
  void requestLastGesture() {
    channel.sink.add(jsonEncode({
      'action': 'get_last_gesture'
    }));
  }
  
  void dispose() {
    channel.sink.close();
  }
}
```

## 🎯 Gestos Disponibles

- `BIEN` - Pulgar arriba
- `SI` - Afirmación
- `HOLA` - Saludo
- `ADIOS` - Despedida

## 📊 Información del Servidor

- **Puerto**: 8080
- **Protocolo**: WebSocket (ws://)
- **Formato**: JSON
- **Reconexión**: Automática recomendada

## 🔄 Flujo de Datos

1. ESP32 envía datos de sensores al servidor
2. Servidor analiza y detecta gestos con ML
3. Servidor envía gestos detectados a todas las apps móviles conectadas
4. Apps móviles actualizan su UI en tiempo real

## ⚡ Notas Importantes

- Los gestos solo se envían cuando la confianza es ≥ 0.7
- El servidor mantiene el último gesto detectado para nuevos clientes
- Múltiples apps pueden conectarse simultáneamente
- La latencia típica es < 100ms desde la detección hasta la recepción 