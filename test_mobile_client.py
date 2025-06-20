#!/usr/bin/env python3
"""
Cliente de prueba para simular una aplicación móvil
conectándose al servidor Smart Glove
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

async def mobile_client():
    uri = "ws://localhost:8080"  # Cambiar a la IP del servidor si es necesario
    
    print(f"🔌 Conectando a {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Conectado al servidor")
            
            # Identificarse como cliente móvil
            await websocket.send(json.dumps({
                "client_type": "mobile"
            }))
            print("📱 Identificado como cliente móvil")
            
            # Solicitar último gesto si existe
            await websocket.send(json.dumps({
                "action": "get_last_gesture"
            }))
            print("🔍 Solicitando último gesto...")
            
            # Escuchar mensajes del servidor
            print("\n👂 Esperando gestos detectados...\n")
            
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "gesture_detected":
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        gesture = data.get("gesture", "UNKNOWN")
                        confidence = data.get("confidence", 0.0)
                        
                        print(f"[{timestamp}] 🎯 GESTO RECIBIDO: {gesture} (Confianza: {confidence:.1%})")
                        
                        # Simular actualización de UI
                        print_gesture_display(gesture, confidence)
                        
                    elif data.get("type") == "pong":
                        print("♥️ Pong recibido")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\n❌ Conexión cerrada por el servidor")
                    break
                except json.JSONDecodeError:
                    print("⚠️ Mensaje no válido recibido")
                    
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ Error de conexión: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def print_gesture_display(gesture, confidence):
    """Simula una visualización del gesto en la app"""
    gesture_emojis = {
        "HOLA": "👋",
        "BIEN": "👍",
        "SI": "✓",
        "ADIOS": "👋"
    }
    
    emoji = gesture_emojis.get(gesture, "❓")
    bar_length = int(confidence * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)
    
    print(f"""
    ┌─────────────────────────┐
    │  {emoji}  {gesture:<10}      │
    │  [{bar}] {confidence:.0%} │
    └─────────────────────────┘
    """)

async def send_periodic_ping(websocket):
    """Envía ping periódicamente para mantener la conexión activa"""
    while True:
        await asyncio.sleep(30)  # Cada 30 segundos
        try:
            await websocket.send(json.dumps({"type": "ping"}))
            print("📡 Ping enviado")
        except:
            break

def main():
    print("=" * 50)
    print("CLIENTE DE PRUEBA - APLICACIÓN MÓVIL")
    print("Simulador de conexión para Smart Glove Server")
    print("=" * 50)
    
    try:
        asyncio.run(mobile_client())
    except KeyboardInterrupt:
        print("\n\n👋 Cliente detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 