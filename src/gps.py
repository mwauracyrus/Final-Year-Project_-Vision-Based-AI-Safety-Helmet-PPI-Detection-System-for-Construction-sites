import socket
import time

ESP_IP = "10.114.86.39"  
PORT = 80

def start_receiver():
    print(f"🔄 Connecting to ESP8266 at {ESP_IP}:{PORT}...")
    
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Give it a slightly longer timeout to account for the 1-second transmission interval
            s.settimeout(10) 
            s.connect((ESP_IP, PORT))
            print("✅ Wireless connection established! Streaming live GPS data...\n")
            
            # Create a file-like object from the socket to read line-by-line cleanly
            buffer = s.makefile('r', encoding='utf-8')
            
            while True:
                # Read a full line sent by client.println()
                line = buffer.readline()
                
                if not line:
                    print("❌ Server closed connection. Reconnecting...")
                    break
                
                data = line.strip()
                if not data:
                    continue
                
                # Split the CSV string
                parts = data.split(',')
                if len(parts) == 4:
                    lat, lng, sats, speed = parts
                    print(f"📡 [Wi-Fi Data] Sats: {sats} | Lat: {lat} | Lng: {lng} | Speed: {speed} km/h")
                else:
                    print(f"Raw Data received: {data}")
                    
        except (socket.error, socket.timeout) as e:
            print(f"⚠️ Connection glitch ({e}). Retrying in 2 seconds...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n👋 Stopping receiver.")
            break
        finally:
            s.close()

if __name__ == "__main__":
    start_receiver()