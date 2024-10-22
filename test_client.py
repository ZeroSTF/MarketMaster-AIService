import requests
import time
import websocket
import json
import threading

BASE_URL = "http://localhost:5000"

def test_api_endpoints():
    # Test asset registration
    print("\nTesting asset registration...")
    response = requests.post(f"{BASE_URL}/api/assets/register", 
                           json={"symbols": ["AAPL", "GOOGL", "MSFT"]})
    print(f"Register response: {json.dumps(response.json(), indent=2)}")

    # Wait for data to be fetched
    time.sleep(2)

    # Test getting all assets
    print("\nTesting get all assets...")
    response = requests.get(f"{BASE_URL}/api/assets/data")
    print(f"All assets response: {json.dumps(response.json(), indent=2)}")

    # Test getting single asset
    print("\nTesting get single asset...")
    response = requests.get(f"{BASE_URL}/api/assets/AAPL")
    print(f"Single asset response: {json.dumps(response.json(), indent=2)}")

def on_message(ws, message):
    print(f"\nReceived WebSocket update: {message}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")

def test_websocket():
    ws = websocket.WebSocketApp("ws://localhost:5000/socket.io/?EIO=4&transport=websocket",
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close,
                              on_open=on_open)
    
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

def main():
    print("Starting tests...")
    
    # Start WebSocket connection
    test_websocket()
    
    # Wait for WebSocket to connect
    time.sleep(1)
    
    # Test REST API endpoints
    test_api_endpoints()
    
    # Keep running to receive WebSocket updates
    print("\nListening for real-time updates (press Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTests completed.")

if __name__ == "__main__":
    main()