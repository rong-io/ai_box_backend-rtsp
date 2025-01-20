import asyncio
import websockets
import argparse
import json

async def send_message(uri, message):
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
        print(f"Sent: {message}")
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client")
    parser.add_argument(
        "--url", 
        type=str, 
        default="ws://0.0.0.0:7860/ws", 
        help="WebSocket server URL (default: ws://0.0.0.0:7860/ws)"
    )
    # 預設值包含 schedule_stream 測試資料
    parser.add_argument(
        "--json", 
        type=str, 
        default=json.dumps({
            "action": "start",
            "stream_id": "camera_1",
            "rtsp_url": "rtsp://35.185.165.215:31554/mystream1",
            "start_time": "",
            "stop_time": ""
        }),
        help="JSON message to send (default: schedule_stream test data)"
    )
    args = parser.parse_args()

    try:
        # 檢查 JSON 格式
        message = json.dumps(json.loads(args.json))
        asyncio.run(send_message(args.url, message))
    except json.JSONDecodeError:
        print("Invalid JSON format. Please provide a valid JSON string.")
