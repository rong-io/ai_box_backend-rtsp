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
    # 預設值包含單筆測試資料
    parser.add_argument(
        "--json", 
        type=str, 
        default='{"coordinate": "[{\\"top_left_x\\": 897.07, \\"top_left_y\\": 368.89, \\"top_right_x\\": 1039.91, \\"top_right_y\\": 368.89, \\"bottom_right_x\\": 1039.91, \\"bottom_right_y\\": 588.74, \\"bottom_left_x\\": 897.07, \\"bottom_left_y\\": 588.74}]"}',
        help="JSON message to send (default: single test data)"
    )
    args = parser.parse_args()

    try:
        # 檢查 JSON 格式
        message = json.dumps(json.loads(args.json))
        asyncio.run(send_message(args.url, message))
    except json.JSONDecodeError:
        print("Invalid JSON format. Please provide a valid JSON string.")
