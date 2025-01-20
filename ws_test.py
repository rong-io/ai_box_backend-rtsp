import asyncio
import websockets

async def view_fps_data():
    uri = "ws://0.0.0.0:7860/ws/fps"
    async with websockets.connect(uri) as websocket:
        print("已連線到 /ws/fps")
        while True:
            message = await websocket.recv()
            print("收到資料:", message)

asyncio.run(view_fps_data())
