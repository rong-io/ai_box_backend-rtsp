import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
from logging.handlers import TimedRotatingFileHandler
import weakref
import cv2
import time
import PIL.Image
import matplotlib.pyplot as plt
from typing import List
import json

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

# LOG FILE SETTING
LOG_FILE = "app_log.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

file_handler = TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
file_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.addHandler(file_handler)

def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors

def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)

def process_coordinates_and_draw(image, coordinates, detections, tree):
    for coord in coordinates:
        try:
            top_left_x = int(coord["top_left_x"])
            top_left_y = int(coord["top_left_y"])
            bottom_left_x = int(coord["bottom_left_x"])
            bottom_left_y = int(coord["bottom_left_y"])

            # 畫出裁切範圍(測試用，整合後註解)
            cv2.rectangle(
                image,
                (top_left_x, top_left_y),
                (bottom_left_x, bottom_left_y),
                (0, 255, 0),
                2
            )

            cropped_image = image[top_left_y:bottom_left_y, top_left_x:bottom_left_x]

            cropped_pil = cv2_to_pil(cropped_image)
            cropped_detections = predictor.predict(
                image=cropped_pil,
                tree=tree,
                clip_text_encodings=prompt_data["clip_encodings"],
                owl_text_encodings=prompt_data["owl_encodings"]
            )

            image = draw_tree_output(image, cropped_detections, tree)
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")

    return image

async def handle_index_get(request: web.Request):
    logging.info("handle_index_get")
    return web.FileResponse("./index.html")

async def websocket_handler(request):
    global prompt_data, coordinates_data

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info("Websocket connected.")
    request.app['websockets'].add(ws)

    try:
        async for msg in ws:
            logging.info(f"Received message from websocket: {msg.data}")
            # ws json object prompt 範例: {"json": "[{\"object\": \"apple\", \"threshold\": \"0.5\"}, {\"object\": \"banana\", \"threshold\": \"0.5\"}]"}
            # ws json 座標範例：{"coordinate": "[{\"top_left_x\": \"200\",\"top_left_y\": \"250\", \"bottom_left_x\": \"500\", \"bottom_left_y\": \"550\"}]"}
            if "json" in msg.data:
                try:
                    data = json.loads(msg.data)['json']
                    prompt = f"[{', '.join([item['object'] for item in data])}]"
                    logging.info(f"Converted prompt: {prompt}")

                    tree = Tree.from_prompt(prompt)
                    clip_encodings = predictor.encode_clip_text(tree)
                    owl_encodings = predictor.encode_owl_text(tree)

                    prompt_data = {
                        "tree": tree,
                        "clip_encodings": clip_encodings,
                        "owl_encodings": owl_encodings
                    }
                    logging.info(f"Set prompt_data successfully: {prompt_data}")
                except Exception as e:
                    logging.error(f"Error generating prompt data: {e}")

            elif "coordinate" in msg.data:
                try:
                    coordinate_data = json.loads(msg.data)["coordinate"]
                    coordinates_data = json.loads(coordinate_data)
                    logging.info(f"Coordinates updated: {coordinates_data}")
                except Exception as e:
                    logging.error(f"Error parsing coordinates: {e}")
    finally:
        request.app['websockets'].discard(ws)

    return ws

async def send_warning_to_clients(warning_data):
    warning_json = json.dumps(warning_data)
    for ws in app["websockets"]:
        try:
            await ws.send_str(warning_json)
        except Exception as e:
            logging.error(f"Failed to send warning to client: {e}")

async def on_shutdown(app: web.Application):
    for ws in set(app['websockets']):
        await ws.close(code=WSCloseCode.GOING_AWAY, message='Server shutdown')

async def detection_loop(app: web.Application):
    loop = asyncio.get_running_loop()
    logging.info("Opening camera.")
    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    logging.info("Loading predictor.")

    def _read_and_encode_image():
        global coordinates_data

        re, image = camera.read()
        logging.info(f"Camera read result: {re}")

        if not re:
            warning_msg = "Failed to capture image from camera."
            logging.warning(warning_msg)

            warning_data = {
                "type": "warning",
                "message": warning_msg
            }

            asyncio.run(send_warning_to_clients(warning_data))
            return re, None

        image_pil = cv2_to_pil(image)

        if prompt_data is not None:
            try:
                prompt_data_local = prompt_data
                logging.info(f"Using prompt_data: {prompt_data_local}")

                if coordinates_data:
                    logging.info(f"Processing coordinates: {coordinates_data}")
                    image = process_coordinates_and_draw(
                        image=image,
                        coordinates=coordinates_data,
                        detections=None,
                        tree=prompt_data_local["tree"]
                    )
                coordinates_data = None

                t0 = time.perf_counter_ns()
                detections = predictor.predict(
                    image=image_pil,
                    tree=prompt_data_local['tree'],
                    clip_text_encodings=prompt_data_local['clip_encodings'],
                    owl_text_encodings=prompt_data_local['owl_encodings']
                )
                t1 = time.perf_counter_ns()
                dt = (t1 - t0) / 1e9

                logging.info(f"Prediction completed in {dt:.3f} seconds.")
                # logging.info(f"Raw detections: {detections}")

                if not detections:
                    logging.warning("No detections made.")
                else:
                    logging.info(f"Detections: {detections}")

                image = draw_tree_output(image, detections, prompt_data_local['tree'])
                logging.info("Image updated with detections.")
            except Exception as e:
                logging.error(f"Prediction error: {e}")

        image_jpeg = bytes(
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
        )

        return re, image_jpeg

    while True:
        re, image = await loop.run_in_executor(None, _read_and_encode_image)

        if not re:
            break

        for ws in app["websockets"]:
            await ws.send_bytes(image)

    camera.release()

async def run_detection_loop(app):
    try:
        task = asyncio.create_task(detection_loop(app))
        yield
        task.cancel()
    except asyncio.CancelledError:
        pass
    finally:
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    CAMERA_DEVICE = args.camera
    IMAGE_QUALITY = args.image_quality

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )

    prompt_data = None

    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)
    web.run_app(app, host=args.host, port=args.port)
