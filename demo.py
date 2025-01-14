import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
from logging.handlers import TimedRotatingFileHandler
import weakref
import cv2
import time
import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot as plt
from typing import List
import json
import numpy as np
import ffmpeg
import threading
from asyncio import Queue

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor, TreeOutput

# from nanoowl.tree_drawing import draw_tree_output
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

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
stream_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


# Queue for audio data
audio_queue = Queue()


# RTSP audio
def extract_audio(rtsp_url, audio_queue):
    process = (
        ffmpeg.input(rtsp_url)
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=2, ar="44100")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    while True:
        audio_data = process.stdout.read(4096)
        if not audio_data:
            break
        audio_queue.put(audio_data)
    process.stdout.close()
    process.wait()


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw_tree_output(
    image, output: TreeOutput, tree: Tree, draw_text=True, num_colors=8
):
    detections = output.detections
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.asarray(image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    colors = get_colors(num_colors)
    label_map = tree.get_label_map()
    label_depths = tree.get_label_depth_map()

    for detection in detections:
        box = [int(x) for x in detection.box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])

        # stop drawing "image" box
        skip_detection = any(
            label_map[label].lower() == "image" for label in detection.labels
        )
        if skip_detection:
            continue

        box_depth = min(label_depths[i] for i in detection.labels)
        cv2.rectangle(image, pt0, pt1, colors[box_depth % num_colors], 3)
        if draw_text:
            offset_y = 30
            offset_x = 8
            for label in detection.labels:
                label_text = label_map[label]
                cv2.putText(
                    image,
                    label_text,
                    (box[0] + offset_x, box[1] + offset_y),
                    font,
                    font_scale,
                    colors[label % num_colors],
                    2,  # thickness
                    cv2.LINE_AA,
                )
                offset_y += 18
    if is_pil:
        image = PIL.Image.fromarray(image)
    return image


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
                2,
            )

            cropped_image = image[top_left_y:bottom_left_y, top_left_x:bottom_left_x]

            cropped_pil = cv2_to_pil(cropped_image)
            cropped_detections = predictor.predict(
                image=cropped_pil,
                tree=tree,
                clip_text_encodings=prompt_data["clip_encodings"],
                owl_text_encodings=prompt_data["owl_encodings"],
            )

            image = draw_tree_output(image, cropped_detections, tree)
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")

    return image


def process_coordinates_and_recognize(image, coordinates, tree):
    """
    根據座標裁切影像並輸入模型進行辨識
    """
    results = []
    for coord in coordinates:
        try:
            top_left_x = int(coord["top_left_x"])
            top_left_y = int(coord["top_left_y"])
            bottom_left_x = int(coord["bottom_left_x"])
            bottom_left_y = int(coord["bottom_left_y"])

            cropped_image = image[top_left_y:bottom_left_y, top_left_x:bottom_left_x]
            cropped_pil = cv2_to_pil(cropped_image)

            cropped_detections = predictor.predict(
                image=cropped_pil,
                tree=tree,
                clip_text_encodings=prompt_data["clip_encodings"],
                owl_text_encodings=prompt_data["owl_encodings"],
            )
            results.append(cropped_detections)
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")

    return results


def process_coordinates_and_recognize_with_overlay(image, coordinates, tree):
    """
    根據座標裁切影像，輸入模型進行辨識，並將結果疊加回原圖。
    """
    for coord in coordinates:
        try:
            top_left_x = int(round(coord["top_left_x"]))
            top_left_y = int(round(coord["top_left_y"]))
            bottom_right_x = int(round(coord["bottom_left_x"]))
            bottom_right_y = int(round(coord["bottom_left_y"]))

            # 確認座標範圍有效性
            cv2.rectangle(
                image,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (0, 255, 0),  # 綠色框
                3,
            )

            logging.info(
                f"Drawing rectangle: Top-Left ({top_left_x}, {top_left_y}), "
                f"Bottom-Right ({bottom_right_x}, {bottom_right_y})"
            )

            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            cropped_pil = cv2_to_pil(cropped_image)

            cropped_detections = predictor.predict(
                image=cropped_pil,
                tree=tree,
                clip_text_encodings=prompt_data["clip_encodings"],
                owl_text_encodings=prompt_data["owl_encodings"],
            )

            if cropped_detections:
                logging.info(f"Cropped detections: {cropped_detections}")
                image = draw_tree_output(image, cropped_detections, tree)
        except Exception as e:
            logging.error(f"Error processing coordinates: {e}")

    return image


# WebSocket handler for audio transmission
async def audio_websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info("Audio WebSocket connected.")

    try:
        while True:
            audio_data = await asyncio.get_running_loop().run_in_executor(
                None, audio_queue.get
            )
            await ws.send_bytes(audio_data)
    finally:
        logging.info("Audio WebSocket disconnected.")
        await ws.close()

    return ws


async def websocket_handler(request):
    global prompt_data, coordinates_data

    stream_handler = request.app["streams"][request.match_info["stream_id"]]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"WebSocket connected for stream: {request.match_info['stream_id']}")
    request.app["websockets"].add(ws)

    try:
        async for msg in ws:
            # ws json object prompt 範例: {"json": "[{\"object\": \"apple\", \"threshold\": \"0.5\"}, {\"object\": \"banana\", \"threshold\": \"0.5\"}]"}
            # ws json 四點位座標範例：{"coordinate": "[{\"top_left_x\": \"200\",\"top_left_y\": \"250\",\"top_right_y\": \"250\",\"top_right_y\": \"270\", \"bottom_left_x\": \"500\", \"bottom_left_y\": \"550\", \"bottom_right_y\": \"550\", \"bottom_right_y\": \"300\"}]"}
            logging.info(f"Received message from websocket: {msg.data}")
            if "json" in msg.data:
                try:
                    data = json.loads(msg.data)["json"]
                    prompt = f"[{', '.join([item['object'] for item in data])}]"
                    threshold = {item["object"]: float(item["threshold"]) for item in data}
                    logging.info(f"Converted prompt: {prompt}")

                    tree = Tree.from_prompt(prompt)
                    clip_encodings = predictor.encode_clip_text(tree)
                    owl_encodings = predictor.encode_owl_text(tree)

                    prompt_data = {
                        "tree": tree,
                        "clip_encodings": clip_encodings,
                        "owl_encodings": owl_encodings,
                        "threshold": threshold,
                    }
                except Exception as e:
                    logging.error(f"Error generating prompt data: {e}")

            elif "coordinate" in msg.data:
                try:
                    data = json.loads(msg.data)
                    coordinate_data = data["coordinate"]
                    coordinates_data = [coordinate_data]
                    logging.info(f"Coordinates updated: {coordinates_data}")
                except Exception as e:
                    logging.error(f"Error parsing coordinates: {e}")
    finally:
        request.app["websockets"].discard(ws)

    return ws


async def send_warning_to_clients(warning_data):
    warning_json = json.dumps(warning_data)
    for ws in app["websockets"]:
        try:
            await ws.send_str(warning_json)
        except Exception as e:
            logging.error(f"Failed to send warning to client: {e}")


async def handle_index_get(request: web.Request):
    logging.info("handle_index_get")
    return web.FileResponse("./index.html")


async def on_shutdown(app: web.Application):
    for ws in set(app["websockets"]):
        await ws.close(code=WSCloseCode.GOING_AWAY, message="Server shutdown")


async def detection_loop(app: web.Application):
    global camera
    loop = asyncio.get_running_loop()

    logging.info("Opening camera.")
    camera = cv2.VideoCapture(RTSP_URL)

    if not camera.isOpened():
        logging.error("Failed to open RTSP stream.")
        return

    logging.info("Loading predictor.")

    def _read_and_encode_image():
        global coordinates_data
        global camera

        re, image = camera.read()
        logging.info(f"RTSP stream read result: {re}")

        if not re:
            warning_msg = "Failed to capture frame from RTSP stream."
            logging.warning(warning_msg)
            warning_data = {"type": "warning", "message": warning_msg}
            asyncio.run(send_warning_to_clients(warning_data))

            # 超時重連
            camera.release()
            camera = cv2.VideoCapture(RTSP_URL)
            if not camera.isOpened():
                logging.error("Reconnection attempt failed.")
                return re, None

            return re, None

        # Detection phase
        image_pil = cv2_to_pil(image)

        if prompt_data is not None:
            try:
                prompt_data_local = prompt_data
                # logging.info(f"Using prompt_data: {prompt_data_local}")

                if coordinates_data:
                    logging.info(f"Processing coordinates: {coordinates_data}")
                    image = process_coordinates_and_recognize_with_overlay(
                        image=image,
                        coordinates=coordinates_data,
                        tree=prompt_data["tree"],
                    )
                    logging.info("Overlay results on original image.")

                else:
                    logging.info(f"Performing full image detection with prompt_data.")
                    t0 = time.perf_counter_ns()
                    detections = predictor.predict(
                        image=image_pil,
                        tree=prompt_data_local["tree"],
                        clip_text_encodings=prompt_data_local["clip_encodings"],
                        owl_text_encodings=prompt_data_local["owl_encodings"],
                        threshold=prompt_data_local["thresholds"],
                    )
                    t1 = time.perf_counter_ns()
                    dt = (t1 - t0) / 1e9

                    logging.info(f"Prediction completed in {dt:.3f} seconds.")
                    logging.info(f"Raw detections: {detections}")

                if not detections:
                    logging.warning("No detections made.")
                else:
                    logging.info(f"Detections: {detections}")

                image = draw_tree_output(image, detections, prompt_data_local["tree"])
                logging.info("Image updated with detections.")
            except Exception as e:
                logging.error(f"Prediction error: {e}")

        image_jpeg = bytes(
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])[1]
        )

        return re, image_jpeg

    try:
        while True:
            re, image = await loop.run_in_executor(None, _read_and_encode_image)

            if not re:
                continue

            for ws in app["websockets"]:
                await ws.send_bytes(image)
    finally:
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


class RTSPStreamHandler:
    def __init__(self, rtsp_url, queue_size=10):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=queue_size)
        self.camera = None
        self.running = False

    def start(self):
        self.running = True
        self.camera = cv2.VideoCapture(self.rtsp_url)
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()

    def _capture_loop(self):
        while self.running:
            if not self.camera.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                self.camera = cv2.VideoCapture(self.rtsp_url)
                continue

            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame)
            else:
                logger.warning(
                    f"Failed to read frame from RTSP stream: {self.rtsp_url}"
                )

    async def get_frame(self):
        return await self.frame_queue.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--rtsp_url",
        type=str,
        help="RTSP stream URL",
        default="rtsp://35.185.165.215:31554/mystream1",
    )
    args = parser.parse_args()

    CAMERA_DEVICE = args.camera
    RTSP_URL = args.rtsp_url
    IMAGE_QUALITY = args.image_quality

    # Start audio extraction thread
    audio_thread = threading.Thread(
        target=extract_audio, args=(RTSP_URL, audio_queue), daemon=True
    )
    audio_thread.start()

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(image_encoder_engine=args.image_encode_engine)
    )

    prompt_data = None
    coordinates_data = None

    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app["streams"] = {}
    app["websockets"] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.router.add_route("GET", "/audio", audio_websocket_handler)

    # Initialize RTSP handlers
    for idx, rtsp_url in enumerate(args.rtsp_urls):
        stream_id = f"stream{idx+1}"
        stream_handler = RTSPStreamHandler(rtsp_url)
        stream_handler.start()
        app["streams"][stream_id] = stream_handler

        # Add WebSocket route for this stream
        app.router.add_route("GET", f"/ws/{stream_id}", websocket_handler)

    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)

    logger.info("Starting server...")
    web.run_app(app, host=args.host, port=args.port)
