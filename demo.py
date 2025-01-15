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


class RTSPStreamHandler:
    def __init__(self, rtsp_url, model, queue_size=10):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=queue_size)
        self.camera = None
        self.running = False
        self.model = model

    def start(self):
        if not self.running:
            self.running = True
            self.camera = cv2.VideoCapture(self.rtsp_url)
            threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        if self.running:
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
                frame = self._process_frame(frame)
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame)
            else:
                logger.warning(
                    f"Failed to read frame from RTSP stream: {self.rtsp_url}"
                )

    def _process_frame(self, frame):
        try:
            detections = self.model.predict(frame)
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                label = detection["label"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
        return frame

    async def get_frame(self):
        return await self.frame_queue.get()


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


def process_coordinates_with_affine(image, coordinates):
    """
    Perform an affine transformation based on the coordinates of an irregular quadrilateral and return the transformed image along with the processed region's information.
    """
    transformed_regions = []

    if not coordinates or not isinstance(coordinates, list):
        raise ValueError("Coordinates must be a non-empty list of dictionaries.")

    for coord in coordinates:
        try:
            required_keys = [
                "top_left_x",
                "top_left_y",
                "top_right_x",
                "top_right_y",
                "bottom_right_x",
                "bottom_right_y",
                "bottom_left_x",
                "bottom_left_y",
            ]
            if not all(key in coord for key in required_keys):
                logging.error(f"Missing keys in coordinate: {coord}")
                raise ValueError(f"Incomplete coordinate data: {coord}")

            logging.info(f"Initializing src_pts with: {coord}")
            src_pts = np.array(
                [
                    [int(coord["top_left_x"]), int(coord["top_left_y"])],
                    [int(coord["top_right_x"]), int(coord["top_right_y"])],
                    [int(coord["bottom_right_x"]), int(coord["bottom_right_y"])],
                    [int(coord["bottom_left_x"]), int(coord["bottom_left_y"])],
                ],
                dtype=np.float32,
            )

            width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
            height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

            dst_pts = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_image = cv2.warpPerspective(image, matrix, (width, height))
            cv2.polylines(
                image,
                [src_pts.astype(np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

            transformed_regions.append((warped_image, (src_pts, dst_pts)))

        except Exception as e:
            logging.error(f"Error during affine transform: {e}")

    return image, transformed_regions


# WebSocket handler for audio transmission
async def audio_websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info("Audio WebSocket connected.")

    # rtsp_url = "rtsp://35.185.165.215:31554/mystream1"

    process = (
        ffmpeg.input(RTSP_URL)
        .output(
            "pipe:",
            format="adts",
            acodec="aac",
            ar="44100",
            ac=1,
            b="128k",
        )
        .global_args("-hide_banner", "-loglevel", "error")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    try:
        while True:
            audio_data = await asyncio.to_thread(process.stdout.read, 2048)
            if not audio_data:
                logging.warning("No audio data received, stream ended.")
                break
            await ws.send_bytes(audio_data)
    except Exception as e:
        logging.error(f"Error in audio stream: {e}")
    finally:
        logging.info("Audio WebSocket disconnected.")
        process.terminate()
        await ws.close()
    return ws


async def websocket_handler(request):
    global prompt_data, coordinates_data

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"WebSocket connected: {request.remote}")
    request.app["websockets"].add(ws)

    global audio_enabled
    audio_enabled = True

    try:
        async for msg in ws:
            # ws json object prompt : {"json": "[{\"object\": \"apple\", \"threshold\": \"0.5\"}, {\"object\": \"banana\", \"threshold\": \"0.5\"}]"}
            # ws json testing data example: {"coordinate": "[{\"top_left_x\": \"200\",\"top_left_y\": \"250\", \"bottom_left_x\": \"500\", \"bottom_left_y\": \"550\"}]"}

            data = json.loads(msg.data)
            action = data.get("action")
            stream_id = data.get("stream_id")

            logging.info(f"Received message from websocket: {msg.data}")

            # ?? prompt
            if "json" in msg.data:
                try:
                    data = json.loads(msg.data)["json"]
                    prompt = f"[{', '.join([item['object'] for item in data])}]"
                    threshold = {
                        item["object"]: float(item["threshold"]) for item in data
                    }
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
                    raw_coordinates = json.loads(msg.data)["coordinate"]
                    coordinates = json.loads(raw_coordinates)

                    for coord in coordinates:
                        required_keys = [
                            "top_left_x",
                            "top_left_y",
                            "top_right_x",
                            "top_right_y",
                            "bottom_right_x",
                            "bottom_right_y",
                            "bottom_left_x",
                            "bottom_left_y",
                        ]
                        if not all(key in coord for key in required_keys):
                            raise ValueError(f"Incomplete coordinate data: {coord}")
                    coordinates_data = coordinates
                    logging.info(f"Coordinates updated: {coordinates_data}")
                except Exception as e:
                    logging.error(f"Error parsing coordinates: {e}")

            # 處理 streams 開關 -> 改成用i/f moudle 讀 config 來執行
            elif "action" in msg.data:
                if action == "enable":
                    for sid, handler in request.app["streams"].items():
                        if sid == stream_id:
                            handler.start()
                            logger.info(f"Stream {stream_id} started.")
                        else:
                            handler.stop()
                            logger.info(f"Stream {sid} stopped.")
                elif action == "disable":
                    if stream_id in request.app["streams"]:
                        request.app["streams"][stream_id].stop()
                        logger.info(f"Stream {stream_id} stopped.")
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

            # ????
            camera.release()
            camera = cv2.VideoCapture(RTSP_URL)
            if not camera.isOpened():
                logging.error("Reconnection attempt failed.")
                return re, None

            return re, None

        """
        Detection phase
        """
        image_pil = cv2_to_pil(image)

        if prompt_data is not None:
            try:
                prompt_data_local = prompt_data
                # logging.info(f"Using prompt_data: {prompt_data_local}")

                if coordinates_data:
                    logging.info(f"Processing coordinates: {coordinates_data}")
                    image, transformed_regions = process_coordinates_with_affine(
                        image, coordinates_data
                    )
                    if not transformed_regions:
                        logging.warning(
                            "No valid regions found after affine transformation."
                        )
                        return image
                    detections = []
                    for warped_image, (src_pts, dst_pts) in transformed_regions:
                        try:
                            cropped_pil = cv2_to_pil(warped_image)
                            region_detections = predictor.predict(
                                image=cropped_pil,
                                tree=prompt_data_local["tree"],
                                clip_text_encodings=prompt_data_local["clip_encodings"],
                                owl_text_encodings=prompt_data_local["owl_encodings"],
                                threshold=prompt_data_local["thresholds"],
                            )
                            detections.extend(region_detections)
                        except Exception as e:
                            logging.error(f"Prediction error for region: {e}")
                    image = draw_tree_output(
                        image, detections, prompt_data_local["tree"]
                    )

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

    for idx, rtsp_url in enumerate(args.rtsp_url):
        stream_id = f"stream{idx+1}"
        stream_handler = RTSPStreamHandler(rtsp_url, predictor)
        app["streams"][stream_id] = stream_handler

        app.router.add_route("GET", f"/ws/{stream_id}", websocket_handler)

    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.router.add_route("GET", "/audio", audio_websocket_handler)

    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)

    logger.info("Starting server...")
    web.run_app(app, host=args.host, port=args.port)
