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
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor, TreeOutput

# from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

infer_enabled = False
camera = None
camera_needs_reset = False # 當 RTSP_URL 更新時重開 camera

# scheduler
scheduler = BackgroundScheduler()
scheduler.start()


# LOG FILE SETTING
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app_log.log")
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
            if not self.camera.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                self.running = False
                return
            threading.Thread(target=self._capture_loop, daemon=True).start()
            logger.info(f"RTSP stream {self.rtsp_url} started.")

    def stop(self):
        if self.running:
            self.running = False
            if self.camera:
                self.camera.release()
            logger.info(f"RTSP stream {self.rtsp_url} stopped.")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                logger.warning(
                    f"Failed to read frame from RTSP stream: {self.rtsp_url}"
                )
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            frame_data = buffer.tobytes()

            for ws in app["websockets"]:
                asyncio.run(ws.send_bytes(frame_data))

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


def get_colors(count: int):
    cmap = plt.cm.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color))
    return colors


def draw_tree_output(
    image,
    output: TreeOutput,
    tree: Tree,
    draw_text=True,
    num_colors=8,
    valid_region=None,
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

        if valid_region:
            if (
                box[0] < valid_region["top_left_x"]
                or box[1] < valid_region["top_left_y"]
                or box[2] > valid_region["bottom_right_x"]
                or box[3] > valid_region["bottom_right_y"]
            ):
                continue

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
    global audio_enabled

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info("Audio WebSocket connected.")

    process = (
        ffmpeg.input(rtsp_url)
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
            if not audio_enabled:
                await asyncio.sleep(0.1)
                continue

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
    global prompt_data, coordinates_data, RTSP_URL, audio_enabled, infer_enabled, camera_needs_reset
    audio_enabled = False

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info(f"WebSocket connected: {request.remote}")
    request.app["websockets"].add(ws)

    try:
        async for msg in ws:
            data = json.loads(msg.data)
            action = data.get("action")
            stream_id = data.get("stream_id")
            rtsp_url = data.get("rtsp_url")

            logging.info(f"Received message from websocket: {msg.data}")

            if action == "start" and rtsp_url:
                if rtsp_url != RTSP_URL:
                    # RTSP_URL = rtsp_url
                    camera_needs_reset = True
                    logger.info(f"RTSP URL updated to: {rtsp_url}")
                else:
                    logger.info("Recived same RTSP URL.")

                if "current_stream" in request.app:
                    request.app["current_stream"].stop()

                stream_handler = RTSPStreamHandler(rtsp_url)
                stream_handler.start()
                request.app["current_stream"] = stream_handler

            elif action == "toggle_infer":
                enable_infer_str = data.get("enable_infer", "true")
                infer_enabled = (enable_infer_str == "true")
                logger.info(f"infer_enabled set to: {infer_enabled}")

            elif action == "stop":
                if "current_stream" in request.app:
                    request.app["current_stream"].stop()
                    del request.app["current_stream"]
                    logger.info("Stream stopped.")

            elif action == "toggle_audio":
                enable_audio = data.get("enable_audio", True)
                audio_enabled = enable_audio
                if audio_enabled:
                    logger.info(f"Audio streaming enabled for stream: {stream_id}")
                else:
                    logger.info(f"Audio streaming disabled for stream: {stream_id}")

            elif action == "schedule_stream":
                rtsp_url = data.get("rtsp_url")
                start_time = data.get("start_time")
                stop_time = data.get("stop_time")

                stream_handler = RTSPStreamHandler(rtsp_url, request.app["predictor"])

                # 排程
                if start_time:
                    scheduler.add_job(
                        stream_handler.start,
                        "date",
                        run_date=datetime.fromisoformat(start_time),
                        id=f"start_{rtsp_url}",
                        replace_existing=True,
                    )
                    logger.info(
                        f"Scheduled start for RTSP stream: {rtsp_url} at {start_time}"
                    )

                if stop_time:
                    scheduler.add_job(
                        stream_handler.stop,
                        "date",
                        run_date=datetime.fromisoformat(stop_time),
                        id=f"stop_{rtsp_url}",
                        replace_existing=True,
                    )
                    logger.info(
                        f"Scheduled stop for RTSP stream: {rtsp_url} at {stop_time}"
                    )

            if "json" in msg.data:
                try:
                    data = json.loads(msg.data)["json"]
                    prompt = f"[{', '.join([item['object'] for item in data])}]"
                    thresholds = {
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
                        "thresholds": thresholds,
                    }
                except Exception as e:
                    logging.error(f"Error generating prompt data: {e}")

            elif any(
                key in msg.data
                for key in ["top_left", "top_right", "bottom_left", "bottom_right"]
            ):
                try:
                    raw_coordinates = json.loads(msg.data)

                    required_keys = [
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                    ]
                    if not all(key in raw_coordinates for key in required_keys):
                        raise ValueError(
                            f"Incomplete coordinate data: {raw_coordinates}"
                        )

                    for key in required_keys:
                        if not all(k in raw_coordinates[key] for k in ["x", "y"]):
                            raise ValueError(
                                f"Incomplete data for {key}: {raw_coordinates[key]}"
                            )

                    coordinates_data = {
                        "top_left_x": raw_coordinates["top_left"]["x"],
                        "top_left_y": raw_coordinates["top_left"]["y"],
                        "top_right_x": raw_coordinates["top_right"]["x"],
                        "top_right_y": raw_coordinates["top_right"]["y"],
                        "bottom_left_x": raw_coordinates["bottom_left"]["x"],
                        "bottom_left_y": raw_coordinates["bottom_left"]["y"],
                        "bottom_right_x": raw_coordinates["bottom_right"]["x"],
                        "bottom_right_y": raw_coordinates["bottom_right"]["y"],
                    }

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
    if "current_stream" in app:
        app["current_stream"].stop
    for ws in set(app["websockets"]):
        await ws.close(code=WSCloseCode.GOING_AWAY, message="Server shutdown")


async def detection_loop(app: web.Application):
    global camera, RTSP_URL, camera_needs_reset
    loop = asyncio.get_running_loop()

    logging.info("Opening camera.")
    camera = cv2.VideoCapture(rtsp_url)

    if not camera.isOpened():
        logging.error("Failed to open RTSP stream.")
        return

    logging.info("Loading predictor.")

    last_time = time.time()
    frame_count = 0
    fps_val = 0

    def _read_and_encode_image():
        global camera, RTSP_URL, camera_needs_reset, infer_enabled

        if camera_needs_reset:
            logger.info(f"switch RTSP input to: {RTSP_URL}")
            if camera is not None:
                camera.release()
            camera = cv2.VideoCapture(rtsp_url)
            camera_needs_reset = False

        re, image = camera.read()
        # logging.info(f"RTSP stream read result: {re}")

        if not re:
            warning_msg = "Failed to capture frame from RTSP stream."
            logging.warning(warning_msg)

            camera.release()
            camera = cv2.VideoCapture(rtsp_url)

            return re, None

        """
        Detection phase
        """
        image_pil = cv2_to_pil(image)
        detections = []

        if prompt_data is not None and infer_enabled:
            try:
                prompt_data_local = prompt_data
                # logging.info(f"Using prompt_data: {prompt_data_local}")

                if coordinates_data:
                    logging.info(f"Processing coordinates: {coordinates_data}")
                    image, transformed_regions = process_coordinates_with_affine(
                        image, [coordinates_data]
                    )
                    if not transformed_regions:
                        logging.warning(
                            "No valid regions found after affine transformation."
                        )
                        return image

                    for warped_image, (src_pts, dst_pts) in transformed_regions:
                        try:
                            cropped_pil = cv2_to_pil(warped_image)

                            region_detections = predictor.predict(
                                image=cropped_pil,
                                tree=prompt_data_local["tree"],
                                clip_text_encodings=prompt_data_local["clip_encodings"],
                                owl_text_encodings=prompt_data_local["owl_encodings"],
                            )

                            if isinstance(region_detections, TreeOutput):
                                detections.extend(region_detections.detections)
                            elif isinstance(region_detections, list):
                                detections.extend(region_detections)
                            else:
                                logging.error("Unexpected predictor output type.")

                        except Exception as e:
                            logging.error(f"Prediction error for region: {e}")
                            region_detections = []
                    try:
                        tree_output = TreeOutput(detections=detections)
                        image = draw_tree_output(
                            image, tree_output, prompt_data_local["tree"]
                        )
                    except Exception as e:
                        logging.error(f"Failed to create TreeOutput: {e}")

                else:
                    logging.info(f"Performing full image detection with prompt_data.")
                    t0 = time.perf_counter_ns()
                    try:
                            detections = predictor.predict(
                                image=image_pil,
                                tree=prompt_data_local["tree"],
                                clip_text_encodings=prompt_data_local["clip_encodings"],
                                owl_text_encodings=prompt_data_local["owl_encodings"],
                                # threshold=prompt_data_local["thresholds"],
                            )
                            t1 = time.perf_counter_ns()
                            dt = (t1 - t0) / 1e9

                            logging.info(f"Prediction completed in {dt:.3f} seconds.")
                            logging.info(f"Raw detections: {detections}")

                            if isinstance(detections, list):
                                detections = TreeOutput(detections=detections)
                            elif not isinstance(detections, TreeOutput):
                                logging.error(
                                    f"Unexpected prediction output type: {type(detections)}"
                                )
                                detections = TreeOutput(detections=[])
                            logging.info(f"Detections: {detections}")
                            image = draw_tree_output(
                                image, detections, prompt_data_local["tree"]
                            )
                    except Exception as e:
                        logging.error(f"Prediction error: {e}")

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
            ret, image_data = await loop.run_in_executor(None, _read_and_encode_image)

            if not ret or image_data is None:
                continue

            for ws in app["websockets"]:
                try:
                    await ws.send_bytes(image_data)
                except Exception as e:
                    logger.error(f"Failed to send frame to client: {e}")

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0: # per 1 sec
                fps_val = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            fps_data = {
                "type": "fps",
                "value": round(fps_val, 2)
            }

            fps_json = json.dumps(fps_data)

            for ws in app["websockets"]:
                try:
                        await ws.send_str(fps_json)
                except Exception as e:
                    logger.error(f"Failed to send fps info: {e}")
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
        default="rtsp://rtspstream:nbuHSDAKaijbiQnDw9fby@zephyr.rtsp.stream/people",
    )
    args = parser.parse_args()

    CAMERA_DEVICE = args.camera
    RTSP_URL = args.rtsp_url
    IMAGE_QUALITY = args.image_quality

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