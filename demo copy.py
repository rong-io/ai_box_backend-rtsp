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

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor, TreeOutput

# from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

RTSP_URL = None

# scheduler
scheduler = BackgroundScheduler()
scheduler.start()


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

class TreePredictor:
    def __init__(self, owl_predictor):
        self.owl_predictor = owl_predictor
        self.tree = None
        self.thresholds = dict()

class RTSPStreamHandler:
    def __init__(self, rtsp_url, model, queue_size=10):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=queue_size)
        self.camera = None
        self.running = False
        self.model = model
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.camera = cv2.VideoCapture(
                f"{self.rtsp_url}?fflags=nobuffer&flags=low_delay",
                cv2.CAP_FFMPEG
            )
            if not self.camera.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                self.running = False
                return

            self.thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self.thread.start()
            logger.info(f"RTSP stream {self.rtsp_url} started.")

    def stop(self):
        if self.running:
            self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

        # 在執行緒結束後，才真的 release camera
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None

        logger.info(f"RTSP stream {self.rtsp_url} stopped.")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                logger.warning(f"Failed to read frame from RTSP: {self.rtsp_url}")
                time.sleep(0.1)
                continue

            try:
                # 如果 self.model 有 tree 屬性且不為 None，就做預測
                if hasattr(self.model, "tree") and self.model.tree:
                    image_pil = cv2_to_pil(frame)
                    tree_output = self.model.predict(image_pil, self.model.tree)
                    frame = draw_tree_output(frame, tree_output, self.model.tree)

                    _, buffer = cv2.imencode(".jpg", frame)
                    frame_data = buffer.tobytes()

                for ws in app["websockets"]:
                    asyncio.run(ws.send_bytes(frame_data))

            except Exception as e:
                logger.error(f"Error during frame processing: {e}")

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

    rtsp_url = request.app.get("rtsp_url", None)
    if not rtsp_url:
        logging.error("No RTSP URL available for audio streaming.")
        return web.Response(
            text="No RTSP URL available for audio streaming.", 
            status=400
        )

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
    global prompt_data, coordinates_data, RTSP_URL, audio_enabled
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

            # if action == "start" or action == "updated":
            #     if rtsp_url:
            #         RTSP_URL = rtsp_url
            #         request.app["rtsp_url"] = rtsp_url
            #         logger.info(f"RTSP URL updated to: {rtsp_url}")
            #     else:
            #         logger.warning("No RTSP URL found in websocket data.")

            #     empty_tree = Tree.from_prompt("[]")
            #     request.app["predictor"].tree = empty_tree

            #     if "current_stream" in request.app:
            #         request.app["current_stream"].stop()

            #     stream_handler = RTSPStreamHandler(RTSP_URL, request.app["predictor"])
            #     stream_handler.start()
            #     request.app["current_stream"] = stream_handler

            if action == "stop":
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
                predicted_object_1 = data.get("predicted_object_1")
                predicted_object_2 = data.get("predicted_object_2")
                thresholds = {
                    predicted_object_1: data.get("predicted_object_1_threshold"),
                    predicted_object_2: data.get("predicted_object_2_threshold"),
                }

                objects = [
                    obj for obj in [predicted_object_1, predicted_object_2] if obj
                ]
                tree_prompt = f"[{', '.join(objects)}]"
                tree = Tree.from_prompt(tree_prompt)

                # 更新 RTSPStreamHandler
                stream_handler = RTSPStreamHandler(rtsp_url, request.app["predictor"])
                stream_handler.model.tree = tree  # 將 Tree 傳遞給模型
                stream_handler.model.thresholds = thresholds  # 設定 threshold

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

            if "json" or "action" == "start" in msg.data:
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
        # logging.info(f"RTSP stream read result: {re}")

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
                        image, [coordinates_data]
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
                            t0 = time.perf_counter_ns()
                            region_detections = predictor.predict(
                                image=cropped_pil,
                                tree=prompt_data_local["tree"],
                                clip_text_encodings=prompt_data_local["clip_encodings"],
                                owl_text_encodings=prompt_data_local["owl_encodings"],
                            )
                            t1 = time.perf_counter_ns()
                            dt = (t1 - t0) / 1e9
                            fps = 1 / dt if dt > 0 else 0

                            logging.info(
                                f"Prediction completed in {dt:.3f} seconds, FPS = {fps:.2f}"
                            )
                            logging.info(f"Raw detections: {detections}")

                            # logging.info("Ready to send FPS data to web...")
                            # fps_msg = json.dumps({"type": "fps", "value": fps})

                            # for ws in app["websockets"]:
                            #     await ws.send_str(json.dumps({"type": "fps", "value": fps}))

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
                        fps = 1 / dt if dt > 0 else 0

                        logging.info(
                            f"Prediction completed in {dt:.3f} seconds, FPS = {fps:.2f}"
                        )
                        logging.info(f"Raw detections: {detections}")

                        # logging.info("Ready to send FPS data to web...")
                        # fps_msg = json.dumps({"type": "fps", "value": fps})

                        # for ws in app["websockets"]:
                        #         await ws.send_str(json.dumps({"type": "fps", "value": fps}))

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

        return re, image_jpeg, fps

    try:
        while True:
            re, image = await loop.run_in_executor(None, _read_and_encode_image)

            if not re:
                continue

            fps_msg = json.dumps({"type": "fps", "value": fps})
            for ws in app["websockets"]:
                await ws.send_str(fps_msg)

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

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(image_encoder_engine=args.image_encode_engine)
    )

    prompt_data = None
    coordinates_data = None

    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app["streams"] = {}
    app["websockets"] = weakref.WeakSet()
    app["predictor"] = predictor
    app["fps_websockets"] = weakref.WeakSet()

    if isinstance(RTSP_URL, str):
        rtsp_urls = [RTSP_URL]
    else:
        rtsp_urls = RTSP_URL

    for idx, rtsp_url in enumerate(args.rtsp_url):
        stream_id = f"stream{idx+1}"
        stream_handler = RTSPStreamHandler(rtsp_url, predictor)
        app["streams"][stream_id] = stream_handler

        app.router.add_route("GET", f"/ws/{stream_id}", websocket_handler)

    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.router.add_route("GET", "/audio", audio_websocket_handler)
    # app.router.add_route("GET", "/ws/fps", fps_websocket_handler)

    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(run_detection_loop)

    logger.info("Starting server...")
    web.run_app(app, host=args.host, port=args.port)
