from flask import Flask, jsonify, request
import json
import os

app = Flask(__name__)

SETTINGS_FILE = "RTSP_SETTINGS.json"


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return []
    with open(SETTINGS_FILE, "r") as file:
        return json.load(file)


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)


@app.route('/get_streams', methods=['GET'])
def get_streams():
    try:
        settings = load_settings()
        return jsonify({"status": "success", "data": settings}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_stream/<stream_id>', methods=['GET'])
def get_stream(stream_id):
    try:
        settings = load_settings()
        stream = next((item for item in settings if item["stream_id"] == stream_id), None)
        if stream:
            return jsonify({"status": "success", "data": stream}), 200
        else:
            return jsonify({"status": "error", "message": "Stream not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/update_stream", methods=["POST"])
def update_stream():
    try:
        data = request.json
        if not data or "stream_id" not in data:
            return jsonify({"status": "error", "message": "Invalid input"}), 400

        settings = load_settings()

        existing_stream = next(
            (item for item in settings if item["stream_id"] == data["stream_id"]), None
        )

        if existing_stream:
            existing_stream.update(data)
            message = "Stream updated"
        else:
            settings.append(data)
            message = "Stream added"

        save_settings(settings)

        return jsonify({"status": "success", "message": message, "data": data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
