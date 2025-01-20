from flask import Flask, jsonify, request
import json
import subprocess
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SETTINGS_FILE = "RTSP_SETTINGS.json"

def load_settings():
    with open(SETTINGS_FILE, 'r') as file:
        return json.load(file)


def get_gpu_metrics():
    try:
        process = subprocess.Popen(['tegrastats', '--interval', '100'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                          universal_newlines=True)

        for _ in range(2):
            process.stdout.readline()

        tegra_line = process.stdout.readline().strip()
        process.terminate()

        gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', tegra_line)
        gpu_usage = int(gpu_match.group(1)) if gpu_match else 0

        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', tegra_line)
        ram_used = int(ram_match.group(1)) if ram_match else 0
        ram_total = int(ram_match.group(2)) if ram_match else 0

        return {
            "gpu_usage": gpu_usage,
            "ram_used": ram_used,
            "ram_total": ram_total
        }
    except Exception as e:
        return {"error": str(e)}



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


@app.route('/get_gpu_metrics', methods=['GET'])
def api_get_gpu_metrics():
    metrics = get_gpu_metrics()
    if "error" in metrics:
        return jsonify({"status": "error", "message": metrics["error"]}), 500
    return jsonify({"status": "success", "data": metrics}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
