from flask import Flask, jsonify
from utils import ConfigManager

app = Flask(__name__)
manager = ConfigManager("config.json")


@app.route("/config", methods=["GET"])
def get_config():
    data = manager.get("settings")
    return jsonify(data)


@app.route("/config/reload", methods=["POST"])
def reload_config():
    manager.reload()
    return jsonify({"status": "reloaded"})
