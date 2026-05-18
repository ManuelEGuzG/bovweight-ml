from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["MODEL_PATH"]           = os.getenv("MODEL_PATH", "app/models/bovine_yolov8.pt")
    app.config["CONFIDENCE_THRESHOLD"] = float(os.getenv("CONFIDENCE_THRESHOLD", 0.1))
    app.config["MIN_WEIGHT_KG"]        = float(os.getenv("MIN_WEIGHT_KG", 80))
    app.config["MAX_WEIGHT_KG"]        = float(os.getenv("MAX_WEIGHT_KG", 900))

    from app.routes.estimate import estimate_bp
    from app.routes.health   import health_bp
    from app.routes.feedback import feedback_bp

    app.register_blueprint(estimate_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp,   url_prefix="/api/v1")
    app.register_blueprint(feedback_bp, url_prefix="/api/v1")

    return app