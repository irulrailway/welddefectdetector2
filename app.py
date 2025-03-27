import os
import cv2
import torch
import requests
import numpy as np
from flask import Flask, request, jsonify
from transformers import DetrImageProcessor
from collections import namedtuple
from PIL import Image

app = Flask(__name__)

# 🔹 Model dari Hugging Face
MODEL_URL = "https://huggingface.co/irulBES/welddefectdetector/resolve/main/model.torchscript2.pt"
MODEL_PATH = "model.torchscript2.pt"

# 🔹 Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("⏳ Mengunduh model dari Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model berhasil diunduh!")

# 🔹 Load model TorchScript
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# 🔹 Load image processor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 🔹 Kelas cacat las
CLASS_NAMES = {
    0: "weld-defect-detect",
    1: "slag inclusion",
    2: "spatter",
    3: "undercut"
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Flask untuk deteksi cacat las berjalan 🚀"})

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")

    # 🔹 Deteksi Objek
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(inputs["pixel_values"])

        # 🔹 Pastikan format output sesuai
        if isinstance(outputs, tuple):
            logits, pred_boxes = outputs
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
            pred_boxes = outputs["pred_boxes"]
        else:
            return jsonify({"error": "Format output model tidak dikenali!"}), 500

        DETR_Output = namedtuple("DETR_Output", ["logits", "pred_boxes"])
        outputs = DETR_Output(logits=logits, pred_boxes=pred_boxes)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]

    # 🔹 Ambil hasil deteksi
    bboxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    class_ids = results["labels"].cpu().numpy()

    detections = []
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        detections.append({
            "class": CLASS_NAMES[class_id],
            "bbox": bbox.tolist(),
            "score": float(score)
        })

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
