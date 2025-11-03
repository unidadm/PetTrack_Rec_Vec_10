# Paso 1: Importaciones versión para Render.com (sin ngrok)
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import torch
from transformers import CLIPProcessor, CLIPModel

# Paso 2: Cargar modelo CLIP (una vez, global)
#         Modelos comunes: "openai/clip-vit-base-patch32" (rápido)
#                     "openai/clip-vit-large-patch14" (más preciso, más pesado)
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Paso 3: Crear la aplicación Flask
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": device})

# Ruta que procesa la imagen y retorna el vector
@app.route("/vector", methods=["POST"])
def generar_vector():
    ###
    # Recibe un archivo de imagen con el campo 'file' (multipart/form-data)
    # y devuelve el embedding de imagen de CLIP como CSV string.
    ###
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        # Asegurar RGB (quita canal alfa si es PNG)
        image = Image.open(file.stream).convert("RGB")

        # Preprocesar con CLIP
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)  # [1, D]
            # Normalizar a norma unitaria (recomendado para similitud coseno)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        # Pasar a lista de floats (Python nativo)
        vector = img_features.squeeze(0).detach().cpu().numpy().astype(float).tolist()
        
        # CSV para VARCHAR(MAX)
        vector_str = ",".join(str(v) for v in vector)
        return jsonify({"vector": vector_str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Paso 4: Punto de entrada
if __name__ == "__main__":
    # Render asigna el puerto mediante la variable de entorno PORT
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
