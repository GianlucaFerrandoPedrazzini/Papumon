import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- IA ---
print("Cargando modelo IA...")
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

pokemon_database = {}

def load_local_pokemons():
    folder = "pokemons"
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Cargando base de datos de Pokémon...")

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, file)

            img = Image.open(path).convert("RGB").resize((224, 224))
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.array(img)
            )

            vec = base_model.predict(
                np.expand_dims(img_array, axis=0),
                verbose=0
            )

            name = os.path.splitext(file)[0].capitalize()
            pokemon_database[name] = vec

    print(f"Base lista: {len(pokemon_database)} Pokémon")

def analyze_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))

        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.array(img)
        )

        target_vec = base_model.predict(
            np.expand_dims(img_array, axis=0),
            verbose=0
        )

        best_name = "Desconocido"
        best_score = 0

        for name, vec in pokemon_database.items():
            score = float(cosine_similarity(target_vec, vec)[0][0])

            if score > best_score:
                best_score = score
                best_name = name

        return {
            "name": best_name,
            "similarity": f"{best_score:.2%}",
            "hp": int(best_score * 200) + 10,
            "attack": int(best_score * 100) + 5
        }

    except Exception as e:
        print("Error procesando imagen:", e)
        return {"error": "imagen inválida"}


# --- ENDPOINT ---
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        result = analyze_image(contents)

        return JSONResponse(content=result)

    except Exception as e:
        print("Error en endpoint:", e)
        return JSONResponse(
            content={"error": "fallo en servidor"},
            status_code=500
        )


# --- INIT ---
if __name__ == "__main__":
    load_local_pokemons()

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # IMPORTANTE (acceso desde celular)
        port=8000
    )
