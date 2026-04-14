import bluetooth
import json
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo para extraer características (sin la capa de clasificación final)
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Diccionario para guardar las "huellas digitales" de tus Pokémon
pokemon_database = {}

def load_local_pokemons():
    folder = "pokemons"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Alerta: Carpeta '{folder}' creada. Agregá imágenes ahí.")
        return

    print("Cargando base de datos de Pokémon...")
    for file in os.listdir(folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
            feature_vector = base_model.predict(np.expand_dims(img_array, axis=0))
            
            name = os.path.splitext(file)[0].capitalize()
            pokemon_database[name] = feature_vector
    print(f"Base de datos lista: {list(pokemon_database.keys())}")

def find_most_similar(image_bytes):
    # Procesar imagen recibida
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    target_vector = base_model.predict(np.expand_dims(img_array, axis=0))

    best_match = "Desconocido"
    highest_score = 0

    # Comparar contra toda la base de datos
    for name, feat_vector in pokemon_database.items():
        score = cosine_similarity(target_vector, feat_vector)[0][0]
        if score > highest_score:
            highest_score = score
            best_match = name

    # Lógica de stats (Similitud de 0.0 a 1.0)
    return {
        "name": best_match,
        "hp": int(highest_score * 200),
        "attack": int(highest_score * 100),
        "similarity": f"{highest_score:.2%}"
    }

def start_server():
    load_local_pokemons()
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    
    uuid = "00001101-0000-1000-8000-00805F9B34FB"
    bluetooth.advertise_service(server_sock, "PokeScanner", service_id=uuid)
    print("Servidor Bluetooth esperando fotos...")

    while True:
        client_sock, client_info = server_sock.accept()
        data = b""
        try:
            while True:
                chunk = client_sock.recv(4096)
                if not chunk: break
                data += chunk
            
            if data:
                result = find_most_similar(data)
                client_sock.send(json.dumps(result))
        except Exception as e:
            print(f"Error procesando: {e}")
        finally:
            client_sock.close()

if __name__ == "__main__":
    start_server()

