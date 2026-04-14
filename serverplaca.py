import asyncio
import json
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from bleak import BleakPeripheral

# --- LÓGICA DE IA (Igual a la anterior) ---
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
pokemon_database = {}

def load_local_pokemons():
    folder = "pokemons"
    if not os.path.exists(folder): os.makedirs(folder)
    for file in os.listdir(folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
            feature_vector = base_model.predict(np.expand_dims(img_array, axis=0))
            pokemon_database[os.path.splitext(file)[0].capitalize()] = feature_vector
    print(f"Base de datos cargada: {list(pokemon_database.keys())}")

def find_most_similar(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    target_vector = base_model.predict(np.expand_dims(img_array, axis=0))
    
    best_match, highest_score = "Desconocido", 0
    for name, feat_vector in pokemon_database.items():
        score = cosine_similarity(target_vector, feat_vector)[0][0]
        if score > highest_score:
            highest_score, best_match = score, name

    return {
        "name": best_match,
        "hp": int(highest_score * 200),
        "attack": int(highest_score * 100),
        "similarity": f"{highest_score:.2%}"
    }

# --- CONFIGURACIÓN BLEAK (SERVIDOR) ---
# UUIDs únicos para identificar tu servicio de Pokémon
SERVICE_UUID = "12345678-1234-5678-1234-567812345678"
CHAR_UUID = "87654321-4321-4321-4321-876543210987"

class PokemonPeripheral:
    def __init__(self):
        self.received_data = bytearray()

    def on_write(self, characteristic, data):
        # El cliente envía la imagen en trozos
        self.received_data.extend(data)
        print(f"Recibiendo datos... ({len(self.received_data)} bytes)")

    async def run(self):
        load_local_pokemons()
        
        # En Bleak 0.22.2 se usa el backend del sistema para anunciar
        async with BleakPeripheral() as server:
            await server.add_service(SERVICE_UUID)
            await server.add_characteristic(
                SERVICE_UUID, 
                CHAR_UUID, 
                ["read", "write", "notify"], 
                self.on_write
            )
            
            print(f"Servidor BLE activo. Buscando por UUID: {SERVICE_UUID}")
            
            while True:
                # Si terminamos de recibir (puedes implementar un timeout o flag)
                if len(self.received_data) > 0:
                    # Esperamos un momento a que termine de llegar todo
                    await asyncio.sleep(2) 
                    print("Procesando imagen...")
                    result = find_most_similar(bytes(self.received_data))
                    
                    # Limpiamos para la próxima foto
                    self.received_data = bytearray()
                    print(f"Resultado: {result}")
                    
                await asyncio.sleep(1)

if __name__ == "__main__":
    peripheral = PokemonPeripheral()
    asyncio.run(peripheral.run())
