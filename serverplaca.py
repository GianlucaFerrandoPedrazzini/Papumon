import asyncio
import json
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from bless import BlessServer, BlessGATTCharacteristic, GATTCharacteristicProperties, GATTAttributePermissions

# --- IA Y PROCESAMIENTO ---
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
            pokemon_database[os.path.splitext(file)[0].capitalize()] = base_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    print(f"Base de datos de Pokémon lista ({len(pokemon_database)} cargados).")

def analyze_image(data):
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB').resize((224, 224))
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
        target_vec = base_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        
        best_name, best_score = "Desconocido", 0
        for name, vec in pokemon_database.items():
            score = float(cosine_similarity(target_vec, vec)[0][0])
            if score > best_score: 
                best_score, best_name = score, name
        
        return json.dumps({
            "name": best_name, 
            "hp": int(best_score * 200) + 10, 
            "attack": int(best_score * 100) + 5, 
            "similarity": f"{best_score:.2%}"
        }).encode('utf-8')
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return b'{"error": "invalid image"}'

# --- SERVIDOR BLE CON BLESS 0.3.0 ---
# UUIDs ACTUALIZADOS para coincidir con tu cliente
SERVICE_UUID = "12345678-1234-5678-1234-567812345678"
CHAR_UUID = "87654321-4321-4321-4321-876543210987"

class PokemonServer:
    def __init__(self):
        self.image_buffer = bytearray()
        self.last_result = b'{"status": "esperando"}'

    def on_write(self, characteristic, value):
        self.image_buffer.extend(value)
        print(f"Recibiendo datos... acumulado: {len(self.image_buffer)} bytes")

    def on_read(self, characteristic):
        return self.last_result

async def run_server():
    load_local_pokemons()
    
    server = BlessServer(name="PokeScanner-Server")
    poke_logic = PokemonServer()

    await server.add_new_service(SERVICE_UUID)
    
    # SOLUCIÓN AL ERROR: GATTAttributePermissions.writeable (con 'e')
    await server.add_new_characteristic(
        SERVICE_UUID, 
        CHAR_UUID,
        GATTCharacteristicProperties.read | GATTCharacteristicProperties.write,
        b"Listo", 
        GATTAttributePermissions.readable | GATTAttributePermissions.writeable
    )
    
    server.write_request_func = poke_logic.on_write
    server.read_request_func = poke_logic.on_read
    
    await server.start()
    print("------------------------------------------")
    print("Servidor ONLINE - Esperando fotos...")
    
    while True:
        # Lógica de detección: si pasan 2 segundos sin recibir datos nuevos, procesamos
        if len(poke_logic.image_buffer) > 0:
            current_size = len(poke_logic.image_buffer)
            await asyncio.sleep(2) 
            if len(poke_logic.image_buffer) == current_size:
                print("Procesando imagen recibida...")
                poke_logic.last_result = analyze_image(poke_logic.image_buffer)
                poke_logic.image_buffer = bytearray() # Reset buffer
                print("Resultado listo para ser leído por el cliente.")
        
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(run_server())
