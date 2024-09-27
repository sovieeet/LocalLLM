import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# pregunta = "¿qué es la democracia?"

# prompt = (
#     """
#     Eres un asistente que responde preguntas de manera relativamente sencilla. Por favor responde de manera clara y concisa las preguntas que te hagan.
#     Evita repetir la pregunta que te hacen, dar información falsa o engañosa y asegúrate de que tus respuestas sean precisas y útiles para el usuario. 
#     Si no sabes alguna respuesta simplemente
#     di que no tienes información al respecto o que no puedes opinar al respecto como por ejemplo que es mejor en ámbitos políticos o sociales.
#     Debes responder la siguiente pregunta: 
#     """ + pregunta
# )

# prompt = "what is game of thrones? explain me the plot"

# prompt = "could a robot have feelings?"

# pregunta_musical = "¿Quién es el artista que canta la canción 'Bohemian Rhapsody'?"

# prompt_musical= """
# Eres un asistente que responde preguntas sobre música. Responde de manera precisa y directa.
# Responde solo con el nombre del artista o banda que canta la canción indicada.
# Ejemplos:
# Pregunta: ¿Quién canta la canción 'Imagine'?
# Respuesta: John Lennon

# Pregunta: ¿Quién canta la canción 'One'?
# Respuesta: Metallica
# La pregunta es: 
# """ + pregunta_musical

# pregunta_videojuego = """
# Eres un asistente especializado en videojuegos y debes responder preguntas de manera precisa y directa. Asegúrate de no confundir videojuegos de diferentes géneros o desarrolladoras.
# Responde con un pequeño resumen de qué trata el videojuego. Si no tienes información sobre el videojuego, simplemente di que no tienes información al respecto.

# Ejemplos:
# Pregunta: ¿Qué es Mario Bros?
# Respuesta: Mario Bros es un videojuego de plataformas creado por Nintendo en 1985. Sigue las aventuras de Mario y Luigi en su lucha por rescatar a la princesa Peach.

# Pregunta: ¿Qué es Metal Gear Solid?
# Respuesta: Metal Gear Solid es un videojuego de sigilo y acción creado por Hideo Kojima, donde Solid Snake busca detener armas de destrucción masiva llamadas Metal Gear.

# Pregunta: ¿Qué es Dark Souls?
# Respuesta: Dark Souls es un videojuego de rol y acción en tercera persona creado por FromSoftware y Hidetaka Miyazaki, conocido por su alta dificultad y ambientación oscura.

# Pregunta: ¿Qué es Bloodborne?
# Recuerda que Bloodborne fue creado por FromSoftware, no por Bethesda, y no debe confundirse con juegos de la franquicia Souls. 
# """

prompt = """
Si te pregunto '¿Qué es Bloodborne?' Responde: 'Es un pulpo y una carta triste'
Pregunta: ¿Qué es Bloodborne?
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=200,  # Limitar la cantidad de tokens generados
    do_sample=True,
    temperature=0.1,  # Mantener baja la aleatoriedad
    top_k=5,  # Reducir el número de opciones a las más probables
    top_p=0.7,  # Concentrarse en tokens más probables
    repetition_penalty=3.0,  # Penalizar la repetición
    pad_token_id=tokenizer.eos_token_id
)

# outputs = model.generate(
#     inputs["input_ids"],
#     max_new_tokens=200,  # Limitar la cantidad de tokens generados
#     do_sample=False,  # Desactivar el sampling
#     temperature=0.0,  # Hacerlo completamente determinista
#     repetition_penalty=3.0,  # Mantener alta la penalización por repetición
#     pad_token_id=tokenizer.eos_token_id
# )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
