import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True

model_name = "microsoft/Phi-3.5-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

pregunta_videojuego = """
Eres un asistente que responde preguntas sobre videojuegos. Responde de manera precisa y directa. Asegúrate de no confundir videojuegos de diferentes géneros o franquicias.
Por ejemplo, Mario Kart es un juego de carreras, mientras que Mario Bros es un juego de plataformas.
Responde con un pequeño resumen de qué trata el videojuego. Si no tienes información sobre el videojuego, simplemente di que no tienes información al respecto.

Ejemplos:
Pregunta: ¿Qué es Mario Bros?
Respuesta: Mario Bros es un videojuego de plataformas creado por Nintendo en 1985. Sigue las aventuras de Mario y Luigi en su lucha por rescatar a la princesa Peach.

Pregunta: ¿Qué es Metal Gear Solid?
Respuesta: Metal Gear Solid es un videojuego de sigilo y acción creado por Hideo Kojima, donde Solid Snake busca detener armas de destrucción masiva llamadas Metal Gear.

Pregunta: ¿Qué es Bloodborne?
"""

inputs = tokenizer(pregunta_videojuego, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.5,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
torch.cuda.empty_cache()

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
