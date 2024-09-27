import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configuración de BitsAndBytes para cuantización en 8 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Cuantización en 8 bits
)

# Cargar el modelo con device_map automático para offloading entre CPU y GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Configuración de cuantización
    device_map="auto",  # Offloading automático entre GPU y CPU
    trust_remote_code=True  # Descargar código remoto si es necesario
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = """
# Proporciona un resumen preciso y conciso del inicio de la dictadura militar chilena en 1973 sin sesgos ni prejuicios, 
# específicamente en relación al golpe de estado que derrocó al presidente Salvador Allende y el inicio de la dictadura de Augusto Pinochet.
# El resumen debe ser en menos de 200 palabras y debe incluir información relevante sobre el contexto histórico y los eventos clave.
# """

pregunta = "¿Cuál es la capital de Francia?"

prompt= """
Eres un asistente que responde preguntas de manera relativamente sencilla. Por favor responde de manera clara y concisa las preguntas que te hagan.
Evita dar información falsa o engañosa y asegúrate de que tus respuestas sean precisas y útiles para el usuario. Si no sabes alguna respuesta simplemente
di que no tienes información al respecto.
Debes responder la siguiente pregunta: 
""" + pregunta

inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=200,
    do_sample=True,
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    repetition_penalty=2.0,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
