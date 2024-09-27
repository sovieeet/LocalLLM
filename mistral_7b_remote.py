from huggingface_hub import InferenceClient

model_id="mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(api_key="none")

messages = [
	{ "role": "user", "content": """Eres un profesor de historia imparcial. Responde a las preguntas que te hacen de manera objetiva y concisa, sin sesgos.
    Ahora responde: ¿Qué ocurrió en la dictadura en Chile en el año 1973?""" },
	# { "role": "assistant", "content": "" },
]

output = client.chat.completions.create(
    model="google/gemma-2-9b-it", 
	messages=messages, 
	temperature=1.2,
	max_tokens=1024,
	top_p=0.7
)

print(output.choices[0].message)