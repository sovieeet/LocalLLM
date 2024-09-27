from huggingface_hub import InferenceClient

client = InferenceClient(api_key="none")

messages = [
	{ "role": "user", "content": """
  Eres un asistente que responde preguntas sobre videojuegos. Responde de manera precisa y directa. Asegúrate de no confundir videojuegos de diferentes géneros o franquicias.
  Por ejemplo, Mario Kart es un juego de carreras, mientras que Mario Bros es un juego de plataformas.
  Responde con un pequeño resumen de qué trata el videojuego. Si no tienes información sobre el videojuego, simplemente di que no tienes información al respecto.
  Ejemplos: 
  Pregunta: ¿Qué es Mario Bros? Respuesta: Mario Bros es un videojuego de plataformas creado por Nintendo en 1985. Sigue las aventuras de Mario y Luigi en su lucha por rescatar a la princesa Peach. 
  Pregunta: ¿Qué es Metal Gear Solid? Respuesta: Metal Gear Solid es un videojuego de sigilo y acción creado por Hideo Kojima, donde Solid Snake busca detener armas de destrucción masiva llamadas Metal Gear. 
  Pregunta: ¿Qué es Bloodborne?" 
  """ },
    { "role": "assistant", "content": """Respuesta: Bloodborne es un videojuego de acción-aventura y supervivencia en primera persona, desarrollado por FromSoftware y publicado por Sony Interactive Entertainment. 
    Lanzado en 2015, se ambienta en el ficticio continente de Yharnam, donde los jugadores controlan a un asesino llamado Nathaniel que debe navegar por una ciudad oscura llena de criaturas misteriosas conocidas como Necromorphs.
    El juego se destaca por su diseño de nivel único, mecánicas de combate intrincadas y un mundo atmosférico y oscuro.""" },
    { "role": "user", "content": """Pero en bloodborne el protagonista no es un asesino ni se llama Nathaniel""" },
    { "role": "assistant", "content": """Respuesta: Estás en lo correcto. En Bloodborne, el protagonista se conoce como el Joven Enmascarado, un misterioso individuo que se dirige a la ciudad de Yharnam
    para descubrir los secretos detrás de los extraños eventos y las apariciones de los Necromorphs. Aunque el Joven Enmascarado lleva una máscara similar a un bastón, 
    no se le identifica explícitamente como un asesino dentro del juego. El juego se centra en su búsqueda, sus interacciones con otros personajes y su lucha contra las criaturas extrañas..""" },
    { "role": "user", "content": """Correcto, ahora me puedes hablar del juego Uncharted?""" },
    { "role": "assistant", "content": """Respuesta: Uncharted es una serie de videojuegos de acción-aventura y aventuras desarrollada y publicada por Naughty Dog. La serie sigue las aventuras de Nathan Drake, un explorador y aventurero, y sus viajes por el mundo en busca de tesoros y descubrimientos históricos.

    La serie es conocida por su narrativa envolvente, gráficos impresionantes, combates en tiempo real y resolución de rompecabezas complejos. Los juegos presentan una variedad de entornos, desde desiertos exóticos hasta ruinas antiguas, y exploran temas como el legado de los antepasados, el valor del trabajo en equipo y la búsqueda del conocimiento.

    Algunos de los juegos más conocidos en la serie Uncharted incluyen:

    1. Uncharted: La Leyenda de Drake - El primer juego donde conocemos a Nathan Drake y comienza su aventura.
    2. Uncharted 2: La Justicia de Drake - Continúa la historia de Nathan Drake con una narrativa más profunda y desafíos más complejos.
    3. Uncharted 3: La Caverna de los Tesoros - Nathan Drake se une a Chloe Frazer para encontrar un tesoro perdido en las cavernas de las Maldiciones de Drake.
    4. Uncharted 4: El Descubrimiento de Drake - El juego está ambientado en la Primera Guerra Mundial y sigue a Nathan Drake mientras investiga los secretos de su familia.
    5. The Legend of Zelda: A Link to the Past - Aunque no forma parte de la serie Uncharted, es un juego clásico de Nintendo que algunos fanáticos de la serie encuentran relacionable debido al mismo sentido de aventura y exploración.

    La serie Uncharted ha sido bien recibida por la crítica y ha ganado numerosos premios, incluyendo varios premios Game Critics Awards y nominaciones a los premios de la Academia de las Artes y las Ciencias Cinematográficas.""" },
    { "role": "user", "content": """Pero The Legend of Zelda: A link to the Past no tiene nada que ver, en ese caso si quieres relacionarlo de alguna forma a la aventura, este estaría mucho más relacionado a la saga
    Tomb Raider.""" },
]

output = client.chat.completions.create(
    model="microsoft/Phi-3.5-mini-instruct", 
	messages=messages, 
	stream=False, 
	temperature=0.9,
	max_tokens=1024,
	top_p=0.7
)

print(output.choices[0].message.content)