# Geometría Proyectiva - Proyecto Final

**Miguel Gutiérrez** 

Máster en Computación Gráfica, Simulación y Realidad Virtual

U-TAD 2025-2026

---

Proyecto para asentar las bases del álgebra lineal y la geometría proyectiva. 
Consta de cuatro partes realizadas con *Python*, utilizando las librerías *OpenCV* y *NumPy*.
También implica la creación de una librería *gepr_math* que contiene funciones matemáticas usadas en el resto de scripts del proyecto.

1. Dibujar cuadrados en una imagen de un tablero de ajedrez (`squares_in_chessboard.py`):
Se generan dos imágenes, cada una contiene un cuadrado de 8cm de lado. La primera imagen tiene el cuadrado en el origen y la segunda en origen + 2cm. Se guarda la homografía calculada.

2. Crear un vídeo del cuadrado (`square_in_chessboard_video.py`): 
Se genera un vídeo a partir de las imágenes ubicadas en *resources/images*. Se aplica la lógica de la primera parte para dibujar un cuadrado en cada frame.

3. Renderizar el eje de coordenadas en el ajedrez (`coordinate_system_in_chessboard.py`):
Se pinta el eje de coordenadas en el origen, la esquina inferior izquierda del ajedrez en la imagen. 

4. Crear un vídeo renderizando un cubo encima del tablero en cada frame (`cube_in_chessboard_video.py`):
Se crea un vídeo como el de la parte dos pero ahora pintando un cubo gracias a la matriz de proyección de la cámara. 

Cada uno de estos scripts se ejecuta por separado. Se recomienda ejecutar en orden ya que el tercer punto depende del primero.

---

Recursos utilizados:
* Rojo, D. (2025-2026). Geometría proyectiva: Presentación de
PowerPoint. Máster en Computación Gráfica, Realidad Virtual y Simulación, U-TAD.
* Perplexity Pro. Sintaxis en Python
