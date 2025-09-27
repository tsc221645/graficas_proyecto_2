# Gráficas por Computadora - Proyecto 2

+ Ana Laura Tschen 221645

### Descripción
El proyecto consiste en elaborar un diorama utilizando cubos texturizados por medio del raytracing. Para elaborarlo se utilizó Rust sin ninguna librería externa para hacer el renderizado. Para poder elaborar el video se utilizó la herramienta de [ffmpeg](https://ffmpeg.org) para transformar los archivos .ppm a un .mp4. 

### Video
[link del video](https://youtu.be/1MC8LG-DyLI)

### Como Ejecutar
Para ejecutar el código se debe clonar el repositorio y correr los siguientes comandos:
+ Para generar los frames desde 0 se ejecuta: ``` cargo run --release ```.
+ Para realizar el video, luego de instalar ffmpeg, se corre el comando: ``` ffmpeg -framerate 30 -i frames/frame_%03d.ppm -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow diorama.mp4 ```

### Nota
+ El renderizado puede tomar bastante tiempo, por lo que se puede reducir la cantidad de frames y ajustar el comando de ffmpeg.
