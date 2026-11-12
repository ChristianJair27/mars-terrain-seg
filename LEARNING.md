# Qué aprendimos construyendo marsseg (noviembre de 2026)

Notas en español para Christian. Explican las ideas en el orden en que nos hicieron falta. Las cifras vienen de
nuestras propias ejecuciones (`results/`) o de las fuentes citadas al final.

## 1. Qué es la segmentación semántica

Una red de clasificación mira una foto y dice "esto es arena". Una red de **segmentación semántica** responde
para **cada píxel**: este píxel es suelo, aquel es roca madre, este otro es arena. El resultado es otra imagen
del mismo tamaño en la que cada píxel guarda un número de clase (aquí 0 suelo, 1 roca madre, 2 arena, 3 roca
grande; 255 significa "nadie lo etiquetó").

Usamos **SegFormer** (Xie et al. 2021): un *transformer* que mira la imagen a cuatro escalas (1/4, 1/8,
1/16 y 1/32 del tamaño) y un decodificador ligero que mezcla esas cuatro vistas. Viene preentrenado en ImageNet
(fotos de la Tierra); nosotros sólo lo afinamos con fotos de Marte. Es la misma idea que estudiar física general
antes de especializarse en enanas blancas.

## 2. Por qué le importa esto a un rover

Los rovers de la NASA conducen solos con AutoNav, que hasta Perseverance juzga el terreno sólo por su
**geometría** (estéreo: alturas y pendientes). Swan et al. (2021) recuerdan que el **tipo** de terreno también
decide: Spirit quedó inmovilizado en una trampa de arena, Curiosity estuvo a punto de enterrarse en la arena de
Hidden Valley y sus ruedas de aluminio se perforaron sobre rocas puntiagudas en suelo duro.

Lo de Spirit pasó así, según el JPL: en abril de 2009 sus ruedas izquierdas rompieron una costra en un sitio
llamado "Troy" y se hundieron en arena suelta; ya tenía una rueda averiada desde 2006 y una segunda dejó de
funcionar siete meses después. Tras meses de maniobras, la NASA dejó de intentar liberarlo y dio por terminada
la misión el 25 de mayo de 2011. Un clasificador que reconozca arena en la imagen de navegación es la primera
pieza para no repetirlo.

## 3. El conjunto de datos AI4Mars

- Imágenes NAVCAM de Curiosity (1024×1024 píxeles, en grises), etiquetadas por voluntarios en Zooniverse:
  cada imagen la marcaron unas 10 personas, y un píxel sólo queda etiquetado si al menos 3 coinciden y la clase
  ganadora supera el 65 % (Swan et al. 2021, sección 3.3).
- Un conjunto de prueba aparte, etiquetado por **expertos** (conductores de rovers y científicos de la misión).
  Hay tres versiones: `min1`, `min2`, `min3` = cuántos de los 3 expertos tienen que haber marcado el píxel (y
  todos con la misma clase). `min3` es el más fiable pero el más escaso.
- Máscaras: la del rover (sus propias piezas salen en la foto) y la de distancia (todo lo que está a más de
  30 m se ignora, porque ahí ya no se distinguen bien los detalles).
- "Roca grande" significa más de ~30 cm de alto y ~50 cm de ancho: lo que el rover no debe pisar.

## 4. Cómo se mide: IoU y mIoU

La **exactitud por píxel** (aciertos / total) engaña cuando una clase domina: si el 80 % de los píxeles es suelo,
una red que diga "suelo" en todas partes ya acierta el 80 %. Por eso se usa la **IoU** (*intersection over
union*, índice de Jaccard) por clase:

    IoU(arena) = píxeles que son arena Y la red dijo arena / píxeles que son arena O la red dijo arena

Vale 1 si coinciden exactamente y 0 si no se tocan. Castiga tanto lo que la red no ve (falsos negativos) como lo
que inventa (falsos positivos). La **mIoU** es la media de las cuatro IoU: así la roca grande, que ocupa muy pocos
píxeles, pesa lo mismo que el suelo.

## 5. Saber cuándo la red no sabe: calibración e incertidumbre

Una red da, para cada píxel, una "probabilidad" de cada clase. Que diga 90 % no significa que acierte el 90 % de
las veces: las redes modernas suelen estar **sobreconfiadas** (Guo et al. 2017). Lo medimos con el **ECE**
(*expected calibration error*): se agrupan los píxeles por confianza (por ejemplo, todos los que la red dio con
80-87 %), se mira cuántos acertó de verdad y se promedia la diferencia. ECE = 0 es una red honesta.

Dos arreglos baratos:
- **Escalado de temperatura**: dividir las salidas de la red por un solo número T, ajustado en datos de
  validación. No cambia qué clase gana; sólo hace más humildes (T > 1) o más atrevidas (T < 1) las probabilidades.
- **MC dropout**: dejar encendido el *dropout* (apagar al azar parte de las neuronas) también al predecir y
  repetir la predicción varias veces. Donde las repeticiones discrepan, la red no sabe. La **información mutua**
  entre esas repeticiones mide la parte de la duda que viene del modelo (no del píxel en sí).

## 6. Del mapa de clases al mapa de riesgo

El mapa de riesgo de `marsseg.hazard` pondera cada clase por un coste (suelo 0,1; roca madre 0,3; arena 0,75; roca
grande 1,0) y suma la incertidumbre como coste extra: donde la red duda, un planificador prudente no debería
fiarse. **Los costes son ilustrativos**, ordenados sólo por los peligros que cita el artículo; un modelo de
movilidad de verdad necesita deslizamiento medido, pendientes del estéreo y la geometría de las ruedas.

## 7. Fuera de dominio: Perseverance

Probamos la red, entrenada sólo con Curiosity en el cráter Gale, en fotos NAVCAM recientes de Perseverance en el
cráter Jezero: otra cámara (en color, más campo de visión), otro terreno, otra luz. Es una prueba **cualitativa**
(no hay etiquetas de expertos para esas fotos); sirve para ver dónde sube la incertidumbre, no para dar una cifra.

## Fuentes que leímos

- Swan, R. M. et al. 2021, *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*, CVPR Workshops
  (AI4Space). PDF de openaccess.thecvf.com.
- JPL, *NASA Trapped Mars Rover Finds Evidence of Subsurface Water* (comunicado de 2010): Troy, abril de 2009.
- JPL, página de la misión Spirit (MER-A): fin de la misión el 25 de mayo de 2011.
- Documentación del conjunto `ai4mars-dataset-merged-0.6` (`info.md`, `label_keys.json`), Zenodo 15995036.
