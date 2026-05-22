# Ejercicio de reconocimiento facial con Fisherfaces

El script `fisherfaces_orl.py` implementa Fisherfaces manualmente para ORL usando
PCA antes de LDA, vecino mas cercano y curva de error variando `d'`. No usa PCA
ni LDA de `sklearn`. La implementacion esta en `biometria/fisherfaces_orl.py` y
el script de la raiz actua como lanzador compatible con la estructura pedida.

Ejemplo:

```bash
python3 fisherfaces_orl.py \
  --data-dir ./biometria/orl \
  --out-dir results_fisherfaces \
  --lda-dims 1,2,3,5,10,15,20,25,30,35,39 \
  --pca-dim auto \
  --reg 1e-6 \
  --resize 0 \
  --save-faces
```

El README especifico con el formato ORL, salidas y decisiones de estabilidad
numerica esta en `biometria/README.md`.

---

# Ejercicio de verificacion biometrica con ROC

Este repositorio incluye un script sencillo en Python para evaluar un sistema de
verificacion biometrica a partir de dos conjuntos de scores: scores de clientes
genuinos y scores de impostores.

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

Con ficheros propios:

```bash
python roc_metrics.py \
  --clients path/to/client_scores.txt \
  --impostors path/to/impostor_scores.txt \
  --fn-x 0.1 \
  --fp-x 0.05 \
  --out-dir results_roc
```

Ejemplo integrado:

```bash
python roc_metrics.py --demo --out-dir results_demo
```

Los ficheros de entrada pueden contener scores separados por espacios, saltos de
linea, comas o punto y coma.

## Regla de decision

Para cada score y umbral:

- `score >= threshold`: se acepta como cliente.
- `score < threshold`: se rechaza.

En clientes genuinos, aceptar es `VP` y rechazar es `FN`. En impostores,
aceptar es `FP` y rechazar es `VN`.

## Metricas

- `C = VP + FN`: numero total de clientes.
- `I = VN + FP`: numero total de impostores.
- `FNR = FN / C`: tasa de falsos negativos.
- `FPR = FP / I`: tasa de falsos positivos.
- `TPR` o sensibilidad `S = VP / C = 1 - FNR`.
- `TNR` o especificidad `E = VN / I = 1 - FPR`.
- `AUC`: area bajo la curva ROC, calculada manualmente con integracion
  trapezoidal tras ordenar los puntos por `FPR` creciente.
- `EER aproximado`: punto donde `abs(FPR - FNR)` es minimo.

## Curva ROC

La curva ROC representa `FPR` en el eje X y sensibilidad `TPR = 1 - FNR` en el
eje Y. El script prueba como umbrales todos los scores de clientes e impostores,
anadiendo tambien un umbral menor que el minimo y otro mayor que el maximo para
cubrir los extremos. Si todos los scores estan entre 0 y 1, incluye ademas los
umbrales `0.0` y `1.0`.

Una curva mas cercana a la esquina superior izquierda indica mejor separacion
entre clientes e impostores. La diagonal representa comportamiento aleatorio.

## D-Prime

El script calcula:

```text
d_prime = (mu_clientes - mu_impostores) / sqrt(var_clientes + var_impostores)
```

`mu_clientes` y `var_clientes` se calculan sobre los scores de clientes.
`mu_impostores` y `var_impostores` se calculan sobre los scores de impostores.
La varianza usada por defecto es varianza poblacional, equivalente a `numpy.var`
con `ddof=0`.

Un `D-Prime` mayor indica mayor separacion entre las distribuciones de clientes e
impostores.

## Ficheros de salida

El directorio indicado con `--out-dir` contiene:

- `roc_points.csv`: tabla con `threshold`, `VP`, `VN`, `FP`, `FN`, `TPR`, `S`,
  `FPR`, `FNR`, `TNR` y `E`.
- `roc_curve.png`: grafica de la curva ROC.
- `report.txt`: informe legible con `AUC`, `D-Prime`, `FP(FN ~= X)`,
  `FN(FP ~= X)` y `EER aproximado`.
- `report.json`: el mismo informe en formato estructurado.

Para `FP(FN = X)`, el informe incluye el punto cuyo `FNR` es mas cercano a `X` y
tambien el mejor punto que cumple `FNR <= X`, minimizando `FPR`. Para
`FN(FP = X)`, incluye el punto cuyo `FPR` es mas cercano a `X` y tambien el
mejor punto que cumple `FPR <= X`, minimizando `FNR`.

---

# Ejercicio de normalizacion local de iluminacion facial

El script `local_light.py` implementa normalizacion y ecualizacion de intensidad
en escala de grises para imagenes faciales ya alineadas en traslacion, escala y
rotacion. El objetivo es corregir variaciones de iluminacion sin usar modelos
externos.

## Formulas implementadas

Normalizacion global:

```text
f(I(x,y)) = (I(x,y) - mu_I) / sigma_I
```

Ecualizacion global:

```text
f(I(x,y)) = ((G - 1) / (M * N)) * H(I(x,y))
```

Normalizacion local:

```text
f(w(x,y)) = (w(x,y) - mu_w) / sigma_w
```

La media y la varianza de cada ventana se calculan con imagenes integrales de
`I` e `I^2`. Si las ventanas se solapan, cada pixel acumula las transformaciones
obtenidas en todas las ventanas donde aparece y se guarda el promedio.

Ecualizacion local:

```text
f(w(x,y)) = ((G - 1) / (m * n)) * H_w(w(x,y))
```

`H_w` es el histograma acumulado de la ventana local. La implementacion incluye
un clipping opcional tipo CLAHE: con `--clip-limit > 0`, el histograma local se
recorta a un maximo relativo y el exceso se redistribuye uniformemente.

## Uso

Instalacion:

```bash
pip install -r requirements.txt
```

Ejecutar todos los metodos sobre una imagen:

```bash
python local_light.py \
  --input images/face.png \
  --out-dir results_light \
  --method all \
  --window-size 15 \
  --stride 1 \
  --clip-limit 0 \
  --save-comparison
```

Procesar un directorio:

```bash
python local_light.py --input path/to/images --out-dir results_light --method local_histeq
```

Formatos soportados: `png`, `jpg`, `jpeg`, `bmp` y `pgm`.

## Salidas

El directorio de resultados contiene una imagen procesada por cada metodo, una
figura comparativa si se usa `--save-comparison`, y `summary.csv` con nombre de
imagen, metodo, tamano de ventana, `stride`, tiempo de ejecucion y ruta de
salida.

---

# Computer Vision Lab (up to 10 points)

## Basic implementations

Check basic implementations on CIFAR10 in the Deep Learning Lab project [here](https://github.com/RParedesPalacios/DeepLearningLab/tree/master/CIFAR/Keras)

![Cifar10](images/cifar10.png)

**Goals:**

* Implement some basic convolutional networks
* Implement different data augmentation
* Implement VGG model

---

## Advanced topologies 

* Wide Resnet  (1 point) 

* Dense Nets   (1 point)


---

## Gender Recognition (3 point)

Images from "Labeled Faces in the Wild" dataset (LFW) in realistic scenarios, poses and gestures. Faces are automatically detected and cropped to 100x100 pixels RGB.


![Face example](images/face.png)


**Training** set: 10585 images

**Test** set: 2648 images 


**Python Notebook**: [here](notebook/gender.ipynb)

**Python code**: [here](src/gender.py)

**Goals:**
* Implement a model with >98% accuracy over test set
* Implement a model with >95% accuracy with less than 100K parameters
  
  get some inspiration from [Paper](https://pdfs.semanticscholar.org/d0eb/3fd1b1750242f3bb39ce9ac27fc8cc7c5af0.pdf)
    

---

## Car Model identification with bi-linear models (5 points)

Images of 20 different models of cars.

![Cars](images/cars.png)

**Training** set: 791 images

**Test** set: 784 images 

* Version 1. Two different CNNs:

  **Python code**: [here](src/cars1.py)

* Version 2. The same CNN (potentially a pre-trained model)

  **Python code**: [here](src/cars2.py)

**Goals:**
* Understand the above Keras implementations:
  * Name the layers
  * Built several models
  * Understand tensors sizes
  * Connect models with operations (outproduct)
  * Create an image generator that returns a list of tensors
  * Create a data flow with multiple inputs for the model

**Suggestion:**
  * Load a pre-trained VGG16, Resnet... model 
  * Connect this pre-trained model and form a bi-linear
  * Train freezing weights first, unfreeze after some epochs, very low learning rate
  * Accuracy >65% is expected 
  
  
[Paper](https://pdfs.semanticscholar.org/3a30/7b7e2e742dd71b6d1ca7fde7454f9ebd2811.pdf)

--------------------------------
## Image classification with transformers vs. CNN (5 puntos)

**Global objective**: Compare classification performance of finetuned vision transformers vs CNN

**Task descritpion**: 
  * Download and setup one of the proposed datasets
  * Finetune a simple CNN  (ResNet, EfficientNet, or MobileNet) for baseline comparison (free choice of CNN)
  * Finetune a Vision transformer (ViT, Swin, Maxvit) of free choice using timm or huggingface. Please select a model that fits your computational capabilities
  * Compare: accuracy, training speed, inference time. 

**Datasets**:
Students should choose between one of these datasets:

  * flowers-102 (Oxford 102 Category Flowers)
    * Fine-grained classification with 102 flower species ssmall enough to train within a reasonable time.
    * Size: 8,189 images, 102 classes.
    * Difficulty: Small inter-class variability, small dataset (risk of overfitting).
    * Dataset Link: [Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102)
  * Stanford Cars
    * Middle size dataset with high inter-class similarity
    * Size: ~16000 images, 196 calses
    * Difficulty: requires attention to details, making a good test to compare vit and CNNs
    * Dataset Link: [Available in torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html) (see instructions for download there)
      
**Results**:
  * Check literature to know expected accuracy
  * Organize results clearly, effect of learning rate, batch size, scheduing, freezing of layers....
  * Having competitive classification results will be a plus
  * It will be also a plus if different transformer architectures are compared (for instance comparison of ViT models size)

--------------------------------
## Visualizing attention maps in vision transformers (3 puntos)

**Global objective**: 

Implement a visualization pipeline to display attention maps from different layers of a ViT

**Task description**:
  * Use a pretrained ViT either from timm or Huggingface
  * Get some pictures compatible with any imagenet class (car, dog, cat...)
  * Write code to extract attention maps from the ViT model
  * Pass images through the model and extract attention weights
  * Write code to display attention maps overlayed on the imput image to see important regions for the ViT.
  * Compare and analyze attention maps of different layers and different images.

**Notes**:
  * The specific code for extracting attention maps may change depending on the specific implementation of the model
  * You can try and discuss different visualization techniques to better understand the attention patterns. For instance, visualize attention maps of each head or fusing attention from several heads of the same layer.

**Extras**:
  * Implement attention rollout and compare with atention of individual layers
  * Compare attention maps with activation maps (gradcam) of a pretrained CNN


---------------------------------

## Image colorization (3 point)

![Cars](images/color.png)

Code extracted and adapted from [github](https://github.com/emilwallner/Coloring-greyscale-images-in-Keras)

**Goals:**

* Understand the above Keras implementations:
  * How to load the inception net 
  * How to merge encoder and inception result


**Python code**: [here](src/colorization.py)


Need help? [Read](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/)


## Image segmentation (4 points)

ISIC Melanoma Segmentation

![Image](images/ISIC_0000000.jpg)
![Mask](images/ISIC_0000000_segmentation.png)

Dataset available here:

[ISIC_Melanoma](https://www.dropbox.com/scl/fi/8v8isahqorzdq3h7rjv98/isic_segmentation.zip?rlkey=2ua7dv4ueioj68mn6466x46zq&st=anjrmz9j&dl=0)

Exercise: 

* Implement a UNET for this task.
* Split data 80% training 20% test
* Get results over test set 


## Other project? 

You are welcome!

















