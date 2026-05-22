# Fisherfaces con ORL

Implementacion manual de Fisherfaces para reconocimiento facial usando PCA + LDA
y vecino mas cercano. El script principal es `fisherfaces_orl.py`.

No usa `sklearn.decomposition.PCA` ni
`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`: la PCA se calcula
diagonalizando la matriz pequena `A A^T` y la LDA se resuelve con
`scipy.linalg.eigh` sobre el problema generalizado `Sb v = lambda Sw v`.

## Estructura de la base ORL

Coloca la base de datos ORL en una carpeta local, por ejemplo:

```text
orl/
  s1/
    1.pgm
    2.pgm
    ...
    10.pgm
  s2/
    1.pgm
    ...
```

Tambien se acepta este formato:

```text
orl/
  1/
    1.pgm
    ...
    10.pgm
  2/
    1.pgm
    ...
```

El split implementado es el solicitado:

- Train: imagenes `1.pgm` a `5.pgm` de cada individuo.
- Test: imagenes `6.pgm` a `10.pgm` de cada individuo.

## Instalacion

Desde esta carpeta:

```bash
python3 -m pip install -r requirements.txt
```

El script usa OpenCV si esta instalado. Si no, usa Pillow para leer y guardar
imagenes.

## Ejecucion

Ejemplo sin redimensionar:

```bash
python3 fisherfaces_orl.py \
  --data-dir ./orl \
  --out-dir results_fisherfaces \
  --lda-dims 1,2,3,5,10,15,20,25,30,35,39 \
  --pca-dim auto \
  --reg 1e-6 \
  --resize 0 \
  --save-faces
```

Ejemplo redimensionando explicitamente a 92x112:

```bash
python3 fisherfaces_orl.py --data-dir ./orl --resize 92x112 --save-faces
```

## Salidas

En el directorio indicado por `--out-dir` se generan:

- `results.csv`: columnas `lda_dim`, `pca_dim`, `accuracy`, `error_rate`,
  `num_errors` y `num_test`.
- `predictions.csv`: predicciones por dimension LDA, ruta de test, etiqueta real,
  etiqueta predicha, distancia al vecino mas cercano y acierto/error.
- `error_curve.png`: tasa de error frente a `d'`.
- `mean_face.png`: cara promedio de entrenamiento.
- `fisherfaces.png`: cuadricula aproximada de las primeras fisherfaces en el
  espacio original si se usa `--save-faces`.

## Consideraciones de estabilidad numerica

- Las imagenes se convierten a escala de grises `float32` en rango `[0, 1]`.
- Los datos se centran restando la media del conjunto de entrenamiento.
- Antes de LDA se aplica PCA para evitar trabajar directamente con matrices
  `d x d` singulares o mal condicionadas.
- Con `--pca-dim auto` se usa `min(n_train - C, d, 150)`, con minimo 1. Si el
  numero de componentes PCA estables es menor, se usa el maximo disponible.
- La dimension LDA se limita a `min(C - 1, pca_dim)`. Para ORL con 40 clases, el
  maximo teorico es 39.
- En LDA se regulariza `Sw` como `Sw + reg * I`, con `reg=1e-6` por defecto. Si
  el solver detecta mala condicion, aumenta la diagonal gradualmente y solo usa
  una solucion con pseudoinversa como fallback.
- Los eigenvectores se ordenan por eigenvalor descendente y se normalizan.
- Si en el fallback aparecen partes complejas residuales debidas a ruido
  numerico, se conserva la parte real solo cuando la componente imaginaria es
  despreciable.
- La clasificacion se realiza con vecino mas cercano y distancia euclidea en el
  espacio PCA + LDA.

---

# Eigenfaces con ORL

Implementacion manual de Eigenfaces para reconocimiento facial usando PCA y vecino mas cercano.

El script principal es `eigenfaces_orl.py`. No usa `sklearn.decomposition.PCA`: la PCA se calcula diagonalizando la matriz pequena `A A^T`, adecuada cuando el numero de imagenes de entrenamiento es mucho menor que el numero de pixeles.

## Estructura de la base ORL

Coloca la base de datos ORL en una carpeta local, por ejemplo:

```text
orl/
  s1/
    1.pgm
    2.pgm
    ...
    10.pgm
  s2/
    1.pgm
    ...
```

Tambien se acepta este formato:

```text
orl/
  1/
    1.pgm
    ...
    10.pgm
  2/
    1.pgm
    ...
```

El split implementado es el solicitado:

- Train: imagenes `1.pgm` a `5.pgm` de cada individuo.
- Test: imagenes `6.pgm` a `10.pgm` de cada individuo.

## Instalacion

Desde esta carpeta:

```bash
python3 -m pip install -r requirements.txt
```

El script usa OpenCV si esta instalado. Si no, usa Pillow para leer y guardar imagenes.

## Ejecucion

Ejemplo sin redimensionar:

```bash
python3 eigenfaces_orl.py \
  --data-dir ./orl \
  --out-dir results_eigenfaces \
  --dims 1,2,3,5,10,15,20,30,40,60,80,100 \
  --resize 0 \
  --save-faces
```

Ejemplo redimensionando explicitamente a 92x112:

```bash
python3 eigenfaces_orl.py --data-dir ./orl --resize 92x112 --save-faces
```

## Salidas

En el directorio indicado por `--out-dir` se generan:

- `results.csv`: columnas `k`, `accuracy`, `error_rate`, `num_errors`, `num_test`.
- `predictions.csv`: predicciones por dimension `k`, ruta de test, etiqueta real, etiqueta predicha y distancia al vecino mas cercano.
- `error_curve.png`: tasa de error frente a `d'`.
- `mean_face.png`: cara promedio.
- `eigenfaces.png`: cuadricula de las primeras eigenfaces si se usa `--save-faces`.

## Consideraciones practicas incluidas

- Las imagenes se convierten a escala de grises `float32` en rango `[0, 1]`.
- No se diagonaliza la covarianza grande `d x d`; se diagonaliza la matriz pequena `n x n`.
- Los eigenvalores se ordenan de mayor a menor.
- Se descartan eigenvectores asociados a eigenvalores numericamente pequenos.
- Cada eigenvector reconstruido en el espacio de pixeles se normaliza a modulo uno.
- La clasificacion se realiza con vecino mas cercano y distancia euclidea en el espacio proyectado.
