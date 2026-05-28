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

### Experiment script

The script `src/gender.py` trains and evaluates two CNNs. It keeps the original
dataset loading path compatible with `x_train.npy`, `x_test.npy`, `y_train.npy`
and `y_test.npy`, downloads `gender.tgz` only when those arrays are missing, and
reuses the arrays already present in `notebook/`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Small model, constrained to less than 100K trainable parameters:

```bash
python src/gender.py \
  --model small \
  --epochs 50 \
  --batch-size 64 \
  --out-dir results_gender \
  --augment
```

Strong model:

```bash
python src/gender.py \
  --model strong \
  --epochs 50 \
  --batch-size 64 \
  --out-dir results_gender \
  --augment
```

Both models:

```bash
python src/gender.py --model both --epochs 50 --batch-size 64 --out-dir results_gender --augment
```

Main outputs in `results_gender/`:

- `best_small.keras` and `best_strong.keras`: best checkpoints selected by
  `val_accuracy`.
- `summary.json` and `summary.txt`: final test accuracy, test loss, trainable
  parameter count and total parameter count.
- `confusion_matrix_small.png` and `confusion_matrix_strong.png`: confusion
  matrices.
- `history_small.png` and `history_strong.png`: training and validation curves.
- `classification_report_small.txt` and `classification_report_strong.txt`:
  precision, recall, F1 and support per class.

The `small` model uses separable convolutions, `BatchNormalization` and
`Dropout`. The script checks that it has fewer than `100000` trainable
parameters, which enforces a compact model instead of solving the exercise with
a large network.
    

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

### Image segmentation: U-Net ISIC

The script `src/isic_unet_segmentation.py` implements a simple PyTorch U-Net for
binary melanoma lesion segmentation. It searches recursively for image files
(`jpg`, `jpeg`, `png`) and matching masks. The usual expected naming is:

```text
data/isic_segmentation/
  ISIC_0000000.jpg
  ISIC_0000000_segmentation.png
  ISIC_0000001.jpg
  ISIC_0000001_segmentation.png
```

The files can also be split into folders such as `images/` and `masks/`; mask
filenames or folder names should contain `segmentation`, `mask` or
`groundtruth`.

Example run:

```bash
python src/isic_unet_segmentation.py \
  --data-dir data/isic_segmentation \
  --out-dir results_isic_unet \
  --epochs 30 \
  --batch-size 8 \
  --image-size 256
```

The split is deterministic: 80% train and 20% test, with an optional validation
split from the training part controlled by `--val-ratio` (default `0.1`). Images
and masks are resized to `--image-size`, images are normalized to `[0, 1]`, and
masks are binarized. Training uses synchronized horizontal flips, vertical flips
and small rotations.

Reported test metrics:

- `test_loss`: BCEWithLogitsLoss plus Dice loss.
- `dice`: overlap score `2 * intersection / (prediction + ground truth)`.
- `iou`: Jaccard index, `intersection / union`.
- `pixel_accuracy`: fraction of correctly classified pixels.

Generated files in `results_isic_unet/`:

- `best_unet.pt`: best checkpoint selected by validation Dice.
- `metrics.json` and `metrics.txt`: final test metrics.
- `loss_curve.png` and `dice_curve.png`: training curves.
- `history.json` and `split.json`: reproducibility metadata.
- `examples/example_XXX.png`: test examples with image, ground truth,
  prediction and overlay.

---

## Wide ResNet on CIFAR10

The script `src/cifar_wideresnet.py` implements a simple Wide ResNet for CIFAR10
with TensorFlow/Keras. It uses `keras.datasets.cifar10`, normalizes images to
`[0, 1]`, trains with sparse labels, and uses CUDA automatically when TensorFlow
detects a GPU.

Example run:

```bash
python src/cifar_wideresnet.py \
  --depth 16 \
  --width 4 \
  --epochs 30 \
  --batch-size 128 \
  --augment \
  --out-dir results_cifar_wrn
```

`--depth` follows the Wide ResNet formula `depth = 6N + 4`, so
`N = (depth - 4) // 6` is the number of residual blocks in each stage. `--width`
is the widening factor `k`: the residual stages use `16k`, `32k` and `64k`
filters. If `--dropout` is greater than zero, dropout is applied inside each
residual block between the two convolutions.

Generated files in `results_cifar_wrn/`:

- `best_wrn.keras`: best model selected by validation accuracy.
- `model_summary.txt`: Keras model summary and parameter count.
- `history.json` and `training_log.csv`: training history.
- `training_curves.png`: loss and accuracy curves.
- `confusion_matrix.csv` and `confusion_matrix.png`: CIFAR10 test confusion
  matrix.
- `summary.json` and `summary.txt`: final test accuracy, test loss, best
  validation accuracy, parameter count and run configuration.


## Other project? 

You are welcome!











