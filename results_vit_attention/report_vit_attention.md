# Visualizing attention maps in Vision Transformers

- Model: `vit_tiny_patch16_224`
- Selected layers: `0,3,6,11`
- Heads: `mean`
- Images processed: `4`
- Attention rollout: `enabled`

Earlier ViT layers usually attend to local texture, edges and small parts of the object. Later layers tend to concentrate more on semantically relevant regions because information has mixed across many transformer blocks. Attention rollout accumulates attention through the network, so it is often smoother and more global than a single-layer CLS-token map.

## Outputs

### ISIC_0000000.jpg

- Top-1 class index: `78`
- Top-1 confidence: `0.4123`
- Original image: `ISIC_0000000/original.png`
- Grid comparison: `ISIC_0000000/attention_grid.png`
- Layer 0 overlay: `ISIC_0000000/layer_00_attention_overlay.png`
- Layer 3 overlay: `ISIC_0000000/layer_03_attention_overlay.png`
- Layer 6 overlay: `ISIC_0000000/layer_06_attention_overlay.png`
- Layer 11 overlay: `ISIC_0000000/layer_11_attention_overlay.png`
- Attention rollout overlay: `ISIC_0000000/attention_rollout_overlay.png`

### ISIC_0000000_segmentation.png

- Top-1 class index: `633`
- Top-1 confidence: `0.0644`
- Original image: `ISIC_0000000_segmentation/original.png`
- Grid comparison: `ISIC_0000000_segmentation/attention_grid.png`
- Layer 0 overlay: `ISIC_0000000_segmentation/layer_00_attention_overlay.png`
- Layer 3 overlay: `ISIC_0000000_segmentation/layer_03_attention_overlay.png`
- Layer 6 overlay: `ISIC_0000000_segmentation/layer_06_attention_overlay.png`
- Layer 11 overlay: `ISIC_0000000_segmentation/layer_11_attention_overlay.png`
- Attention rollout overlay: `ISIC_0000000_segmentation/attention_rollout_overlay.png`

### cars.png

- Top-1 class index: `609`
- Top-1 confidence: `0.7912`
- Original image: `cars/original.png`
- Grid comparison: `cars/attention_grid.png`
- Layer 0 overlay: `cars/layer_00_attention_overlay.png`
- Layer 3 overlay: `cars/layer_03_attention_overlay.png`
- Layer 6 overlay: `cars/layer_06_attention_overlay.png`
- Layer 11 overlay: `cars/layer_11_attention_overlay.png`
- Attention rollout overlay: `cars/attention_rollout_overlay.png`

### cifar10.png

- Top-1 class index: `916`
- Top-1 confidence: `0.3964`
- Original image: `cifar10/original.png`
- Grid comparison: `cifar10/attention_grid.png`
- Layer 0 overlay: `cifar10/layer_00_attention_overlay.png`
- Layer 3 overlay: `cifar10/layer_03_attention_overlay.png`
- Layer 6 overlay: `cifar10/layer_06_attention_overlay.png`
- Layer 11 overlay: `cifar10/layer_11_attention_overlay.png`
- Attention rollout overlay: `cifar10/attention_rollout_overlay.png`
