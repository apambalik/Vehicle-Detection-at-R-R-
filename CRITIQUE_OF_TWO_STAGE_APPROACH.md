Now I have a comprehensive understanding of your project. Let me compile my analysis.

---

## Analysis of Your Current State

Based on my exploration of your codebase, here is what I found:

**Baseline (regular RF-DETR from `results.json`):**
- mAP@50:95 = **0.521**, mAP@50 = **0.811**
- Weak classes: SUV (0.418), Motorcycle (0.352)
- Strong classes: Truck (0.620), Bus (0.596), Van (0.567)

**Approach 1 (Hyperbolic head replacement) from `hyperbolic_rfdetr.ipynb`:**
- 30 epochs of training, best mAP@50:95 peaked around **0.538** (epoch ~24), mAP@50 around **0.828**
- Final epoch 29: mAP@50:95 = **0.505**, mAP@50 = **0.785** (declining/oscillating)
- This is only a marginal improvement over baseline, and the model appears to be overfitting in later epochs

**Hyperbolicity testing from `test_hyperbolicity.ipynb`:**
- Decoder features: relative delta = **0.389** (moderately hyperbolic)
- Cropped VGG features: relative delta = **0.254** (more hyperbolic)
- Class-level centroids: relative delta = **0.215**

---

## How to Implement the Two-Stage Approach

Here is a concrete implementation plan:

### Stage 1: RF-DETR Detection (unchanged)

Keep your existing trained RF-DETR model (the baseline `checkpoint_best_regular.pth`) for bounding box detection only. At inference, run it to get bounding boxes with confidence > some threshold, but **ignore** its class predictions.

### Stage 2: Hyperbolic Crop Classifier

**Step 1 -- Build a crop dataset:**

```python
import os, json, cv2
from PIL import Image
from pathlib import Path

def build_crop_dataset(coco_json_path, images_dir, output_dir):
    """Extract vehicle crops from COCO annotations into class folders."""
    with open(coco_json_path) as f:
        ann = json.load(f)
    
    class_names = {c["id"]: c["name"] for c in ann["categories"]}
    
    for cls_name in class_names.values():
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)
    
    img_lookup = {img["id"]: img for img in ann["images"]}
    
    for i, a in enumerate(ann["annotations"]):
        img_info = img_lookup[a["image_id"]]
        img = Image.open(os.path.join(images_dir, img_info["file_name"])).convert("RGB")
        
        x, y, w, h = a["bbox"]
        if w < 15 or h < 15:
            continue
        
        crop = img.crop((x, y, x + w, y + h))
        cls_name = class_names[a["category_id"]]
        crop.save(os.path.join(output_dir, cls_name, f"crop_{i:06d}.jpg"))
```

**Step 2 -- Build a lightweight hyperbolic classifier:**

This is where the actual hyperbolic approach lives. Rather than a ProtoNet (which is a few-shot method you don't actually need since you have thousands of annotations), use a simple CNN backbone + HyperbolicMLR:

```python
import torch
import torch.nn as nn
import torchvision.models as models
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.nn import HyperbolicMLR
from hypll.tensors import TangentTensor

class HyperbolicVehicleClassifier(nn.Module):
    def __init__(self, num_classes=7, embed_dim=128, curvature=1.0):
        super().__init__()
        # Lightweight backbone (MobileNetV3-Small for speed)
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        backbone.classifier = nn.Identity()
        self.backbone = backbone  # outputs 576-d
        
        self.manifold = PoincareBall(c=Curvature(value=curvature, requires_grad=False))
        
        # Project to embedding space
        self.proj = nn.Sequential(
            nn.Linear(576, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Hyperbolic classification
        self.classifier = HyperbolicMLR(embed_dim, num_classes, self.manifold)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.proj(features)
        # Map to tangent space at origin, then expmap inside HyperbolicMLR
        tangent = TangentTensor(embeddings, man_dim=-1, manifold=self.manifold)
        logits = self.classifier(tangent)
        return logits
```

**Step 3 -- Train the classifier:**

```python
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from hypll.optim import RiemannianAdam

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder("crops/train", transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

model = HyperbolicVehicleClassifier(num_classes=7)
optimizer = RiemannianAdam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for images, labels in train_loader:
        logits = model(images.cuda())
        loss = criterion(logits, labels.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Step 4 -- Inference pipeline:**

```python
class TwoStagePipeline:
    def __init__(self, detector, classifier, class_names, conf_threshold=0.3):
        self.detector = detector
        self.classifier = classifier
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image):
        # Stage 1: detect boxes
        detections = self.detector.predict(image, threshold=self.conf_threshold)
        
        # Stage 2: classify each crop
        results = []
        pil_img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        
        crops = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = pil_img.crop((x1, y1, x2, y2))
            crops.append(self.transform(crop))
        
        if crops:
            batch = torch.stack(crops).cuda()
            with torch.no_grad():
                logits = self.classifier(batch)
                pred_classes = logits.argmax(dim=-1)
            
            for i, (box, cls_idx) in enumerate(zip(detections.xyxy, pred_classes)):
                results.append({
                    "bbox": box,
                    "class": self.class_names[cls_idx.item()],
                    "confidence": detections.confidence[i],
                })
        
        return results
```

---

## Critique of the Two-Stage Approach

### Fundamental Problems

**1. Your data doesn't strongly justify hyperbolic methods.**

Your delta-hyperbolicity test returned **0.389** for decoder features and **0.254** for VGG crops. These are moderate values. The literature (Khrulkov et al., 2020) found that hyperbolic embeddings shine when delta < 0.1 (highly tree-like data). Your vehicle classes have a shallow, roughly 2-level hierarchy:

```
Vehicle
├── Passenger: Sedan, SUV, Pickup
├── Commercial: Van, Truck, Bus
└── Two-wheel: Motorcycle
```

This is barely a hierarchy -- it is essentially a flat 7-class problem with mild inter-class similarity. Hyperbolic geometry provides the biggest advantage for deep hierarchies (WordNet with 100+ levels, taxonomies with hundreds of classes at varying granularities). For 7 flat classes, Euclidean space works just fine.

**2. The real classification problem is visual similarity, not hierarchy.**

Looking at your results, the confused classes (SUV vs. Sedan vs. Pickup, mAP@50:95 of 0.42-0.56) are hard because they look alike in surveillance footage -- not because the model fails to capture hierarchical relationships. Hyperbolic embeddings won't solve appearance-level confusion. You need:
- Better data augmentation
- Higher resolution crops  
- Attention to viewpoint, occlusion, and scale variation

**3. Two-stage pipeline latency is a real cost for "real-time" detection.**

You mentioned this in your pros/cons, but consider the math. If RF-DETR runs at ~ 30fps and a scene has 10-20 vehicles, running even MobileNet-V3-Small on 20 crops at 224x224 adds \~15-25ms, bringing you from ~ 33ms to ~ 55ms per frame (~ 18fps). This is a significant drop for a traffic monitoring scenario. Batching the crops helps, but GPU memory becomes a constraint.

**4. COCO mAP metric couples localization and classification -- you can't decouple them neatly.**

The two-stage approach only improves classification after Stage 1 provides boxes. But COCO mAP penalizes both localization and classification errors together. If Stage 1 already assigns a wrong class to a box (which affects NMS), reassigning post-hoc can actually harm results because:
- NMS in Stage 1 may suppress correct detections based on the (wrong) class scores
- The detector's confidence scores (used for precision-recall computation) come from Stage 1, not your Stage 2 classifier

**5. ProtoNet is the wrong tool here.**

ProtoNet is designed for **few-shot learning** (1-5 examples per class). You have thousands of annotations per class (Sedan: 10,840, Pickup: 11,049). With this much data, a standard classifier with cross-entropy loss will outperform few-shot methods. ProtoNet sacrifices accuracy in the high-data regime for its generalization to new classes with few examples -- that's not your scenario.

### What Would Actually Help

Given your results, here are more impactful alternatives:

1. **Class-balanced sampling / focal loss tuning.** Your dataset is heavily imbalanced: Bus has 115 annotations vs. Sedan's 10,840 (94x ratio). This explains poor Bus and Motorcycle performance far better than any geometry argument. Try adjusting the class frequency weighting in the focal loss.

2. **More training data for rare classes.** 115 bus annotations is extremely few. Data augmentation (Mosaic, MixUp) targeted at rare classes, or sourcing more bus/motorcycle images, would likely yield a bigger boost than any architectural change.

3. **Longer training with cosine LR schedule.** Your Approach 1 used a constant LR (lr_drop=100 with 30 epochs means no drop ever happened). The model appeared to be still improving around epoch 12-15 and then oscillated. A cosine schedule or a proper LR drop at epoch 20 could stabilize convergence.

4. **If you still want to try the two-stage approach**, use a standard EfficientNet-B0 or MobileNet classifier with cross-entropy loss on the crops. Compare that to the hyperbolic version. My prediction is the Euclidean classifier will perform nearly identically, because the bottleneck is visual similarity, not geometric representation.

5. **Approach 1 (hyperbolic head) showed only marginal gains.** Your best was 0.538 vs. baseline 0.521. This is within noise range for a 30-epoch run and suggests the Poincare ball geometry isn't contributing meaningful discriminative power for this specific problem.

### Bottom Line

The two-stage approach is implementable (code above), but the hypothesis that hyperbolic geometry will meaningfully improve 7-class vehicle classification is weakly supported by your own empirical evidence. Your delta-hyperbolicity values are moderate, the class hierarchy is shallow, and Approach 1 showed only marginal improvement. Before investing in this complex pipeline, I'd recommend first addressing the class imbalance problem and training schedule, which are likely the dominant factors in your current performance gap.