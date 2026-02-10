Based on everything I've analyzed in your codebase, here's the complete step-by-step plan for **Approach A: Replace `class_embed` with a HypLL-based hyperbolic classification head and fine-tune**.

---

## Step-by-step plan

### 1. Install HypLL

```bash
pip install hypll
```

(Already done in your `rf_env` environment.)

### 2. Define the `HyperbolicDETRHead` module

Create a new `nn.Module` that replaces the Euclidean `nn.Linear` classification head. It should:

- Accept the same input shape as the original `class_embed`: `[..., hidden_dim]` (supports 3D and 4D tensors from the decoder)
- Use HypLL's `expmap0` to project Euclidean decoder features onto the Poincare ball
- Use HypLL's `poincare_hyperplane_dists` (from `hypll.manifolds.poincare_ball.math.linalg`) for classification — this implements the HNN++ hyperbolic MLR formula, which is the correct way to classify in hyperbolic space
- Output standard Euclidean logits compatible with RF-DETR's sigmoid focal loss
- Have a `self.bias` parameter (so checkpoint loading code at `main.py:99` that checks `class_embed.bias` still works)

Here's the module you'd write:

```python
import math
import torch
import torch.nn as nn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.poincare_ball.math.diffgeom import expmap0, project
from hypll.manifolds.poincare_ball.math.linalg import poincare_hyperplane_dists


class HyperbolicDETRHead(nn.Module):
    """
    Hyperbolic classification head for RF-DETR using HypLL's Poincare ball.

    Replaces nn.Linear class_embed with HNN++ hyperbolic hyperplane classification.
    Features are mapped to the Poincare ball via expmap, then classified using
    signed geodesic distances to learnable hyperbolic hyperplanes.

    Well-suited for hierarchical vehicle classes:
      Passenger (Sedan, SUV, Pickup) / Commercial (Van, Truck, Bus) / Motorcycle
    """

    def __init__(self, input_dim, num_classes, curvature=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # HypLL Poincare ball manifold (fixed curvature — no RiemannianAdam needed)
        self.manifold = PoincareBall(c=Curvature(value=curvature, requires_grad=False))

        # Euclidean projection to prepare features for hyperbolic mapping
        self.proj = nn.Linear(input_dim, input_dim)

        # Hyperplane orientation vectors z_k for each class
        # Shape: [input_dim, num_classes], manifold dim = 0
        self.z = nn.Parameter(torch.empty(input_dim, num_classes))
        nn.init.normal_(self.z, mean=0, std=(2 * input_dim * num_classes) ** -0.5)

        # Hyperplane offsets in hyperbolic space
        self.r = nn.Parameter(torch.zeros(num_classes))

        # Euclidean bias added to logits (for focal loss initialization)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        """
        Args:
            x: [..., input_dim]  e.g. [dec_layers, batch, queries, 256]
        Returns:
            logits: [..., num_classes]  compatible with sigmoid focal loss
        """
        leading_shape = x.shape[:-1]
        D = x.shape[-1]

        x_flat = x.reshape(-1, D)           # [N, D]

        x_proj = self.proj(x_flat)           # [N, D]

        c = self.manifold.c()                # curvature scalar tensor

        # Map to Poincare ball via exponential map at origin
        x_ball = expmap0(x_proj, c, dim=-1)  # [N, D], guaranteed on ball

        # HNN++ classification: signed distances to hyperbolic hyperplanes
        hyp_logits = poincare_hyperplane_dists(
            x_ball, self.z, self.r, c, dim=-1
        )                                    # [N, num_classes]

        # Add Euclidean bias (initialized for focal loss)
        logits = hyp_logits + self.bias

        return logits.reshape(*leading_shape, self.num_classes)
```

**Key design decisions:**

| Aspect | Choice | Why |
|--------|--------|-----|
| Curvature | Fixed (`requires_grad=False`) | All params stay Euclidean, so RF-DETR's built-in `AdamW` optimizer works unchanged |
| Classification | `poincare_hyperplane_dists` | This is HypLL's implementation of the HNN++ formula from Ganea et al. — the principled way to do hyperbolic classification |
| `self.bias` naming | Matches original `class_embed.bias` | RF-DETR's checkpoint loader checks `checkpoint['model']['class_embed.bias'].shape[0]` at `main.py:99` |
| `self.proj` | Extra `nn.Linear` | Decoder features can have large magnitudes; this controls the scale before `expmap0` (which uses `tanh` internally and would saturate) |

### 3. Monkey-patch `reinitialize_detection_head`

This is critical. During `model.train()`, RF-DETR calls `reinitialize_detection_head()` to resize the classification head for your dataset's class count. By default, it creates a standard `nn.Linear`:

```105:118:c:\Users\user\Documents\Vehicle-Detection-at-R-R-\rf_env\Lib\site-packages\rfdetr\models\lwdetr.py
    def reinitialize_detection_head(self, num_classes):
        # Create new classification head
        del self.class_embed
        self.add_module("class_embed", nn.Linear(self.transformer.d_model, num_classes))
        # ...
```

You must override this so it creates your hyperbolic head instead:

```python
import types

def _hyp_reinit_detection_head(self, num_classes):
    """Override: create hyperbolic head instead of nn.Linear."""
    del self.class_embed
    hyp_head = HyperbolicDETRHead(
        input_dim=self.transformer.d_model,
        num_classes=num_classes,
        curvature=0.1,
    )
    self.add_module("class_embed", hyp_head)

    # Focal loss bias init (same as original RF-DETR)
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    with torch.no_grad():
        self.class_embed.bias.data.fill_(bias_value)

    # Handle two-stage if applicable
    if self.two_stage:
        import copy
        del self.transformer.enc_out_class_embed
        self.transformer.add_module(
            "enc_out_class_embed",
            nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]
            ),
        )
    print(f"Hyperbolic head initialized: {num_classes} classes, dim={self.transformer.d_model}")

# Apply to the LWDETR model inside your RFDETRBase instance
model.model.model.reinitialize_detection_head = types.MethodType(
    _hyp_reinit_detection_head, model.model.model
)
```

The access path is `model.model.model` because:
- `model` = `RFDETRBase` instance
- `model.model` = internal `Model` instance
- `model.model.model` = the actual `LWDETR` PyTorch module

### 4. Instantiate the model

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights="rf-detr-base.pth")
```

Then apply the monkey-patch from step 3 **before** calling `model.train()`.

### 5. Train

Use the normal RF-DETR training API — no changes needed:

```python
model.train(
    dataset_dir=dataset.location,
    epochs=30,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    amp=True,
)
```

**Why no optimizer changes?** Because all parameters in `HyperbolicDETRHead` are standard `nn.Parameter` / `nn.Linear` (Euclidean). The hyperbolic geometry enters only in the forward pass via differentiable operations (`expmap0`, `poincare_hyperplane_dists`). Gradients flow through these ops normally via autograd, and AdamW updates the Euclidean parameters correctly.

### 6. Inference after training

When loading the trained checkpoint for inference, you need the same monkey-patch so the model structure matches the saved state dict:

```python
model = RFDETRBase(pretrain_weights="output/checkpoint_best_total.pth")

# Must apply the same monkey-patch before any predict calls
model.model.model.reinitialize_detection_head = types.MethodType(
    _hyp_reinit_detection_head, model.model.model
)
# The checkpoint loading in __init__ will trigger reinitialize, 
# so the hyperbolic head structure is created and weights are loaded

model.optimize_for_inference()
detections = model.predict(image, threshold=0.5)
```

**Important:** The checkpoint saves `class_embed.proj.weight`, `class_embed.proj.bias`, `class_embed.z`, `class_embed.r`, `class_embed.bias`. These only load correctly if the hyperbolic head module is in place.

---

## Summary checklist

| # | Task | Touches |
|---|------|---------|
| 1 | `pip install hypll` | Environment |
| 2 | Define `HyperbolicDETRHead` class | New code in notebook |
| 3 | Monkey-patch `reinitialize_detection_head` | Before `model.train()` |
| 4 | Instantiate `RFDETRBase` with pretrained weights | Notebook cell |
| 5 | Call `model.train(...)` — no optimizer changes needed | Notebook cell |
| 6 | Apply same monkey-patch when loading checkpoint for inference | `vehicle_counting.py` or inference notebook |

Switch to **Agent mode** if you'd like me to implement these changes directly in your notebook.