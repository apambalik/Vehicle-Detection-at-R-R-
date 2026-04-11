# Section 3.3 Exploratory Data Analysis (EDA) — Structure and Recommendations

This document provides the recommended sub-heading structure, graph inclusions, and write-up guidance for **Section 3.3 Exploratory Data Analysis** in Chapter 3 (Methodology) of the vehicle detection and classification project.

---

## Recommended Sub-heading Structure

```
3.3 Exploratory Data Analysis
    3.3.1 Dataset Overview
    3.3.2 Class Distribution Analysis
    3.3.3 Bounding Box Spatial and Size Analysis
    3.3.4 Image-Level Co-occurrence Analysis
    3.3.5 Challenges in Class Balancing
    3.3.6 Dataset Export Formats
```

---

## 3.3.1 Dataset Overview

**What to write:**
- Total number of images across train/valid/test splits.
- Total number of annotations (bounding boxes).
- Number of vehicle classes and list them (sedan, pickup, motorcycle, bus, truck, etc.).
- A brief note on image resolution consistency (all images from the same CCTV source or multiple sources).

**Graph to include:**

| # | Graph | Source from your EDA |
|---|-------|---------------------|
| 1 | **Dataset Split Distribution** (bar chart showing number of images per train/valid/test split) | Your bar chart "Images per Split" — the one showing train ~356, valid ~102, test ~51 images |

---

## 3.3.2 Class Distribution Analysis

**What to write:**
- Present the annotation counts per class across all splits and per-split.
- Highlight the severe class imbalance: sedan and pickup dominate, while classes like bus, truck, and motorcycle are underrepresented.
- Provide exact counts and percentages.

**Graphs to include:**

| # | Graph | Source from your EDA |
|---|-------|---------------------|
| 2 | **Overall Class Distribution** (bar chart of total annotations per class) | Your horizontal bar chart "Class Distribution in Full Dataset" showing sedan (~3400), pickup (~1700), motorcycle (~700), bus (~350), truck (~250) etc. |
| 3 | **Class Distribution per Split** (grouped bar chart or side-by-side bar charts for train/valid/test) | Your grouped bar chart "Class Distribution per Split" showing per-class counts across train, valid, and test |
| 4 | **Class Proportion Pie/Donut Chart** (optional, but useful to visualize dominance) | If you have one; otherwise the bar chart suffices |

---

## 3.3.3 Bounding Box Spatial and Size Analysis

**What to write:**
- Describe the spatial distribution of bounding box centres — where in the image vehicles tend to appear (this relates to the CCTV camera angle and field of view).
- Describe the distribution of bounding box sizes (width and height) and aspect ratios — this informs anchor design or model input resolution choices.
- Note any patterns: e.g., vehicles closer to the camera have larger boxes; certain classes (bus, truck) have systematically larger bounding boxes than motorcycles.

**Graphs to include:**

| # | Graph | Source from your EDA |
|---|-------|---------------------|
| 5 | **Bounding Box Centre Heatmap** (2D density plot of bbox centre x,y coordinates) | Your heatmap "Bounding Box Center Distribution" — the scatter/density plot showing where centres cluster in the image frame |
| 6 | **Bounding Box Width vs Height Scatter Plot** (coloured by class) | Your scatter plot "Bbox Width vs Height by Class" showing the size clusters per vehicle type |
| 7 | **Bounding Box Area Distribution** (histogram or box plot per class) | Your box plot or histogram "Bbox Area Distribution per Class" showing the spread of areas |
| 8 | **Bounding Box Aspect Ratio Distribution** (histogram per class) | Your histogram "Aspect Ratio Distribution per Class" |

---

## 3.3.4 Image-Level Co-occurrence Analysis

**What to write:**
- Analyse how many classes co-occur within the same image.
- Show that sedan and pickup appear in the vast majority of images, often alongside other classes.
- This is critical context for the class balancing challenge (Section 3.3.5).

**Graphs to include:**

| # | Graph | Source from your EDA |
|---|-------|---------------------|
| 9 | **Class Co-occurrence Matrix** (heatmap showing how frequently two classes appear in the same image) | Your heatmap "Co-occurrence Matrix" — the correlation/co-occurrence matrix across all classes |
| 10 | **Number of Classes per Image** (histogram showing distribution of how many distinct classes each image contains) | Your histogram "Number of Classes per Image" |
| 11 | **Class Presence per Image** (percentage of images that contain at least one instance of each class) | Your bar chart showing e.g., sedan appears in ~95% of images, pickup in ~85%, etc. |

---

## 3.3.5 Challenges in Class Balancing

> **This is the key finding you want to highlight.**

**What to write:**

This sub-section should present the argument clearly, structured as follows:

### The Problem

The dataset exhibits severe class imbalance — sedan and pickup annotations collectively represent over 75% of all annotations, while minority classes such as bus, truck, and motorcycle are significantly underrepresented. At first glance, conventional class balancing strategies might seem applicable:

- **Over-sampling (augmenting minority classes):** Duplicating or augmenting images containing minority classes.
- **Under-sampling (removing majority class annotations):** Deleting some sedan/pickup annotations to reduce their dominance.

### Why Conventional Balancing Fails for This Dataset

However, analysis of the dataset reveals that **these approaches are not viable** due to the pervasive co-occurrence of majority classes across nearly all images:

1. **Sedan and pickup vehicles appear in the background of almost every image.** The co-occurrence analysis (Section 3.3.4) shows that sedan appears in approximately 95% of images and pickup in approximately 85% of images. These vehicles are part of the natural traffic scene captured by the CCTV camera.

2. **Under-sampling by deleting annotations creates unannotated positives.** If sedan or pickup annotations are selectively removed from images to reduce their count, the corresponding vehicles remain visible in the image. The detection model will then learn to treat these clearly visible vehicles as **background/negative examples**. This is a form of **label noise** that directly degrades model performance — the model learns to suppress detections of the very classes it should detect.

3. **Over-sampling minority classes does not address the co-occurrence issue.** Augmenting or duplicating images containing minority classes (e.g., bus, truck) will simultaneously duplicate the sedan and pickup instances in those same images, further inflating the majority class counts rather than balancing them.

4. **Cropping individual objects removes contextual information.** An alternative of cropping individual vehicles and creating single-class images would destroy the spatial context (road scene, perspective, surrounding vehicles) that is essential for accurate detection in real CCTV footage.

### The Adopted Strategy

Given these constraints, the following approach was adopted instead:

- **Use the dataset as-is** with its natural class distribution, preserving annotation integrity.
- **Apply class-aware loss weighting** during training (e.g., focal loss, class-weighted loss) to give minority classes higher importance in the loss function.
- **Evaluate with per-class metrics** (per-class AP, per-class precision/recall) rather than relying solely on mAP, to ensure minority class performance is monitored.
- *(Add any other strategies you actually used, such as mosaic augmentation, mixup, etc.)*

**Graph to include:**

| # | Graph | Source from your EDA |
|---|-------|---------------------|
| 12 | **Class Presence Across Images** (reuse or reference the bar chart from 3.3.4 showing % of images containing each class) | Same as Graph #11 — emphasise the near-universal presence of sedan/pickup |

Additionally, consider creating a **visual diagram** (even a simple annotated screenshot) showing:
- An example CCTV image with ALL vehicles annotated (full annotations).
- The same image with sedan/pickup annotations removed — highlighting that the vehicles are still clearly visible but now unannotated, demonstrating the label noise problem.

---

## 3.3.6 Dataset Export Formats

**What to write:**
- Briefly describe that the same dataset was exported in two formats to support the two detection architectures being evaluated:
  - **COCO format (JSON)** — used for RT-DETR / RF-DETR training. Describe the structure: `instances_train.json`, `instances_val.json`, `instances_test.json` with image entries and annotation entries containing `bbox` in `[x, y, width, height]` format and `category_id`.
  - **YOLOv26 format (TXT)** — used for YOLO26 training. Describe the structure: one `.txt` file per image with each line containing `class_id x_center y_center width height` in normalised coordinates, plus a `data.yaml` file listing class names and paths.
- Note that both formats represent the identical dataset (same images, same annotations) to ensure a fair comparison between the two models.

**No graph needed for this sub-section** — a small table or side-by-side code snippet showing the format structure would suffice.

---

## Summary of All Graphs/Figures for Section 3.3

| Figure # | Description | Sub-section |
|----------|-------------|-------------|
| 1 | Dataset Split Distribution (bar chart) | 3.3.1 |
| 2 | Overall Class Distribution (bar chart) | 3.3.2 |
| 3 | Class Distribution per Split (grouped bar chart) | 3.3.2 |
| 4 | Class Proportion (pie/donut — optional) | 3.3.2 |
| 5 | Bounding Box Centre Heatmap | 3.3.3 |
| 6 | Bounding Box Width vs Height Scatter | 3.3.3 |
| 7 | Bounding Box Area Distribution per Class | 3.3.3 |
| 8 | Aspect Ratio Distribution per Class | 3.3.3 |
| 9 | Class Co-occurrence Matrix (heatmap) | 3.3.4 |
| 10 | Number of Classes per Image (histogram) | 3.3.4 |
| 11 | Class Presence per Image (% bar chart) | 3.3.4 |
| 12 | Annotated vs Unannotated Example (diagram) | 3.3.5 |

**Total: 11–12 figures** (depending on whether you include the optional pie chart and the annotated example diagram).

---

## Writing Tips

1. **Every graph must be referenced in the text.** Don't just place a graph — describe what it shows, what the key observation is, and why it matters for your methodology.

2. **Use a consistent figure numbering scheme** that matches your thesis template (e.g., Figure 3.1, Figure 3.2, ...).

3. **For Section 3.3.5 (Class Balancing Challenge)**, use clear, logical argumentation. Present it almost like a proof:
   - State the assumption (balancing seems desirable).
   - Present the evidence (co-occurrence data).
   - Derive the conclusion (conventional balancing is counterproductive).
   - State the alternative (what you actually did).

4. **Keep the dataset format section (3.3.6) brief** — it's a methodology detail, not an analysis finding. A short paragraph per format with a small example is sufficient.
