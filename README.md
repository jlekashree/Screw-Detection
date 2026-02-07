# Screw Detection & Classification: YOLOv8-Seg + K-Means Clustering

This repository contains a specialized computer vision pipeline designed to detect and classify three distinct types of screws: **Long Screws**, **Short Screws**, and **Black Screws**. 

The project addresses the challenge of high-similarity object detection with a limited dataset by combining state-of-the-art instance segmentation (**YOLOv8n-seg**) with unsupervised machine learning (**K-Means Clustering**) for geometric analysis.

---

## The Challenge: Small Data & Fine-Grained Classification
The primary hurdle for this project was the **initial dataset size of only 22 unique training images**. 

To build a robust model under these constraints, we shifted from a "pure-classification" approach to a **Hybrid Detection-Geometric approach**:
1.  **Class 1 (Black Screw):** Identified directly by the model due to distinct color features.
2.  **Class 0 (Screws):** Identified as a generic class, then programmatically split into **Long** and **Short** based on pixel-length analysis to avoid labeling errors caused by subtle size differences.

---

## Strategy & Training

### 1. Data Augmentation
To prevent overfitting on the 22-image dataset, we utilized augmentation strategy to simulate variations in orientation, lighting, and composition:

```yaml
# Augmentation Configuration
mosaic: 1.0       # Combine 4 images into one
mixup: 0.2        # Blend two images
degrees: 180.0    # Full rotation range
flipud: 0.5       # Vertical flips
fliplr: 0.5       # Horizontal flips
hsv_v: 0.4        # Brightness jitter
translate: 0.1    # Random translation
scale: 0.5        # Zooming
shear: 2.0        # Perspective shear
patience: 30      # Early stopping
close_mosaic: 10  # Disable mosaic in final epochs for refinement
```


### 2. Post-Inference Classification Logic
Because "Long" and "Short" screws are identical in every aspect except length, we use K-Means Clustering on the predicted masks:

Mask Extraction: YOLOv8n-seg provides binary masks for every detected screw.

Geometric Measurement: We calculate the `minAreaRect` for each mask and extract the longest dimension.

Unsupervised Splitting: K-Means ($k=2$) identifies the two primary size groups in the image and calculates a dynamic LENGTH_THRESHOLD based on the mean of the cluster centers.

### Evaluation Metrics
The model performance was evaluated using standard COCO segmentation metrics:

| Metric     | Value |
|------------|--------|
| mAP@50     |  0.95 |
| mAP@50-95  | 0.70 |
| Precision  | 0.96 |
| Recall     | 0.92 |

### Annotated Dataset Sample
Example of the initial ground-truth labeling for Black Screws and generic Screws.

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/39b337bf-1bd0-4a74-a014-bc1787e0cef6" />

### Predicted & Classified Output

Final inference result: The model detects the screws, and the script classifies them into Short, Long, and Black categories.

<img width="420" height="420" alt="image" src="https://github.com/user-attachments/assets/e9cb14b0-868f-412f-a6af-ee6f62b3336f" />


<b> Short screws: 22 </br>
<b> Long screws: 40 </br>
<b> Black screws: 27 </br>
<b> TOTAL screws: 89 

---

# To Run:
Three main scripts to run:
- `trainer.py` → Ttrain the YOLOv8n-seg model  
- `inference.py` → To run inference and visualize predictions  
- `evaluation.py` → To evaluate model performance (mAP, precision, recall)  



# Install Dependencies

```bash
pip install -r requirements.txt
```

