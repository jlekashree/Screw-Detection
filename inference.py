from sklearn.cluster import KMeans
import cv2
import numpy as np
from ultralytics import YOLO
import os

TRAINED_MODEL_PATH = "runs/segment/train/weights/best.pt"
model = YOLO(TRAINED_MODEL_PATH)

def predict(file_path: str, conf=0.25):
  results = model(file_path, conf=conf)
  results = list(results)

  lengths = []
  black_count = 0

  for r in results:
    if r.masks is None:
      continue

  masks = r.masks.data.cpu().numpy()
  classes = r.boxes.cls.cpu().numpy()
  image = r.orig_img

  for mask, cls_id in zip(masks, classes):
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    if cls_id == 1:
      black_count += 1
      continue

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
      continue

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    length = max(w, h)

    if length > 15:   # filtering noise
      lengths.append(length)

  # classifying normal screws by length using KMeans
  lengths = np.array(lengths)
  if len(lengths) >= 2:
    X = lengths.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())
    LENGTH_THRESHOLD = np.mean(centers)
  else:
    LENGTH_THRESHOLD = lengths[0] * 1.1

  short_count = np.sum(lengths <= LENGTH_THRESHOLD)
  long_count  = np.sum(lengths > LENGTH_THRESHOLD)

  image = annotate_image(image , masks , classes, lengths, LENGTH_THRESHOLD)
  cv2.imshow("Image Display", image)
  cv2.waitKey(0)
  print("Short screws:", short_count)
  print("Long screws:", long_count)
  print("Black screws:", black_count)
  print("TOTAL screws:", short_count + long_count + black_count)

def annotate_image(image, masks, classes, lengths, length_threshold):
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    idx = 0  

    for mask, cls_id in zip(masks, classes):
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        if int(cls_id) == 1:
            color = (0, 0, 255)  
            label = "Black Screw"
        else:
            length = lengths[idx]
            idx += 1

            if length <= length_threshold:
                label = "Short Screw"
                color = (0, 255, 0)  
            else:
                label = "Long Screw"
                color = (255, 0, 0)  

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated, label, (x, y-10),
                    font, 0.5, color, 2)

    return annotated

if __name__ == '__main__':
  for file_path in os.listdir("test"):
    predict(file_path = os.path.join('test',file_path) , conf=0.25)
