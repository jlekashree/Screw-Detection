from ultralytics import YOLO

MODEL_NAME = "yolov8n-seg.pt"
YAML_PATH = "data.yaml"

model = YOLO(MODEL_NAME)

model.train(
  data=YAML_PATH,
  epochs=200,
  imgsz=640,
  mosaic=1.0,
  mixup=0.2,
  degrees=180.0,
  flipud = 0.5,
  fliplr=0.5,
  hsv_v=0.4,
  translate=0.1,
  scale=0.5,
  shear=2.0,
  patience=30,
  close_mosaic=10,
)
