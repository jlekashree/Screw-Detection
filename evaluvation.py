from ultralytics import YOLO

MODEL_PATH = "runs/segment/train/weights/best.pt"
DATA_YAML="data.yaml"

model = YOLO(MODEL_PATH)
metrics = model.val(data = DATA_YAML)

print("mAP50:", metrics.seg.map50)
print("mAP50-95:", metrics.seg.map)
print("Precision:", metrics.seg.mp)
print("Recall:", metrics.seg.mr)
