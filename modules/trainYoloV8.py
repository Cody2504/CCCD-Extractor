from ultralytics import YOLO

class TrainYOLOv8():
    def __init__(self, device = 'cuda:0'):
        self.model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        self.model.to(device)
        self.device = device

    def __call__(self, epochs=50, imgsz=640, models_name = 'models_yolo4CornerRotate', models_path='yolov8/detect/train', yaml_path='./datasets/yolo_4CornerRotate/data.yaml'):
        results = self.model.train(
            data = yaml_path,
            epochs = epochs,
            imgsz = imgsz,
            project = models_name,
            name = models_path,
            device = 0,
            batch = 8,
            workers = 0
        )
        return results
        
