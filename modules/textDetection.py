from ultralytics import YOLO
from PIL import Image
import numpy as np
class TextDetection():
    def __init__(self, model_path_textDet, device = 'cuda'):
        self.detector = YOLO(model_path_textDet)
        self.detector.to(device)
        
    def __call__(self, img_array):
        results = self.detector.predict(source=img_array, conf=0.3, save=False)
    
        detected_items = []
        img = Image.fromarray(img_array)
        for result in results[0]:
            bbox = result.boxes.xywh.tolist()[0]
            class_id = int(result.boxes.cls.tolist()[0])

            x_center, y_center, width, height = bbox
            top_y = float(y_center - height / 2)
            
            cropped_img = self.crop_image(img, bbox)
            if cropped_img is not None:
                detected_items.append((class_id, top_y, bbox, cropped_img))

        detected_items.sort(key=lambda x: (x[0], x[1]))

        return [(item[0],item[3]) for item in detected_items]
    
    @staticmethod
    def crop_image(img, bbox):
        x_center, y_center, width, height = bbox

        left = float(x_center - width / 2)
        top = float(y_center - height / 2)
        right = float(x_center + width / 2)
        bottom = float(y_center + height / 2)
 
        cropped_img = img.crop((left, top, right, bottom)).convert("RGB")
        if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
            cropped_img = None

        if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
            cropped_img = None

        return cropped_img
