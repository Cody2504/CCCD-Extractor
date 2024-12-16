from ultralytics import YOLO
import cv2
import numpy as np
class Detect4Corner():
    def __init__(self, model_path, device = 'cuda'):
        self.detector = YOLO(model_path)
        self.detector.to(device)
        
    def __call__(self, img_source):
        results = self.detector.predict(source=img_source, conf=0.45, save=False)
        
        detected_items = []
        for result in results[0]:
            bbox = result.boxes.xywh.tolist()[0]
            class_id = int(result.boxes.cls.tolist()[0])
            detected_items.append((class_id, bbox))

        corners = []
        detected_corners = {}

        for item in detected_items:
            class_id, bbox = item
            x_center, y_center, width, height = bbox
            
            if class_id == 0:  # top-left
                detected_corners[class_id] = [
                    x_center - width / 2,
                    y_center - height / 2
                ]
            elif class_id == 1:  # top-right
                detected_corners[class_id] = [
                    x_center + width / 2, 
                    y_center - height / 2 
                ]
            elif class_id == 2:  # bottom-right
                detected_corners[class_id] = [
                    x_center + width / 2,  
                    y_center + height / 2
                ]
            elif class_id == 3:  # bottom-left
                detected_corners[class_id] = [
                    x_center - width / 2,
                    y_center + height / 2
                ]
        for class_id in range(4):
            if class_id in detected_corners:
                corners.append(detected_corners[class_id])
            else:
                if class_id == 0:  # top-left
                    mid_x = (detected_corners[1][0] + detected_corners[3][0]) / 2
                    mid_y = (detected_corners[1][1] + detected_corners[3][1]) / 2
                    missing_corner = [
                        2 * mid_x - detected_corners[2][0],
                        2 * mid_y - detected_corners[2][1],
                    ]
                elif class_id == 1:  # top-right
                    mid_x = (detected_corners[0][0] + detected_corners[2][0]) / 2
                    mid_y = (detected_corners[0][1] + detected_corners[2][1]) / 2
                    missing_corner = [
                        2 * mid_x - detected_corners[3][0],
                        2 * mid_y - detected_corners[3][1],
                    ]
                elif class_id == 2:  # bottom-right
                    mid_x = (detected_corners[1][0] + detected_corners[3][0]) / 2
                    mid_y = (detected_corners[1][1] + detected_corners[3][1]) / 2
                    missing_corner = [
                        2 * mid_x - detected_corners[0][0],
                        2 * mid_y - detected_corners[0][1],
                    ]
                elif class_id == 3:  # bottom-left
                    mid_x = (detected_corners[0][0] + detected_corners[2][0]) / 2
                    mid_y = (detected_corners[0][1] + detected_corners[2][1]) / 2
                    missing_corner = [
                        2 * mid_x - detected_corners[1][0],
                        2 * mid_y - detected_corners[1][1],
                    ]
                corners.append(missing_corner)
        
        return np.array(corners, dtype=np.float32)

    @staticmethod
    def perspective_transform(img_path, corners, output_size=(800, 500)):
        img = cv2.imread(img_path)
        
        width, height = output_size
        dst_points = np.array([
            [0, 0],           
            [width - 1, 0],   
            [width - 1, height - 1], 
            [0, height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        warped_img = cv2.warpPerspective(img, matrix, (width, height))

        return warped_img