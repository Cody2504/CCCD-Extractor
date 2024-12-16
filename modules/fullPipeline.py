from modules.textDetection import TextDetection
from modules.textRecognition import TextRecognition
from modules.detect4Corner import Detect4Corner

class FullPipelineOCR():
    def __init__(self, model_path_4Corner='./models_yolo4CornerRotate/yolov8/detect/train/weights/best.pt', model_path_textDet='./models_yolo/yolov8/detect/train/weights/best.pt', model_path_textRec = './weights/transformerocr.pth', device = 'cuda'):
        self.detectCorner = Detect4Corner(model_path=model_path_4Corner)
        self.detector = TextDetection(model_path_textDet=model_path_textDet, device=device)
        self.recognizer = TextRecognition(weights_path=model_path_textRec, device=device)
        
    def __call__(self, front_image, output_size=(800, 500)):
        front_text = {}
        if front_image is not None:
            corners = self.detectCorner(front_image)
            if len(corners) != 4: 
                return "Ảnh chụp bị thiếu góc, vui lòng chụp lại!"
            warped_image = self.detectCorner.perspective_transform(front_image, corners, output_size)
            result_detected = self.detector(warped_image) 
            for class_id, cropped_img in result_detected:
                text, confidence = self.recognizer(cropped_img)
                
                if class_id not in front_text:
                    front_text[class_id] = text
                else:
                    front_text[class_id] += " " + text
        return front_text
    
    

