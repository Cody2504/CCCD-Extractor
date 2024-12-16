from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

class TextRecognition():
    def __init__(self, weights_path, device = 'cuda:0', text_recognition_config_path='./configs/configVietOCR.yml'):
        config = Cfg.load_config_from_file(text_recognition_config_path)
        config['weights'] = weights_path
        config['device'] = device
        self.recognizer = Predictor(config)

    def __call__(self, img):
        text, confidence = self.recognizer.predict(img, return_prob = True)
        return text, confidence