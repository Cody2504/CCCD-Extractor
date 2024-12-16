import os
from PIL import Image
import json
import numpy as np
from sklearn.model_selection import train_test_split
labels = []
img_paths = []
with open('Label.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        img_path = line.split("\t")[0]
        label_str = line.split("\t")[1]
        img_paths.append(img_path)
        data_list = json.loads(label_str)
        label = []
        for item in data_list:
            label.append(item)
        labels.append(label)    
    f.close()
img_train, img_val, y_train, y_val = train_test_split(img_paths, labels, test_size=0.2, random_state=42)
def split_bboxes(img_paths, labels, save_dir, annotation_path, type='train/'):
    os.makedirs(save_dir, exist_ok=True)
    dest_img_dir = os.path.join(save_dir, type)
    os.makedirs(dest_img_dir, exist_ok=True)
    texts = []
    count = 0
    for img_path, label in zip(img_paths, labels):
        source_img_path = os.path.join('train/', img_path)
        img = Image.open(source_img_path)
        for label_line in label:
            points = label_line['points']
            x_min = min(point[0] for point in points)
            y_min = min(point[1] for point in points)
            x_max = max(point[0] for point in points)
            y_max = max(point[1] for point in points)

            text = label_line['transcription']
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue
            filename = f'{count:06d}.jpg'
    
            cropped_img.save(os.path.join(dest_img_dir, filename))
            new_img_path = os.path.join(type, filename)
            text = new_img_path + '\t' + text
            texts.append(text)
            count += 1
    
    with open(os.path.join(save_dir, annotation_path), 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"{text}\n")

save_dir_train = './datasets/ocr_data/'
annotation_path_train = 'train_annotation.txt'
split_bboxes(img_train, y_train, save_dir_train, annotation_path_train, 'train/')
save_dir_val = './datasets/ocr_data/'
annotation_path_val = 'val_annotation.txt'
split_bboxes(img_val, y_val, save_dir_val, annotation_path_val, 'val/')