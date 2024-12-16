import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import yaml
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

def write_labels(img_paths, labels, output_dir):
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    for count, (img_path, label) in enumerate(zip(img_paths, labels), start=1):
        source_img_path = os.path.join('./train/', img_path)
        img = Image.open(source_img_path)
        width, height = img.size

        dest_img_path = os.path.join(images_dir, f'{count}.jpg')
        shutil.copy(source_img_path, dest_img_path)

        with open(os.path.join(labels_dir, f'{count}.txt'), 'w', encoding='UTF-8') as fw:
            for index, label_line in enumerate(label):
                label_tmp = label_line
                if len(label) == 8:
                    label_tmp['category_id'] = index
                elif len(label) == 9:
                    if index <= 5:
                        label_tmp['category_id'] = index
                    elif index == 6 or index == 7:
                        label_tmp['category_id'] = 6
                    else:
                        label_tmp['category_id'] = 7
                elif len(label) == 10:
                    if index <= 4:
                        label_tmp['category_id'] = index
                    elif index == 5 or index == 6:
                        label_tmp['category_id'] = 5
                    elif index == 7 or index == 8:
                        label_tmp['category_id'] = 6
                    else:
                        label_tmp['category_id'] = 7

                points = label_tmp['points']
                x_min = min(point[0] for point in points)
                y_min = min(point[1] for point in points)
                x_max = max(point[0] for point in points)
                y_max = max(point[1] for point in points)

                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                norm_width = (x_max - x_min) / width
                norm_height = (y_max - y_min) / height

                # Ghi vào file theo định dạng YOLO
                fw.write(f"{label_tmp['category_id']} {x_center} {y_center} {norm_width} {norm_height}\n")

write_labels(img_train, y_train, './datasets/train/')
write_labels(img_val, y_val, './datasets/val/')
class_labels = {
    0: 'id',
    1: 'name',
    2: 'DOB',
    3: 'Sex',
    4: 'Nationality',
    5: 'Origin',
    6: 'Residence',
    7: 'Expiry'
}
data_yaml = {
    'path': 'yolo_data',
    'train': 'train/images/',
    'test': '',
    'val': 'val/images/',
    'nc': len(class_labels),
    'names': class_labels
}

yolo_yaml_path = os.path.join('./datasets/yolo_data', 'data.yml')
with open(yolo_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)