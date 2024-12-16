import os
from sklearn.model_selection import train_test_split
import shutil
import yaml

images_dir = './yolo_data4Corner/images'
labels_dir = './yolo_data4Corner/labels_xml'
output_dir = './datasets/yolo_4CornerRotate'

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

image_paths = [os.path.join(images_dir, f) for f in image_files]
label_paths = [os.path.join(labels_dir, f) for f in label_files]

img_train, img_val, lbl_train, lbl_val = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42
)

def prepare_data_split(image_paths, label_paths, split_name):
    images_output_dir = os.path.join(output_dir, split_name, 'images')
    labels_output_dir = os.path.join(output_dir, split_name, 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    for img_path, lbl_path in zip(image_paths, label_paths):
        shutil.copy(img_path, os.path.join(images_output_dir, os.path.basename(img_path)))
        shutil.copy(lbl_path, os.path.join(labels_output_dir, os.path.basename(lbl_path)))

prepare_data_split(img_train, lbl_train, 'train')
prepare_data_split(img_val, lbl_val, 'val')

class_labels = {
    0: 'top_left',
    1: 'top_right',
    2: 'bottom_right',
    3: 'bottom_left'
}

data_yaml = {
    'path': 'yolo_4CornerRotate',
    'train': 'train/images',
    'val': 'val/images',
    'test': '', 
    'nc': len(class_labels),
    'names': class_labels
}

yolo_yaml_path = os.path.join(output_dir, 'data.yaml')
with open(yolo_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)