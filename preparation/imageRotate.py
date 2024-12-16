from PIL import Image
import os

def rotate_90_left(image):
    return image.transpose(Image.Transpose.ROTATE_90)

def rotate_90_right(image):
    return image.transpose(Image.Transpose.ROTATE_270)

def rotate_45(image, clockwise=True):
    angle = -45 if clockwise else 45
    return image.rotate(angle, expand=True)

def save_image(image, output_folder, file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_path = os.path.join(output_folder, file_name)
    image.save(save_path)
    print(f"Đã lưu ảnh tại: {save_path}")

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        if os.path.isfile(input_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image = Image.open(input_path)
                base_name = os.path.splitext(file_name)[0]

                transformations = {
                    f"{base_name}_rotated_90_left.jpg": rotate_90_left(image),
                    f"{base_name}_rotated_90_right.jpg": rotate_90_right(image),
                    f"{base_name}_rotated_45_clockwise.jpg": rotate_45(image, clockwise=True),
                    f"{base_name}_rotated_45_counterclockwise.jpg": rotate_45(image, clockwise=False),
                }

                for new_file_name, transformed_image in transformations.items():
                    save_image(transformed_image, output_folder, new_file_name)
            
            except Exception as e:
                print(f"Lỗi khi xử lý tệp {file_name}: {e}")

if __name__ == "__main__":
    input_directory = "./dataForAugmentation"
    output_directory = "./AugmentedData"

    augment_images(input_directory, output_directory)
