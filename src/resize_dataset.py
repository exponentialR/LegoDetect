from datasets import LegoDataset
import os
from PIL import Image
from tqdm import tqdm

PATH_TO_DATASET = '/home/iamshri/Documents/Dataset/Legos'
TARGET_SIZE = (960, 540)


def resize_dataset_save(dataset, target_size, new_location):
    os.makedirs(new_location, exist_ok=True)
    labels_location = os.path.join(new_location, 'labels')
    os.makedirs((labels_location), exist_ok=True) if not os.path.exists(labels_location)  else None
    for _, labels, image_path in tqdm(dataset, desc='Resizing Images'):
        image = Image.open(image_path)
        image = image.resize(target_size)
        relative_path = os.path.relpath(image_path, dataset.img_dir)
        new_image_path = os.path.join(new_location, relative_path)
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        image.save(new_image_path)

        label_file_name = os.path.basename(image_path).rsplit('.', 1)[0] + '.txt'
        label_file_path = os.path.join(labels_location, label_file_name)

        with open(label_file_path, 'w') as f:
            for label in labels:
                class_label = label['label']
                bbox = label['bbox']
                bbox_str = ' '.join(map(str, [class_label] + bbox))
                f.write(f"{bbox_str}\n")


if __name__ == '__main__':
    dataset = LegoDataset(PATH_TO_DATASET)
    # print(PATH_TO_DATASET, TARGET_SIZE, 'data/lego_resized')
    resize_dataset_save(dataset, TARGET_SIZE, 'Legos-Dataset-Resized/images')
