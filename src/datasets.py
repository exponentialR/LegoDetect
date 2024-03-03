import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


class LegoDataset(Dataset):
    def __init__(self, data_dir, target_size=(500, 500), transform=None):
        self.img_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.transform = transform
        self.target_size = target_size

        self.image_files = [i for i in os.listdir(self.img_dir) if i.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_name = self.image_files[item]
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        label_path = os.path.join(os.path.join(self.label_dir, image_name.replace('.jpg', '.txt')))
        label = self._read_label(label_path)
        image, label = self._resize_image(image, label)

        # Apply Transformation
        if self.transform:
            image = self.transform(image)

        return image, label

    def _read_label(self, label_path):
        label = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                class_label = int(line[0])
                bbox = list(map(float, line[1:]))
                label.append({'label': class_label, 'bbox': bbox})

        return label

    def _resize_image(self, image, label):
        width, height = image.size
        image = image.resize(self.target_size, Image.BILINEAR)

        scale_width = self.target_size[0] / width
        scale_height = self.target_size[1] / height

        for item in label:
            bbox = item['bbox']
            bbox[0] *= scale_width  # Adjust x-coordinate
            bbox[1] *= scale_height  # Adjust y-coordinate
            bbox[2] *= scale_width  # Adjust width
            bbox[3] *= scale_height  # Adjust height

        return image, label


if __name__ == '__main__':
    project_dir = '/home/qub-hri/Documents/Datasets/Legos'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LegoDataset(project_dir, transform=transform)
    output_dir = 'visualization_images'
    os.makedirs(output_dir, exist_ok=True)
    num_images_visualise = 5

    for example_idx in range(num_images_visualise):

        image, labels = dataset[example_idx]
        image = image.permute(1, 2, 0).numpy()

        image_with_boxes = image.copy()
        dh, dw, _ = image.shape

        for idx, item in enumerate(labels):
            class_label = item['label']
            bbox = item['bbox']
            x, y, w, h = bbox

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2 ) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            cv2.rectangle(image_with_boxes, (l, t), (r, b), (255, 0, 255), 2)
        cv2.imshow("With Bounding Box", image_with_boxes)
        cv2.waitKey(50)




