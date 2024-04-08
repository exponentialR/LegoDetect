from torch.utils.data import Dataset, DataLoader
import os
import cv2


class LegoDataset(Dataset):
    def __init__(self, data_dir, target_size=(960, 540), transform=None):
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
        image = cv2.imread(image_path)
        label_path = os.path.join(os.path.join(self.label_dir, image_name.replace('.jpg', '.txt')))
        label = self._read_label(label_path)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

        # Apply Transformation
        if self.transform:
            image = self.transform(image)

        return image, label, image_path

    def _read_label(self, label_path):
        label = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                class_label = int(line[0])
                bbox = list(map(float, line[1:]))
                label.append({'label': class_label, 'bbox': bbox})

        return label

