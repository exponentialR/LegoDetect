import cv2
import os
import numpy as np
from torchvision import transforms
from datasets import LegoDataset


# Assuming LegoDataset is defined elsewhere

def visualise_dataset(dataset, num_images_visualise):
    current_idx = 0

    while True:
        image, labels, _ = dataset[current_idx]
        print(image.shape, len(labels))
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
            b = int((y + h / 2) * dh)

            l = max(l, 0)
            r = min(r, dw - 1)
            t = max(t, 0)
            b = min(b, dh - 1)

            cv2.rectangle(image_with_boxes, (l, t), (r, b), (255, 0, 255), 2)

        cv2.imshow("With Bounding Box", image_with_boxes)
        key = cv2.waitKey(0)

        if key == 27:  # ESC key
            break
        elif key == ord('d'):  # Right arrow simulation
            current_idx = (current_idx + 1) % num_images_visualise
        elif key == ord('a'):  # Left arrow simulation
            current_idx = (current_idx - 1) % num_images_visualise

    cv2.destroyAllWindows()


# Example usage
if __name__ == '__main__':
    project_dir = '/home/iamshri/Documents/Dataset/Legos'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LegoDataset(project_dir, transform=transform)
    num_images_visualise = 100

    visualise_dataset(dataset, num_images_visualise)
