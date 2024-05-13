import cv2
from ultralytics import YOLO


class YOLO_ISC:
    def __init__(self, mode='' , model_path=None, epoch=100, imgsz=640):
        self.model = YOLO(trained_model)
        self.model_path = model_path
        self.mode = mode
        self.epoch = epoch
        self.imgsz = imgsz

    def run(self):
        if self.mode == 'train':
            results = model.train(data='config.yaml', epochs=self.epoch, imgsz=self.imgsz)
        elif self.mode == 'test':
            model.predict('/home/alien_arise/Dataset/mixed_validation_set/images', conf=0.75, show_labels=True,
                          save=True, save_conf=True, save_txt=True)
        else:
            raise Exception


if __name__ == '__main__':
    model_path = '/home/qub-hri/PycharmProjects/LegoDetect/runs/detect/train8/weights/Lego_YOLO.pt'
    model = YOLO_ISC(mode='test', model_path=model_path, epoch=100, imgsz=640)
    model.run()
