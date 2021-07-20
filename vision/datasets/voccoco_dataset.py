import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VOCCOCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"        
        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            # self.class_names = ('BACKGROUND',
            # 'aeroplane', 'bicycle', 'bird', 'boat',
            # 'bottle', 'bus', 'car', 'cat', 'chair',
            # 'cow', 'diningtable', 'dog', 'horse',
            # 'motorbike', 'person', 'pottedplant',
            # 'sheep', 'sofa', 'train', 'tvmonitor')

            self.class_names = ('BACKGROUND',
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')        
        
        
        self.ids = self._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            # print(image_id)
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    
    def _read_image_ids(self, image_sets_file):
        valid_inds = []
        with open(image_sets_file) as f:
            for line in f:
                image_id = line.rstrip()
                if not self.is_test:  # filter images without gt_bbox
                    annotation_file = self.root / f"Annotations/{image_id}.xml"
                    objects = ET.parse(annotation_file).findall("object")
                    for object in objects:
                        _class_name = object.find('name')
                        if _class_name  is not None:
                            class_name = _class_name.text.lower().strip()
                            if class_name in self.class_names:
                                valid_inds.append(image_id)
                                break 
               
                else:
                    valid_inds.append(image_id)
        return valid_inds

    # def _filter_imgs(self):
    #     """Filter images  without annotation."""
        
    #     valid_inds = []
    #     for i, img_info in enumerate(self.data_infos):


    #         img_id = img_info['id']
    #         xml_path = os.path.join(self.img_prefix, 'Annotations',
    #                             f'{img_id}.xml')
    #         tree = ET.parse(xml_path)
    #         root = tree.getroot()
    #         for obj in root.findall('object'):
    #             name = obj.find('name').text
    #             if name in self.CLASSES:
    #                 valid_inds.append(i)
    #                 break

    #     return valid_inds


        

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            _class_name = object.find('name')
            if _class_name  is not None:
                class_name = _class_name.text.lower().strip()
            else:
                continue

            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



