"""

This file was originally inplemented in the EfficienctDet repository.
"https://github.com/xuannianz/EfficientDet"

**There are no big changes have been made in original implementation.

Note: This file will not be calculating the mean average precision (mAP), while,
1- Load the images
2- Make predictions
3- Draw predicted bbox and class labeles on corresponding image
4- Save it to directory names "eveluated_images"

"""
"""
Usage:
    1- Please create a directory in which your test images will be located. For instance
        "/datasets/config_folder/dataset/testset_images/"
    
    2- "checkpoints" is the base directory for model weight matrix. While, models are saved in sub-directoryies. So, give the path of your model dierectory as
    "2020-07-26 (B1 MND WBiFPN Perfect)"
    
    3- Give your .h5 (ckp) name in directory as
    "csv_300_0.0398_0.6976.h5"
    
    4- Write a jason file for classes names and labels in correct order with your classes.csv file and give it to "classes" variable in "eveluate_testset()" function 
    5- The eveluated image will be saved in "evaluated_images"
    
"""
# Import the necessary libraries

import cv2
import json
import numpy as np
import os
import time
import glob


# Import models and prediction processing tools
from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


class TestsetEveluation:
    def __init__(self, relative_path: str = "/datasets/config_folder/dataset/") -> None:
        self.current_path = os.getcwd()
        self.base_path = self.current_path + relative_path
        self.ckp_model_dir = "2020-08-04 (B1 MND WBiFPN LLR)"
        self.ckp_model_file = "csv_298_0.0389_0.6384.h5"

        # Created the directory where the eveluated images will be saved
        if not os.path.exists("evaluated_images"):
            os.mkdir("evaluated_images")

    def eveluate_testset(self, ckp_path: str = "checkpoints/"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        phi = 1
        weighted_bifpn_flag = True
        model_path = ckp_path + self.ckp_model_dir + "/" + self.ckp_model_file
        image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
        image_size = image_sizes[phi]
        # my dataset classes
        classes = {
            value["id"] - 1: value["name"]
            for value in json.load(
                open("mine_classes_eveluation_ds.json", "r")
            ).values()
        }
        num_classes = 13
        score_threshold = 0.50
        colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
        _, model = efficientdet(
            phi=phi,
            num_classes=num_classes,
            weighted_bifpn=weighted_bifpn_flag,
            score_threshold=score_threshold,
        )
        model.load_weights(model_path, by_name=True)

        counter = 0
        for image_path in glob.glob(self.base_path + "testset_images/*.jpg"):

            image = cv2.imread(image_path)
            src_image = image.copy()
            # BGR -> RGB
            image = image[:, :, ::-1]
            h, w = image.shape[:2]

            image, scale = preprocess_image(image, image_size=image_size)
            # run network
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(
                [np.expand_dims(image, axis=0)]
            )
            boxes, scores, labels = (
                np.squeeze(boxes),
                np.squeeze(scores),
                np.squeeze(labels),
            )
            print(time.time() - start)
            boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

            # select indices which have a score above the threshold
            indices = np.where(scores[:] > score_threshold)[0]

            # select those detections
            boxes = boxes[indices]
            labels = labels[indices]

            draw_boxes(src_image, boxes, scores, labels, colors, classes)

            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imwrite(
                "evaluated_images/evaluated_image_" + str(counter) + ".jpg", src_image
            )
            counter += 1
            # cv2.imshow('image', src_image)
            # cv2.waitKey(0)

            if counter == 500:
                break
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    te = TestsetEveluation()

    te.eveluate_testset()
