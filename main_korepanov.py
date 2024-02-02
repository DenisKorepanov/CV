import cv2
import glob
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    # YOUR CODE HERE

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light_orange = (0, 169, 150)
    dark_orange = (90, 255, 255)
    light_white = (70, 0, 200)
    dark_white = (155, 170, 255)
    orange_mask_1 = cv2.inRange(img_hsv, light_orange, dark_orange)
    white_mask_2 = cv2.inRange(img_hsv, light_white, dark_white)
    mask_sum = cv2.bitwise_or(orange_mask_1, white_mask_2)

    return mask_sum


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'
    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)


    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
        print(compute_ious(masks, osp.join("dataset", stage, "masks")))
        
