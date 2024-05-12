"""This module implements the ChessRecognitionDataset class."""
import json
from pathlib import Path
from typing import Callable, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
from inference import get_model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os


def order_points(pts):
        # order a list of 4 coordinates:
        # 0: top-left,
        # 1: top-right
        # 2: bottom-right,
        # 3: bottom-left

        rect = np.zeros((4, 2), dtype = "float32")
        try:
            s = np.sum(pts, axis = 1)
        except Exception as e:
            print("couldn't order pts of shape", pts.shape)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

def detect_corners(image):
    """
    Detects the corners of a given image. Set show_image to True to display the image with corners and intersections marked.
    """
    corner_model = get_model(model_id="chessnet-corner/4", api_key = "BH54WGNf3SHi0kl69JH0")
    results = corner_model.infer( image=image.rotate(-90, expand=True), confidence=0.01, iou_threshold=0.01)[0].predictions
    corners = np.array([(box.x, box.y) for box in results])
    corners = order_points(corners)

    return corners

# perspective transforms an image with four given corners

def four_point_transform(image, pts):

    # img = Image.open(image)
    image = asarray(image)

    rotated_img = img.rotate(-90, expand=True)
    image = asarray(rotated_img)

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    img = Image.fromarray(warped)
    #img = Image.fromarray(warped, "RGB") #to make it black and white as original
    # img.show()
    # return the warped image
    return img

def perspective_transform(img):
    pts = detect_corners(img)
    if pts.shape != (4,2):
        print("not transformed", pts.shape)
        return img
    else:
        print("transformed")
        return four_point_transform(img, pts)


class ChessRecognitionDataset(Dataset):
    """ChessRecognitionDataset class.

    The ChessRecognitionDataset class implements a custom pytorch dataset.
    """

    def __init__(self,
                 dataroot: Union[str, Path],
                 split: str,
                 transform: Union[Callable, None] = None) -> None:
        """Initialize a ChessRecognitionDataset.

        Args:
            dataroot (str, Path): Path to the directory containing the Chess
                                  Dataset.
            transform (callable, optional): Transform to be applied on the
                                            image samples of the dataset.
        """
        super(ChessRecognitionDataset, self).__init__()

        self.dataroot = dataroot
        self.split = split
        self.transform = transform

        # Load annotations
        data_path = Path(dataroot, "annotations.json")
        if not data_path.is_file():
            raise (FileNotFoundError(f"File '{data_path}' doesn't exist."))

        with open(data_path, "r") as f:
            annotations_file = json.load(f)

        # Load tables
        self.annotations = pd.DataFrame(
            annotations_file["annotations"]['pieces'],
            index=None)
        self.categories = pd.DataFrame(
            annotations_file["categories"],
            index=None)
        self.images = pd.DataFrame(
            annotations_file["images"],
            index=None)

        # Get split info
        self.length = annotations_file['splits'][split]['n_samples']
        self.split_img_ids = annotations_file['splits'][split]['image_ids']

        # Keep only the split's data
        self.annotations = self.annotations[self.annotations["image_id"].isin(
            self.split_img_ids)]
        self.images = self.images[self.images['id'].isin(self.split_img_ids)]

        assert (self.length == len(self.split_img_ids) and
                self.length == len(self.images)), (
            f"The numeber of images in "
            f"the dataset ({len(self.images)}) for split:{self.split}, does "
            f"not match neither the length specified in the annotations "
            f"({self.length}) or the length of the list of ids for the split "
            f"{len(self.split_img_ids)}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a dataset sample.

        Args:
            index (int): Index of the sample to return.

        Returns:
            img (Tensor): A 3xHxW Tensor of the image corresponding to `index`.
            img_anns (Tensor): A 64x13 Tensor containing the annotations for
                               each of the chessboard's squares in one-hot
                               encoding.
        """
        # LOAD IMAGE
        img_id = self.split_img_ids[index]
        img_path = Path(
            self.dataroot,
            self.images[self.images['id'] == img_id].path.values[0])

        img = read_image(str(img_path)).float()
        # from torchvision.transforms import Resize, ToTensor
        # # Add perspective transform
        # corners = detect_corners(img_path)
        # if len(corners) != 4:
        #     print("bad corners")
        # else:
        #     img = Resize((1024, 1024))(four_point_transform(img_path, corners))
        #     img = ToTensor()(img)

        if self.transform is not None:
            img = self.transform(img)

        # GET ANNOTATIONS
        cols = "abcdefgh"
        rows = "87654321"

        empty_cat_id = int(
            self.categories[self.categories['name'] == 'empty'].id.values[0])

        img_anns = self.annotations[
            self.annotations['image_id'] == img_id].copy()

        # Convert chessboard positions to 64x1 array indexes
        img_anns['array_pos'] = img_anns["chessboard_position"].map(
            lambda x: 8*rows.index(x[1]) + cols.index(x[0]))

        # Keep columns of interest
        img_anns = pd.DataFrame(
            img_anns['category_id']).set_index(img_anns['array_pos'])

        # Add category_id for 'empty' in missing row indexes and create tensor
        img_anns = torch.tensor(list(img_anns.reindex(
            range(64), fill_value=empty_cat_id)['category_id'].values))

        img_anns = F.one_hot(img_anns)
        img_anns = img_anns.flatten().float()

        return (img, img_anns)
