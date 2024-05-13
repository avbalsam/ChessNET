"""This module implements the ChessRecognitionDataset class."""
import json
from pathlib import Path
from typing import Callable, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

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


# def order_points(pts):
#         # order a list of 4 coordinates:
#         # 0: top-left,
#         # 1: top-right
#         # 2: bottom-right,
#         # 3: bottom-left

#         rect = np.zeros((4, 2), dtype = "float32")
#         try:
#             s = np.sum(pts, axis = 1)
#         except Exception as e:
#             print("couldn't order pts of shape", pts.shape)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
        
#         diff = np.diff(pts, axis = 1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
        
#         return rect

# def detect_corners(image):
#     """
#     Detects the corners of a given image. Set show_image to True to display the image with corners and intersections marked.
#     """
#     corner_model = get_model(model_id="chessnet-corner/4", api_key = "BH54WGNf3SHi0kl69JH0")
#     results = corner_model.infer( image=image.rotate(-90, expand=True), confidence=0.01, iou_threshold=0.01)[0].predictions
#     corners = np.array([(box.x, box.y) for box in results])
#     corners = order_points(corners)

#     return corners

# # perspective transforms an image with four given corners

# def four_point_transform(image, pts):

#     # img = Image.open(image)
#     image = asarray(image)

#     rotated_img = img.rotate(-90, expand=True)
#     image = asarray(rotated_img)

#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect

#     # compute the width of the new image
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))


#     # compute the height of the new image
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))

#     # construct set of destination points to obtain a "birds eye view"
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype = "float32")

#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

#     img = Image.fromarray(warped)
#     #img = Image.fromarray(warped, "RGB") #to make it black and white as original
#     # img.show()
#     # return the warped image
#     return img

# def perspective_transform(img):
#     pts = detect_corners(img)
#     if pts.shape != (4,2):
#         print("not transformed", pts.shape)
#         return img
#     else:
#         print("transformed")
#         return four_point_transform(img, pts)
    

def get_square_from_image(projected_image: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """
    Given a projected image of a chessboard, gets an image of the chessboard square at (row, col)
    on the chessboard (with 1/4 padding on each side).

    Args:
        projected_image: Tensor representing an image of a chessboard.
        row: Row of the image to get, where 0 <= row < 8
        col: Column of the image to get, where 0 <= col < 8

    Returns:
        images: An image representing the cell at (row, col) on the chessboard grid
    """
    assert 0 <= row < 8
    assert 0 <= col < 8

    projected_image: Image = to_pil_image(projected_image)

    image_width = projected_image.size[0]
    image_height = projected_image.size[1]
    cell_width, cell_height = image_width/8, image_height/8 # width and height of a single chessboard square

    height_offset = 1/4 * cell_height
    width_offset = 1/4 * cell_width

    start_width = max(col * cell_width - width_offset, 0)
    start_height = max(row * cell_height - height_offset, 0)
    end_width = min((col+1) * cell_width + width_offset, image_width)
    end_height = min((row+1) * cell_height + height_offset, image_height)

    sub = projected_image.crop((start_width, start_height, end_width, end_height))
    sub = sub.resize((int(cell_width + 2 * width_offset), int(cell_height + 2 * height_offset)))

    return pil_to_tensor(sub).to(torch.float32)


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
        # self.length = annotations_file['splits'][split]['n_samples'] * 64 # multiply by 64 since each chessboard image has 64 squares
        self.split_img_ids = annotations_file['splits'][split]['image_ids']

        # Remove any images which do not have a corresponding transformed image
        self.split_img_ids = [img_id for img_id in self.split_img_ids if os.path.exists(str(self.get_path(img_id)).replace("images", "projected_imgs"))]
        assert(len(self.split_img_ids) > 0)

        self.length = len(self.split_img_ids) * 64

        # Keep only the split's data with a corresponding transformed file (corner detection worked well)
        self.annotations = self.annotations[self.annotations["image_id"].isin(
            self.split_img_ids)]
        self.images = self.images[self.images['id'].isin(self.split_img_ids)]

        # This nasty one-liner gets the path to the images by class. The i-th index of images_by_category
        # contains a list of the images in the i-th class. Unfortunately, it's not working yet
        # self.images_by_category = [
        #     list(set(map(
        #         lambda img_id: "N/A" if len(images := list(map(lambda s: s.replace("images", "projected_imgs"), self.images[self.images["id"] == img_id]["path"].tolist()))) == 0 else images[0],
        #         self.annotations[self.annotations["category_id"] == category_id]["image_id"].tolist()
        #     )).difference({"N/A"})) for category_id in range(12)
        # ]

        # for each image in self.images, add 64 images to self.squares with the correct annotation (using the logic from __getitem__)
        self.squares = [] # a list of tuples ((img_path, row, col), ann), where img is the path to an image of a single chessboard square, row and col are coordinates in [0,7] and ann is the identity of the piece in the image (one-hot encoding)
        for img_id in self.images["id"].tolist():
            for i in range(64):
                row, col = divmod(i, 8)

                # LOAD IMAGE
                img_path = Path(
                    self.dataroot,
                    self.images[self.images['id'] == img_id].path.values[0])
                img_path = str(img_path).replace("images", "projected_imgs") # get from the projected_imgs directory instead of the images directory

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
                # img_anns = img_anns.flatten().float()

                ann = img_anns[row * 8 + col]

                self.squares.append(((img_path, row, col), ann))

        self.imgs_by_category = [[(img_info, ann) for (img_info, ann) in self.squares if np.argmax(ann.numpy()) == category_id] for category_id in range(12)] # blank squares do not have a category
        print(list(map(len, self.imgs_by_category)))

        self.class_weights = [ # the weights with which we should sample each class
            5,
            2,
            1.5,
            1.5,
            1,
            1,
        ] * 2
        self.class_weights.append(64 - sum(self.class_weights))
        self.class_weights[-1] /= 2 # weight the blanks lower so that we train faster (we do perfectly on them anyway)

        assert (self.length == len(self.split_img_ids) * 64 and
                self.length == len(self.images) * 64), (
            f"The numeber of images in "
            f"the dataset ({len(self.images) * 64}) for split:{self.split}, does "
            f"not match neither the length specified in the annotations "
            f"({self.length}) or the length of the list of ids for the split "
            f"{len(self.split_img_ids) * 64}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length
    
    def get_path(self, img_id) -> Path:
        """
        Returns the path to the image specified by img_id (as specified in self.images)
        """
        return Path(
            self.dataroot,
            self.images[self.images['id'] == img_id].path.values[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a dataset sample.

        Args:
            index (int): Index of the sample to return. Index % 64 is the number 
                square of the board to query (top to bottom, left to right),
                and index // 64 is the number of the chessboard image.

        Returns:
            img (Tensor): A 3xHxW Tensor of the image corresponding to 
                the chessboard square at `index`.
            img_anns (Tensor): A (13,)-shape Tensor containing the annotation for
                               the chessboard square in one-hot encoding.
        """
        img_idx, square_idx = divmod(index, 64)
        row, col = divmod(square_idx, 8)
        # LOAD IMAGE
        img_id = self.split_img_ids[img_idx]
        img_path = Path(
            self.dataroot,
            self.images[self.images['id'] == img_id].path.values[0])
        img_path = str(img_path).replace("images", "projected_imgs") # get from the projected_imgs directory instead of the images directory
        img = read_image(img_path).float()

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
        # img_anns = img_anns.flatten().float()

        square_img = get_square_from_image(img, row, col)
        ann = img_anns[square_idx]

        # return (img, img_anns)
        return (square_img, ann)
