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
        print(pts)
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
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
    results = corner_model.infer( image=Image.open(image).rotate(-90, expand=True), confidence=0.01, iou_threshold=0.01)[0].predictions
    corners = np.array([(box.x, box.y) for box in results])
    corners = order_points(corners)

    return corners

# perspective transforms an image with four given corners

def four_point_transform(image, pts):

    img = Image.open(image)
    #image = asarray(img)

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

# calculates chessboard grid

def plot_grid_on_transformed_image(image):
    
    corners = np.array([[0,0], 
                    [image.size[0], 0], 
                    [0, image.size[1]], 
                    [image.size[0], image.size[1]]])
    
    corners = order_points(corners)

    figure(figsize=(10, 10), dpi=80)

    # im = plt.imread(image)
    implot = plt.imshow(image)
    
    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL


def perspective_projection(img_filename, show=False):
     """
     Transforms the chessboard image with filename img_filename into a new board with everything 
     but corners cropped and perspective projected, then returns the transformed image (does not
     mutate the original image).

     Args:
        img_filename: path of the image to modify
        show: Whether the display a plot of the new image with gridlines

     Returns:
        New image representing the image at img_filename with perspective projection.
     """
     corners = detect_corners(img_filename)
     projected_image = four_point_transform(img_filename, corners)

     if show:
          plot_grid_on_transformed_image(projected_image)
    
     return projected_image
