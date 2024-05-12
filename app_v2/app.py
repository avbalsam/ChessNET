from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt


import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

import cv2

from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
from inference_sdk import InferenceHTTPClient
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from inference import get_model
from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os

from corner_detection import perspective_projection
from dataset import ChessRecognitionDataset

class PieceClassifier(nn.Module):
    def __init__(self):
        super(PieceClassifier, self).__init__()
        self.piece_classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer
        self.piece_classifier.fc = nn.Linear(self.piece_classifier.fc.in_features, 8) #[P, H, B, R, Q, K, Blank, Color (0/W - 1/B)]

    def forward(self, x):
        x = self.piece_classifier(x)
        softmax_output = F.softmax(x[:, :7], dim=1)  # Apply softmax to the first 7 outputs
        relu_output = F.relu(x[:, 7])  # Apply ReLU to the 8th output
        final_output = torch.cat((softmax_output, relu_output.unsqueeze(1)), dim=1)  # Concatenate softmax and ReLU outputs
        return final_output

    def calculate_loss(self, output, target):
        softmax_output = output[:, :7]  # First 7 outputs
        relu_output = output[:, 7]  # 8th output

        # Calculate Negative Log Likelihood (NLL) loss for softmax outputs
        nll_loss = F.nll_loss(F.log_softmax(softmax_output, dim=1), target[:, 7].to(torch.long))

        # Calculate Mean Squared Error (MSE) loss for ReLU output
        is_not_blank = (target[:, 6] != 1)
        mse_loss = F.mse_loss(relu_output, target[:, 7]) * is_not_blank #ignore the loss value here for blank space
        # Total loss
        total_loss = nll_loss + mse_loss
        return total_loss

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def fen_to_board(fen):
    board = []
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( [" "] * int(c) )
            else:
                brow.append(c)

        board.append( brow )
   # print(board)
    return [[element for element in row] for row in board]

from pprint import pprint
pprint( fen_to_board(fen) )

#train_data = [
#    (
#        transforms.ToTensor()(training_data_temp[0][0]).unsqueeze(0),  # Input tensor
#        torch.tensor([[1, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)  # Target tensor
#    )
#]

def avi_to_vec(inp): #avi character to vector; #[P, N, B, R, Q, K, ' ', Color (0/W - 1/B)] ["lowercase is white"]
    v = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    v[0][-1] = (0 if inp.islower() else 1)
    map = {"p": 0, "n": 1, "b": 2, "r": 3, "q":4, "k":5,  " ": 6}
    if inp in map: v[0][map[inp]] = 1
    elif inp.lower() in map: v[0][map[inp.lower()]] = 1
    return v

def vec_to_avi(v):
    c = ' '
    map = {0:"p", 1:"n", 2:"b", 3:"r", 4:"q", 5:"k",  6:" "}
    for i in range(7):
        if v[0][i] == 1:
            c = map[i]
            break
    if v[0][-1] == 1: c = c.upper()
    return c
    
def crop_board(projected_image):
    images = []
    w, h = projected_image.size[0]/8, projected_image.size[1]/8, 
    for i in range(8):
        for j in range(8):
            sub = projected_image.crop((j * w, i * h, (j+1) * w, (i+1) * h))
            sub = sub.resize((int(w), int(h)))
            images.append(sub)
    return images

# def create_data_pairs(image_filename, fen_notation):
#     projected_image = perspective_projection(image_filename)
#     labels = [avi_to_vec(element) for element in np.array(fen_to_board(fen_notation)).flatten()] #changing data to vector format
#     images = [transforms.ToTensor()(img).unsqueeze(0) for img in crop_board(projected_image)] #changing images to proper tensors; remove for nice pics
#     return list(zip(images, labels))

if __name__ == "__main__":
    num_epochs = 30

    chess_image_transform = transforms.Compose([
            transforms.Resize(1024, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])
    train_dataset = ChessRecognitionDataset(
        dataroot="/Users/avbalsam/Desktop/6.8301/real-life-chess-vision/app_v2/chessred", 
        split="train", 
        transform=chess_image_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataset = ChessRecognitionDataset(
        dataroot="/Users/avbalsam/Desktop/6.8301/real-life-chess-vision/app_v2/chessred",
        split="val",
        transform=chess_image_transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataset = ChessRecognitionDataset(
        dataroot="/Users/avbalsam/Desktop/6.8301/real-life-chess-vision/app_v2/chessred",
        split="test",
        transform=chess_image_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for epoch in range(num_epochs):
        i = 0
        for inputs, labels in train_dataloader:
            i += 1
            print(i)
            if (i == 25):
                num_to_piece = {
                    0: "P",
                    1: "R",
                    2: "N",
                    3: "B",
                    4: "Q",
                    5: "K",
                    6: "p",
                    7: "r",
                    8: "n",
                    9: "b",
                    10: "q",
                    11: "k",
                    12: " ",
                }
                board_arr = [[num_to_piece[torch.argmax(labels[0][13 * (8 * row + col) : 13 * (8 * row + col + 1)]).item()] for col in range(8)] for row in range(8)]
                print("inp shape", inputs.shape)
                print("labels shape", labels.shape)
                pprint(board_arr)

                print(labels[0][0:13])
                print(labels[0][13:26])

                plt.imshow(np.array(inputs[0][0]))
                plt.show()
                exit(0)
