import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

from shapely.geometry import Polygon
import torch
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dataset import ChessRecognitionDataset
from train import ChessNET
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
            
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

            # # knight on bottom left
            # # board_arr = [[num_to_piece[torch.argmax(labels[0][13 * (8 * row + col) : 13 * (8 * row + col + 1)]).item()] for col in range(8)] for row in range(8)]
            # print("inp shape", inputs.shape)
            # print("labels shape", labels.shape)
            # print("label", labels[0])
            # # pprint(board_arr)

            # plt.imshow(np.array(inputs[0][0]), cmap="grey")
            # plt.show()

if __name__ == "__main__":

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

    # compute accuracy of model
    print("total number of images:", len(val_dataloader))

    model = ChessNET.load_from_checkpoint("/Users/avbalsam/Desktop/6.8301/real-life-chess-vision/app_v2/lightning_logs/version_22/checkpoints/last.ckpt")
    model.eval()
    num_correct = 0
    total = 0
    correct_by_class = [0 for _ in range(len(num_to_piece))]
    total_by_class = [0 for _ in range(len(num_to_piece))]
    type_pred, type_label = [], []
    color_pred, color_label = [], []

    for inputs, labels in val_dataloader:
        
        assert len(inputs) == len(labels) == 1, \
            "the val dataloader should only have one image per batch"
        inp = inputs
        label = labels[0]
        correct_piece_number = np.argmax(label).item()

        inp = inp.to(torch.device("mps"))

        res = model(inp)
        res = res[0].cpu().detach().numpy()
        pred_piece_name = num_to_piece[np.argmax(res)]
        color_pred.append('W' if pred_piece_name.isupper() else 'B')
        type_pred.append(pred_piece_name.upper())
        correct = pred_piece_name == num_to_piece[correct_piece_number]
        color_label.append('W' if num_to_piece[correct_piece_number].isupper() else 'B')
        type_label.append(num_to_piece[correct_piece_number].upper())

        print(correct, "num_correct:",num_correct,"total:",total,"accuracy by class", [round(c / t, 2) if t > 0 else 0 for c,t in zip(correct_by_class, total_by_class)],"accuracy:",round(num_correct/total,5) if total > 0 else 0, end='\r')
        if correct:
            num_correct += 1
            correct_by_class[correct_piece_number] += 1
        total += 1
        total_by_class[correct_piece_number] += 1

    print("model accuracy: %d".format((correct / total) * 100))
    cf_matrix_type = confusion_matrix(type_label, type_pred)
    classes = (piece.upper() for piece in num_to_piece.values())
    df_cm = pd.DataFrame(cf_matrix_type / np.sum(cf_matrix_type, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('type_heatmap.png')

    cf_matrix_color = confusion_matrix(color_label, color_pred)
    classes = ('B', 'W')
    df_cm = pd.DataFrame(cf_matrix_color / np.sum(cf_matrix_color, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('color_heatmap.png')