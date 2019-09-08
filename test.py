import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import copy
import numpy as np
import optuna
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

import time
import cv2

class MyDataSet(Dataset):
    def __init__(self, csv_path, root_dir):
        self.image_dataframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.images = os.listdir(self.root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 画像読み込み
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, image_name))
        image = image.convert('RGB')  # PyTorch 0.4以降
        # label (0 or 1 or 2 or 3 or 4 or 5)
        label = self.image_dataframe.query('ImageName=="' + image_name + '"')['ImageLabel'].iloc[0]
        return self.transform(image), int(label)


imgDataset = MyDataSet('newgame.csv', 'data/')

_ , test_data = train_test_split(imgDataset, test_size=0.2)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.activation = F.elu
        # self.activation = trial.suggest_categorical('activation', [F.relu, F.elu])
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5)  # 64*64*3 -> 60*60*30
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5)  # 30*30*30 -> 26*26*60
        self.conv2_drop = nn.Dropout2d(p=0.345478067932839)  # 0〜0.8の間でサンプリング
        self.fc1 = nn.Linear(13 * 13 * 60, 150)
        self.fc2 = nn.Linear(150, 6)

    def forward(self, x):
        x = self.activation(F.max_pool2d(self.conv1(x), 2))
        x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 13 * 13 * 60)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_final(model, device, test_loader):
    model.eval()
    correct = 0
    name_list = ["aoba", "hajime", "hihumi", "kou", "rin", "yun"]
    transform = transforms.Compose([transforms.ToPILImage()])
    with torch.no_grad():  # 計算グラフを作らない
        for data, target in test_loader:
            data_org = copy.deepcopy(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            for i in range(data.size()[0]):
                image = transform(data_org[i])
                image = np.array(image)
                image = image[:,:,::-1]
                image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
                # image.show()
                # time.sleep(3)
                print("target : {}".format(name_list[target[i]]))
                print("pred : {}".format(name_list[pred[i]]))

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(image, '{}'.format(name_list[pred[i]]), (10, 150), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, '{}'.format(name_list[pred[i]]), (10, 180), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('result', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print('Accuracy : {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load('model'))

test_final(model, device, test_loader)
