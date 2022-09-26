import numpy as np
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import os
import sys
import json
import random
import string
from datetime import date
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

dir_path = os.path.dirname(os.path.realpath(__file__))
# inisilaisasi
input_path = dir_path+"/dataset/"
input_size = 224
batch_size = 32 #sesuaikan dengan konfigurasi train
num_clases = 3 #sesuaikan dengan konfigurasi train
feature_extract = True
model_name = r''+dir_path+"/model/weight__iZY.h5"
dataset_short_path = r''+"/dataset/validation/"
main_data = r''+dir_path+"/dataset/validation/"
path_to = r''+dir_path+"/dataset/validation/"


# augmentasi
data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        # normalize
    ]),
}


image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# models
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_clases)).to(device)
model.load_state_dict(torch.load(model_name)['model_state_dict'])


# prediksi
def pre_image(image_path,short_path,model):
    print(image_path)
    img = Image.open(r''+image_path)
    transform_norm = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224,224))
    ])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    img_normalized = img_normalized.to(device)
    # print(img_normalized.shape)
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        sofmax = F.softmax(output, dim=1).cpu().data.numpy()
        #   index = output.data.cpu().numpy().argmax()
        index =sofmax.argmax()
        classes = image_datasets["validation"].classes
        class_name = classes[index]
        num_predik = output.cpu().numpy()[0].tolist()
        return {
            "type"       : "prediksi", 
            "prediksi"   : class_name,
            "predik"     : sofmax[0].tolist(),
            "img_path"   : short_path,
            "clases"     : classes,
            "obj_predik" : [{"class" : classes[num_predik.index(x)],"val": x} for x in  num_predik]
        }

def data_test(dataset,model):
    d = []
    for direc in os.listdir(dataset):
        print(direc)
        path_image_file = dataset+"/"+direc
        for im in os.listdir(path_image_file):
            short_path = main_data+"/"+direc
            pth = short_path+"/"+im
            predict_class = pre_image(pth,short_path,model)
            print(predict_class)
            d.append(predict_class)
    return d

data_test(path_to,model)
