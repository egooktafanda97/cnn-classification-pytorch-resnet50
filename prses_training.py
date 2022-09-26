# cnn Proses
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import time
import copy
import os
import sys
import json
import random
import string
from datetime import date

today = date.today()
dir_path = os.path.dirname(os.path.realpath(__file__)) #dir file
# dataset
input_path = dir_path+"/dataset"+"/" #directory dataset
input_size = 224 # image size
ep = 2 #epoch
lr = 0.001 
batch_size = 32 # jumlahd gambar 1x proses epoch
num_clases = 3 # jumlah kelas
feature_extract = True # widget ResNet50

 

# augmentasi ============================================================

# aktifkan normalize jika data perlu di normalize.
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

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
# =======================================================================================

# cek device, untuk resNet50 direkomendasikan menggunakan GPU (cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# inisialisasi model ResNet50 ==========================================================
model = models.resnet50(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False   
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_clases)).to(device)
# =======================================================================================


# muali training =========================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(),lr=lr)

train_losses, val_losses = [], []
train_acc, val_acc = [], []
time_lear = []
response_train = []

def train_model2(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        since1 = time.time()
        t_loss,t_acc = [],[]
        v_loss,v_acc = [],[]
        _best_iterasi = []
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)  
                t_loss = epoch_loss
                t_acc = epoch_acc
                
            else:
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
                v_loss = epoch_loss
                v_acc = epoch_acc
                
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                _best_iterasi = epoch 
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                
        _time_elapsed = time.time() - since1
        response = {
            "tipe"              : "train",
            "_train_loss"       : t_loss,
            "_train_acc"        : t_acc.cpu().item(),
            "_validation_loss"  : v_loss,
            "_validation_acc"   : v_acc.cpu().item(),
            "_epochs"           : epoch + 1,
            "epochs"            : num_epochs,
            "_timer"            : [_time_elapsed // 60, _time_elapsed % 60],
            "_best_iterasi"     : _best_iterasi
        }
        response_train.append(response)
        print(response)

    time_elapsed = time.time() - since
    time_lear.append(time_elapsed)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model_trained = train_model2(model,dataloaders,criterion, optimizer, num_epochs=ep)


_history = {
    "_train_loss" : train_losses,
    "_train_acc"  : [x.cpu().item() for x in train_acc],
    "_validation_loss" : val_losses,
    "_validation_acc"  : [x.cpu().item() for x in val_acc],
    "input_size"        : input_size,
    "epochs"           : ep,
    "batch_size"       : batch_size,
    "num_class"        : num_clases,
    "dataset_path"     : input_path,
    "return"           : response_train
}

def random_char(y):
       return ''.join(random.choice(string.ascii_letters) for x in range(y))
   
def save_checkpoint(model, optimizer,history, save_path, epoch):
    torch.save({
        'model_state_dict': model[0].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history' : history,
        'val_acc' : model[1],
        'epoch': epoch,
        "model_"    : model_names,
        "tanggal"   : today.strftime("%d-%m-%Y"),
        "tipe"      : "result",
    }, save_path)


dir_path = os.path.dirname(os.path.realpath(__file__))

model_names = "weight__"+random_char(3)+".h5"
save_path = dir_path+"/model/"+model_names
save_checkpoint(model_trained,optimizer,_history,save_path,ep)
# ======================================================================================================


# cekpoin ==============================================================================================
# load model
model = models.resnet50(pretrained=False).to(device)
mod = torch.load(save_path);
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 3)).to(device)
model.load_state_dict(mod['model_state_dict'])

_train_loss = [x['_train_loss'] for x in mod['history']['return']]
_train_acc = [x['_train_acc'] for x in mod['history']['return']]
_validation_loss = [x['_validation_loss'] for x in mod['history']['return']]
_validation_acc = [x['_validation_acc'] for x in mod['history']['return']]
_history = {
    "_train_loss" : _train_loss,
    "_train_acc"  : _train_acc,
    "_validation_loss" : _validation_loss,
    "_validation_acc"  : _validation_acc,
    "epochs"           : mod['history']['epochs']
}

NUM_EPOCHS = _history["epochs"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
t_loss = _history["_train_loss"]
v_loss = _history["_validation_loss"]
t_acc = _history["_train_acc"]
v_acc =  _history["_validation_acc"]
# print(NUM_EPOCHS)


ax[0].plot(np.arange(NUM_EPOCHS), t_loss,label='Training loss')
ax[0].plot(np.arange(NUM_EPOCHS), v_loss, label='validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Cross entropy loss')
ax[0].legend()


ax[1].plot(np.arange(NUM_EPOCHS),t_acc, label='Training accuracy')
ax[1].plot(np.arange(NUM_EPOCHS),v_acc, label='validation accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy %')
ax[1].legend()
plt.show()
# =====================================================================================

