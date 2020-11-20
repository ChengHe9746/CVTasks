import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import cv2
import torchvision
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import os
import random
from sklearn.model_selection import train_test_split
from models import Vgg16Conv
from models import Vgg16Deconv
from utils import decode_predictions

def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)

            img = cv2.resize(img, (RESIZE,RESIZE))
            img = np.rollaxis(img, 2, 0)

            img = img.astype(np.float32)
            img = img / 127.5 - 1  # -1~1

            IMG.append(np.array(img))
    return IMG

benign = np.array(Dataset_loader('./data/benign',224))
malign = np.array(Dataset_loader('./data/malignant',224))

benign_label = np.zeros(len(benign))
malign_label = np.ones(len(malign))

X = np.concatenate((benign, malign), axis = 0)
Y = np.concatenate((benign_label, malign_label), axis = 0)

x_train, val_x, y_train, val_y = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=11
)

x_train=torch.Tensor(x_train)
y_train=torch.Tensor(y_train)
val_x=torch.Tensor(val_x)
val_y=torch.Tensor(val_y)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True
)
test_dataset = torch.utils.data.TensorDataset(val_x, val_y)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=True
)

net = Vgg16Conv()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(),lr=0.05)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
ckpt_dir = "ckpt"
best_acc = 0
for epoch in range(25):
    train_accs = []
    test_accs = []
    train_loss = []
    running_loss = 0.0
    if epoch in [5, 25]:
        optimizer.param_groups[0]['lr'] *= 0.6
    for i, data in enumerate(train_loader, 0):  
        #         inputs,labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d,%5d] loss :%.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
        train_loss.append(loss.item())

        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)  
        correct = (predicted == labels).sum().item()
        if i % 20 == 19:
            print('[%d,%5d] correct :%.3f' %
                  (epoch + 1, i + 1, correct/ total))  
        train_accs.append(correct / total)
        
    epoch_loss = np.mean(train_loss)
    epoch_accs = np.mean(train_accs)
    print("loss[seg]: {:0.4f}, accs: {:0.4f}".format(epoch_loss, epoch_accs))
    

    net.eval()
    correct = 0
    test_acc = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        pred = net(inputs)
        correct = 0
        total = 0
        _, predicted = torch.max(pred.data, 1)
        total = labels.size(0)  
        correct = (predicted == labels).sum().item()
  
        test_accs.append(correct / total)
    test_acc = np.mean(test_accs)
    print('epoch: %d, test dataset accuracy: %0.4f %%' % (epoch, test_acc * 100))
    if test_acc > best_acc:
        best_acc = test_acc
        if epoch % 2 == 0:
            torch.save(net.state_dict(), "{}/epoch_{}_accs{:0.2f}.pth".format(ckpt_dir, epoch,test_acc))

print('Finished Training')