import torch
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
from matplotlib import rcParams


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = torch.device("cpu")
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = torch.device("cuda")


images = []
lesions = []
check_images = []
root = 'segmentation'
check_root = 'test'

for root, dirs, files in os.walk(os.path.join(root)):
    for file in files:
        if file.endswith('s.bmp'):
            lesions.append(imread(os.path.join(root, files[0]), as_gray = True))
        else:
            images.append(imread(os.path.join(root, files[0]), as_gray = False))



temp = 0
for check_root, dirs, files in os.walk(os.path.join(check_root)):
    for file in files:
        check_images.append(imread(os.path.join(check_root, files[temp]), as_gray = True))
        temp += 1


size = (512, 512)
X = [resize(x, size, mode='constant', anti_aliasing=True) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.05 for y in lesions]
T = [resize(t, size, mode='constant', anti_aliasing=True) for t in check_images]


X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
T = np.array(T, np.float32)
#print(f'Loaded {len(X)} images')
#print(f'Loaded {len(Y)} images')
#print(f'Loaded {len(T)} images')


#plt.figure(figsize=(18, 6))
#for i in range(6):
#    plt.subplot(2, 6, i+1)
#    plt.axis("off")
#    plt.imshow(X[i])

#    plt.subplot(2, 6, i+7)
#    plt.axis("off")
#    plt.imshow(Y[i])
#plt.show()


ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [12, 20])


#plt.figure(figsize=(18, 6))
#for i in range(6):
#    plt.subplot(2, 6, i+1)
#    plt.axis("off")
#    plt.imshow(T[i])
#plt.show()


batch_size = 2
data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])), batch_size=batch_size, shuffle=True)

data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])), batch_size=batch_size, shuffle=True)

data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])), batch_size=batch_size, shuffle=True)

train_dl = DataLoader(T, shuffle = True, batch_size = batch_size, num_workers = 2, drop_last = True)


rcParams['figure.figsize'] = (15,4)


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.enc_conv0 = nn.Sequential(
                          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU()
                        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 256 -> 128

        self.enc_conv1 = nn.Sequential(
                          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU()
                        )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 128 -> 64
        
        self.enc_conv2 = nn.Sequential(
                          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU()
                        )
        self.pool2 =  nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 64 -> 32

        self.enc_conv3 = nn.Sequential(
                          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU()
                        )
        self.pool3 =  nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
                                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0, stride=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU()
                              )

        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(2, 2) # 16 -> 32
        self.dec_conv0 = nn.Sequential(
                          nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU()
                        )
        
        self.upsample1 = nn.MaxUnpool2d(2, 2) # 32 -> 64
        self.dec_conv1 = nn.Sequential(
                          nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU()
                        )
        
        self.upsample2 = nn.MaxUnpool2d(2, 2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
                          nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU()
                        )
        self.upsample3 = nn.MaxUnpool2d(2, 2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
                          nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(1),
                          nn.ReLU(),

                          nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1),
                          nn.BatchNorm2d(1),
                        )

    def forward(self, x):
        # encoder
        e0, ind0 = self.pool0(self.enc_conv0(x))
        e1, ind1 = self.pool1(self.enc_conv1(e0))
        e2, ind2 = self.pool2(self.enc_conv2(e1))
        e3, ind3 = self.pool3(self.enc_conv3(e2))

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.dec_conv0(self.upsample0(b,  ind3))
        d1 = self.dec_conv1(self.upsample1(d0, ind2))
        d2 = self.dec_conv2(self.upsample2(d1, ind1))
        d3 = self.dec_conv3(self.upsample3(d2, ind0))
        
        return d3


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1).byte()   # Forming outputs to the correct form
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded 


def bce_loss(y_real, y_pred):
    loss = y_pred - y_real*y_pred + (1+ torch.exp(-1*y_pred)).log()
    return loss.mean()


def train(model, optimizer, scheduler, loss_fn, score_fn, epochs, data_tr, data_vl, device):

    torch.cuda.empty_cache()

    losses_train = []
    losses_val = []
    scores_train = []
    scores_val = []

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)            

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            Y_pred = model(X_batch)
            #print("Y_pred len: ", Y_pred.size())
            #print("Y_batch len: ", Y_batch.size())
            loss = loss_fn(Y_pred, Y_batch) # forward-pass

            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate loss to show the user
            avg_loss += loss / len(data_tr)

        toc = time()
        print('train_loss: %f' % avg_loss)
        losses_train.append(avg_loss)

        # train score
        avg_score_train = score_fn(model, iou_pytorch, data_tr)
        scores_train.append(avg_score_train)

        # val loss
        avg_loss_val = 0
        model.eval()  # testing mode
        for X_val, Y_val in data_vl:
            with torch.no_grad():
                Y_hat = model(X_val.to(device)).detach().cpu()# detach and put into cpu

                loss = loss_fn(Y_hat, Y_val) # forward-pass
                avg_loss_val += loss / len(data_vl)

        toc = time()
        print('val_loss: %f' % avg_loss_val)
        losses_val.append(avg_loss_val)

        # val score
        avg_score_val = score_fn(model, iou_pytorch, data_vl)
        scores_val.append(avg_score_val)

        if scheduler:
            #scheduler.step(avg_score_val)
            scheduler.step()

        torch.cuda.empty_cache()
          
    return (losses_train, losses_val, scores_train, scores_val)


def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [ X_batch.to(DEVICE) for X_batch, _ in data]
    return np.array(Y_pred)


def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        with torch.no_grad():
            X_batch.to(DEVICE)
            Y_label.to(DEVICE)
            Y_pred = model(X_batch)
            
            #We need to make outputs in the range from 0 to 1, as masks are
            #If output is bigger than treshhold level => it is 1, else - zero            
            #Treshold is 0.1
            
            # 'torch.ones_like' returns the tensor with the size like the input matrix size
            Y_pred = torch.ones_like(Y_pred) * (Y_pred > 0.1)
            scores += metric(Y_pred, Y_label.to(DEVICE)).mean().item()

    return scores/len(data)


model = SegNet().to(DEVICE)


max_epochs = 20

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00100, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.7)
results = train(model, optimizer, scheduler, bce_loss, score_model, max_epochs, data_tr, data_val, DEVICE)


def dice_loss(y_real, y_pred):
    
    smooth = 1e-8
    outputs = y_pred.sigmoid().squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = y_real.squeeze(1)    

    num = (outputs * labels).sum()
    den = (outputs + labels).sum()
    res = 1 - ((2. * num + smooth) / (den + smooth))#/(256*256)
    
    return res 


model_dice = SegNet().to(DEVICE)

max_epochs = 60
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00100, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.7)
results_dice = train(model_dice, optimizer, scheduler, dice_loss, score_model, max_epochs, data_tr, data_val, DEVICE)


def focal_loss(y_real, y_pred, eps = 1e-8, gamma = 2):
    y = y_pred.sigmoid()+eps
    loss = -((1-y)**gamma*y_real*y.log()+(1-y_real)*(1-y).log())
    return loss.mean()


model_focal = SegNet().to(DEVICE)

max_epochs = 60
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00100, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.7)
results_focal = train(model_focal, optimizer, scheduler, focal_loss, score_model, max_epochs, data_tr, data_val, DEVICE)


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding):
        super(conv2DBatchNormRelu, self).__init__()

        self.unit = nn.Sequential(
            nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.unit(inputs)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            conv2DBatchNormRelu(3, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, 64, 3, 1, 1)            
        )
        self.pool0 = nn.MaxPool2d(2, 2, return_indices=True)  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            conv2DBatchNormRelu(64, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),       
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True) # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            conv2DBatchNormRelu(128, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1)            
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True) # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            conv2DBatchNormRelu(256, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1)            
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True) # 32 -> 16

        self.bottle_neck = nn.Sequential(
            conv2DBatchNormRelu(512, 1024, 1, 1, 0),
            conv2DBatchNormRelu(1024, 512, 1, 1, 0)   
        )

        self.upsample3 = nn.MaxUnpool2d(2, 2) # 16 -> 32
        self.dec_conv3 = nn.Sequential(
            conv2DBatchNormRelu(512*2, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
        )

        self.upsample2 = nn.MaxUnpool2d(2, 2) # 32 -> 64
        self.dec_conv2 = nn.Sequential(
            conv2DBatchNormRelu(256*2, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
        )

        self.upsample1 = nn.MaxUnpool2d(2, 2) # 64 -> 128
        self.dec_conv1 = nn.Sequential(
            conv2DBatchNormRelu(128*2, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, 64, 3, 1, 1),
        )

        self.upsample0 = nn.MaxUnpool2d(2, 2) # 128 -> 256
        self.dec_conv0 = nn.Sequential(
            conv2DBatchNormRelu(64*2, 1, 3, 1, 1),
            conv2DBatchNormRelu(1, 1, 3, 1, 1),

            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # encoder
        pre_e0 = self.enc_conv0(x)
        e0, ind0 = self.pool0(pre_e0)
        pre_e1 = self.enc_conv1(e0)
        e1, ind1 = self.pool1(pre_e1)
        pre_e2 = self.enc_conv2(e1)
        e2, ind2 = self.pool2(pre_e2)
        pre_e3 = self.enc_conv3(e2)
        e3, ind3 = self.pool3(pre_e3)        

        # bottleneck        
        bottle_neck = self.bottle_neck(e3)

        # decoder
        d3 = self.dec_conv3(torch.cat([self.upsample3(bottle_neck, ind3), pre_e3], 1))
        d2 = self.dec_conv2(torch.cat([self.upsample2(d3, ind2), pre_e2], 1))
        d1 = self.dec_conv1(torch.cat([self.upsample1(d2, ind1), pre_e1], 1))
        d0 = self.dec_conv0(torch.cat([self.upsample0(d1, ind0), pre_e0], 1))

        # no activation
        return d0
    

unet_model = UNet().to(DEVICE)


max_epochs = 80

optimizer = torch.optim.AdamW(unet_model.parameters(), lr=0.00100, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 40], gamma=0.7)
results_unet = train(unet_model, optimizer, scheduler, focal_loss, score_model, max_epochs, data_tr, data_val, DEVICE)


def plot_items(model, loader, cnt):

    model.eval()
    X, Y = next(iter(loader))
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    Y_pred = model(X)

    p = Y_pred.detach().cpu()
    p_post = torch.ones_like(p) * (p > 0.1)
    y = Y.detach().cpu()
    
    plt.figure(figsize=(15, 10))
    for i in range(cnt):
        plt.subplot(3, cnt, i+1+cnt*0)
        plt.imshow(np.rollaxis(p[i,0].numpy(), 0), cmap='gray')
        plt.title('Output')
        plt.axis('off')
        
        plt.subplot(3, cnt, i+1+cnt*1)
        plt.imshow(np.rollaxis(p_post[i,0].numpy(), 0), cmap='gray')
        plt.title('Post-processing')
        plt.axis('off')

        plt.subplot(3, cnt, i+1+cnt*2)
        plt.imshow(np.rollaxis(y[i,0].numpy(), 0), cmap='gray')
        plt.title('Real')
        plt.axis('off')


plot_items(unet_model, train_dl, 10)
