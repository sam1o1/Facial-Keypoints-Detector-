## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # f-w/s +1
        # the output Tensor for one image, will have the dimensions: (32, 64, 64)
        # The adopted arech from this paper 
        # https://arxiv.org/pdf/1710.00977.pdf

        # The Convolutional  layers 
        self.conv_1 = nn.Conv2d(1, 32, 4) 
        self.conv_2 = nn.Conv2d(32, 64, 3) 
        self.conv_3 = nn.Conv2d(64, 128, 2) 
        self.conv_4 = nn.Conv2d(128, 256, 1)

        # Maxpooling
        self.pool=nn.MaxPool2d(2,2)

        # Dropout layers 
        self.dropout_1=nn.Dropout(p=0.1)
        self.dropout_2=nn.Dropout(p=0.2)
        self.dropout_3=nn.Dropout(p=0.3)
        self.dropout_4=nn.Dropout(p=0.4)
        self.dropout_5=nn.Dropout(p=0.5)
        self.dropout_6=nn.Dropout(p=0.6)


        		# Fully Connected Layers (fc)
        self.fc_1 = nn.Linear(in_features = 43264, out_features = 1000)  #torch.Size([10, 36864]) => in_features:  = 36864
        self.fc_2 = nn.Linear(in_features = 1000, out_features =  1000)  
        self.fc_3 = nn.Linear(in_features =  1000, out_features =   136)    #68 keypoints with x and y coordinate => out_features: 136
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        x = self.pool(F.relu(self.conv_1(x)))
        x=self.dropout_1(x)
        x = self.pool(F.relu(self.conv_2(x)))
        x=self.dropout_2(x)
        x = self.pool(F.relu(self.conv_3(x)))
        x=self.dropout_3(x)
        x = self.pool(F.relu(self.conv_4(x)))
        x=self.dropout_4(x)

        x = x.view(x.size(0), -1)

        x = self.dropout_5(F.relu(self.fc_1(x)))
        x = self.dropout_6(F.relu(self.fc_2(x)))
        x = self.fc_3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
