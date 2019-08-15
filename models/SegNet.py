import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):

    def __init__(self, NUM_CLASSES):
        super(SegNet, self).__init__()
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.predictions = nn.Linear(in_features=4096, out_features=1000, bias=True)
        self.max_unpool_indices = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.block5_deconv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_deconv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_deconv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_deconv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_deconv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_deconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_deconv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_deconv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_deconv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_deconv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_deconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block1_deconv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block1_deconv1 = nn.Conv2d(in_channels=64, out_channels=NUM_CLASSES, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)  # !!!! GENERAL TRAINING

    def forward(self, x):

        # encoder = VGG16
        block1_conv1_pad = F.pad(x, (1, 1, 1, 1))  # same padding
        block1_conv1 = self.block1_conv1(block1_conv1_pad)
        block1_conv1_activation = F.relu(block1_conv1)
        block1_conv2_pad = F.pad(block1_conv1_activation, (1, 1, 1, 1))
        block1_conv2 = self.block1_conv2(block1_conv2_pad)
        block1_conv2_activation = F.relu(block1_conv2)
        block1_pool, indices_1 = F.max_pool2d(block1_conv2_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        block2_conv1_pad = F.pad(block1_pool, (1, 1, 1, 1))
        block2_conv1 = self.block2_conv1(block2_conv1_pad)
        block2_conv1_activation = F.relu(block2_conv1)
        block2_conv2_pad = F.pad(block2_conv1_activation, (1, 1, 1, 1))
        block2_conv2 = self.block2_conv2(block2_conv2_pad)
        block2_conv2_activation = F.relu(block2_conv2)
        block2_pool, indices_2 = F.max_pool2d(block2_conv2_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        block3_conv1_pad = F.pad(block2_pool, (1, 1, 1, 1))
        block3_conv1 = self.block3_conv1(block3_conv1_pad)
        block3_conv1_activation = F.relu(block3_conv1)
        block3_conv2_pad = F.pad(block3_conv1_activation, (1, 1, 1, 1))
        block3_conv2 = self.block3_conv2(block3_conv2_pad)
        block3_conv2_activation = F.relu(block3_conv2)
        block3_conv3_pad = F.pad(block3_conv2_activation, (1, 1, 1, 1))
        block3_conv3 = self.block3_conv3(block3_conv3_pad)
        block3_conv3_activation = F.relu(block3_conv3)
        block3_pool, indices_3 = F.max_pool2d(block3_conv3_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        block4_conv1_pad = F.pad(block3_pool, (1, 1, 1, 1))
        block4_conv1 = self.block4_conv1(block4_conv1_pad)
        block4_conv1_activation = F.relu(block4_conv1)
        block4_conv2_pad = F.pad(block4_conv1_activation, (1, 1, 1, 1))
        block4_conv2 = self.block4_conv2(block4_conv2_pad)
        block4_conv2_activation = F.relu(block4_conv2)
        block4_conv3_pad = F.pad(block4_conv2_activation, (1, 1, 1, 1))
        block4_conv3 = self.block4_conv3(block4_conv3_pad)
        block4_conv3_activation = F.relu(block4_conv3)
        block4_pool, indices_4 = F.max_pool2d(block4_conv3_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        block5_conv1_pad = F.pad(block4_pool, (1, 1, 1, 1))
        block5_conv1 = self.block5_conv1(block5_conv1_pad)
        block5_conv1_activation = F.relu(block5_conv1)
        block5_conv2_pad = F.pad(block5_conv1_activation, (1, 1, 1, 1))
        block5_conv2 = self.block5_conv2(block5_conv2_pad)
        block5_conv2_activation = F.relu(block5_conv2)
        block5_conv3_pad = F.pad(block5_conv2_activation, (1, 1, 1, 1))
        block5_conv3 = self.block5_conv3(block5_conv3_pad)
        block5_conv3_activation = F.relu(block5_conv3)
        block5_pool, indices_5 = F.max_pool2d(block5_conv3_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)

        # decoder = mirror VGG16

        block5_depool5 = self.max_unpool_indices(block5_pool, indices_5, block5_conv3_activation.size())
        block5_deconv3_pad = F.pad(block5_depool5, (1, 1, 1, 1))
        block5_deconv3 = self.block5_deconv3(block5_deconv3_pad)
        block5_deconv3_activation = F.relu(block5_deconv3)

        block5_deconv2_pad = F.pad(block5_deconv3_activation, (1, 1, 1, 1))
        block5_deconv2 = self.block5_deconv2(block5_deconv2_pad)
        block5_deconv2_activation = F.relu(block5_deconv2)
        block5_deconv1_pad = F.pad(block5_deconv2_activation, (1, 1, 1, 1))
        block5_deconv1 = self.block5_deconv1(block5_deconv1_pad)
        block5_deconv1_activation = F.relu(block5_deconv1)

        block4_depool4 = self.max_unpool_indices(block5_deconv1_activation, indices_4, block4_conv3_activation.size())
        block4_deconv3_pad = F.pad(block4_depool4, (1, 1, 1, 1))
        block4_deconv3 = self.block4_deconv3(block4_deconv3_pad)
        block4_deconv3_activation = F.relu(block4_deconv3)
        block4_deconv2_pad = F.pad(block4_deconv3_activation, (1, 1, 1, 1))
        block4_deconv2 = self.block4_deconv2(block4_deconv2_pad)
        block4_deconv2_activation = F.relu(block4_deconv2)
        block4_deconv1_pad = F.pad(block4_deconv2_activation, (1, 1, 1, 1))
        block4_deconv1 = self.block4_deconv1(block4_deconv1_pad)
        block4_deconv1_activation = F.relu(block4_deconv1)

        block3_depool3 = self.max_unpool_indices(block4_deconv1_activation, indices_3, block3_conv3_activation.size())
        block3_deconv3_pad = F.pad(block3_depool3, (1, 1, 1, 1))
        block3_deconv3 = self.block3_deconv3(block3_deconv3_pad)
        block3_deconv3_activation = F.relu(block3_deconv3)
        block3_deconv2_pad = F.pad(block3_deconv3_activation, (1, 1, 1, 1))
        block3_deconv2 = self.block3_deconv2(block3_deconv2_pad)
        block3_deconv2_activation = F.relu(block3_deconv2)
        block3_deconv1_pad = F.pad(block3_deconv2_activation, (1, 1, 1, 1))
        block3_deconv1 = self.block3_deconv1(block3_deconv1_pad)
        block3_deconv1_activation = F.relu(block3_deconv1)

        block2_depool2 = self.max_unpool_indices(block3_deconv1_activation, indices_2, block2_conv2_activation.size())
        block2_deconv2_pad = F.pad(block2_depool2, (1, 1, 1, 1))
        block2_deconv2 = self.block2_deconv2(block2_deconv2_pad)
        block2_deconv2_activation = F.relu(block2_deconv2)
        block2_deconv1_pad = F.pad(block2_deconv2_activation, (1, 1, 1, 1))
        block2_deconv1 = self.block2_deconv1(block2_deconv1_pad)
        block2_deconv1_activation = F.relu(block2_deconv1)

        block1_depool1 = self.max_unpool_indices(block2_deconv1_activation, indices_1, block1_conv2_activation.size())
        block1_deconv2_pad = F.pad(block1_depool1, (1, 1, 1, 1))
        block1_deconv2 = self.block1_deconv2(block1_deconv2_pad)
        block1_deconv2_activation = F.relu(block1_deconv2)
        block1_deconv1_pad = F.pad(block1_deconv2_activation, (1, 1, 1, 1))
        block1_deconv1 = self.block1_deconv1(block1_deconv1_pad)

        softmax_output = F.softmax(block1_deconv1, dim=1)

        return softmax_output



