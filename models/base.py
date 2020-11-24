import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.utils as torchutils
from torch.nn import init, Parameter


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inp, oup, stride)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = conv3x3(oup, oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)
        self.lat = 0
        self.flops = 0
        self.params = 0

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class GroupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride, kernel, num_action, num_block, groups=1):
        super(GroupBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = nn.Conv2d(oup, oup, kernel_size=kernel, stride=1, padding=kernel//2, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(oup)
        self.bn2_sbn = SwitchableBatchNorm2d(num_action, oup)
        self.params = 0
        

    def forward(self, x, use_sbn=False, policy=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if use_sbn:
            assert policy != None
            out = self.bn2_sbn(out, policy)
        else:
            out = self.bn2(out)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride, kernel, num_action, num_block):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = nn.Conv2d(oup, oup, kernel_size=kernel, stride=1, padding=kernel//2, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.bn2_sbn = SwitchableBatchNorm2d(num_action, oup)
        self.params = 0

    def forward(self, x, use_sbn=False, policy=None):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if use_sbn:
            assert policy != None
            out = self.bn2_sbn(out, policy)
        else:
            out = self.bn2(out)
        out = F.relu(out)

        return out




class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_of_actions, out_size):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_of_actions = num_of_actions
        bns = []
        for i in range(num_of_actions):
            bns.append(nn.BatchNorm2d(out_size))
        self.bn = nn.ModuleList(bns)

    def forward(self, input, action):
        action_mask = [action[:, i].contiguous().float().view(-1, 1, 1, 1) for i in range(action.size(1))]
        feature_map_raw = [self.bn[i](input) for i in range(action.size(1))]
        feature_map = [feature_map_raw[i] * action_mask[i] for i in range(action.size(1))]
        y = sum(feature_map)
        return y

'''
class CBN(nn.Module):

    def __init__(self, lstm_size, emb_size, out_size, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.lstm_size = lstm_size # size of the lstm emb which is input to MLP (num_of_actions x num_blocks) say 85
        self.emb_size = emb_size # size of hidden layer of MLP say 512
        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def create_cbn_input(self, lstm_emb):
        lstm_emb = lstm_emb.float()
        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

        return delta_betas, delta_gammas

    def create_init_beta_gamma(self, batch_size, channels):
        self.betas = nn.Parameter(torch.zeros(batch_size, channels).cuda())
        self.gammas = nn.Parameter(torch.ones(batch_size, channels).cuda())

    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape
        
        # get initial beta and gamma
        self.create_init_beta_gamma(self.batch_size, self.channels)

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = betas_cloned.view(self.batch_size, self.channels, 1, 1)
        # betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        # betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = gammas_cloned.view(self.batch_size, self.channels, 1, 1)
        # gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        # gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out
'''