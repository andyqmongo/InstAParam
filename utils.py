import shutil
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
import math

def category_to_onehot(data, num_of_actions, num_of_blocks):
    output = []
    num_of_samples = data.shape[0]
    for i in range(num_of_samples):
        tmp = []
        sample = data[i]
        for block in sample:
            if block == num_of_actions: #Drop All
                arch = np.zeros((num_of_actions))
            else:
                arch = np.zeros((num_of_actions))
                arch[block] = 1
            tmp.append(arch)
        tmp = np.asarray(tmp).reshape((num_of_blocks,num_of_actions))
        output.append(tmp)
    output = np.asarray(output).reshape((num_of_samples, num_of_blocks,num_of_actions))
    return output

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt', 'w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(2).sum(1).sum(0)/policies.size(0)

    policy_set = [np.reshape(p.cpu().numpy().astype(
        np.float).astype(np.str), (-1)) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, sparsity, policy_set


def adjust_learning_rate_cos(optimizer, epoch, max_epochs, lr, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = max_epochs * nBatch
        T_cur = (epoch % max_epochs) * nBatch + batch
        new_lr = 0.5 * lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        new_lr, decay_rate = lr, 0.1
        if epoch >= max_epochs * 0.75:
            new_lr *= decay_rate**2
        elif epoch >= max_epochs * 0.5:
            new_lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch % self.epoch_step == 0:
                print(' [*] setting learning_rate to %.2E' % lr)



def cutout(mask_size, p, cutout_inside=False, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout




def get_model(model, dset_name, detailed=True):

    from models import controller, base, instanas, instanas_single

    if model == 'InstAParam':
        if dset_name == 'C10' or dset_name == 'Fuzzy-C10':
            instanet = instanas.ResNet18(num_classes=10, detailed=detailed)
            agent = controller.Policy32([1, 1, 1], num_blocks=8, num_of_actions=5)
        elif dset_name == 'C100':
            instanet = instanas.ResNet18(num_classes=100,detailed=detailed)
            agent = controller.Policy32([1, 1, 1], num_blocks=8, num_of_actions=5)

        elif dset_name == 'Tiny':
            instanet = instanas.ResNet18_64(num_classes=200, detailed=detailed)
            agent = controller.Policy224([1,1,1,1], num_blocks=8, num_of_actions=5)
        else:
            raise NotImplementedError(' [*] Unkown model.')
    elif model == 'InstAParam-single':
        if dset_name == 'C10' or dset_name == 'Fuzzy-C10':
            instanet = instanas_single.ResNet18(num_classes=10, detailed=detailed)
            agent = controller.Policy32([1, 1, 1], num_blocks=8, num_of_actions=4)
        elif dset_name == 'C100':
            instanet = instanas_single.ResNet18(num_classes=100,detailed=detailed)
            agent = controller.Policy32([1, 1, 1], num_blocks=8, num_of_actions=4)

        elif dset_name == 'Tiny':
            instanet = instanas_single.ResNet18_64(num_classes=200, detailed=detailed)
            agent = controller.Policy224([1,1,1,1], num_blocks=8, num_of_actions=4)
    else:
        raise NotImplementedError(' [*] Unkown model.')
    
    return instanet, agent
