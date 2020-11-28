import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy
import torchvision
from models import base_single
import utils
from utils import *
import tqdm
import gc

def arch_is_picked(n, policy):
    n_split = n.split('.')
    
    if len(n_split) >= 5:
        
        num_layer = int(n_split[1])
        action = int(n_split[2])
        
        if policy.shape[0] == 1:
            return policy[0][num_layer][action].item() > 0.
        else:
            tmp = torch.mean(policy, 0)
            return tmp[num_layer][action].item() > 0.
    else:
        return True


class MetaGraph(nn.Module):
    '''
    For EWC related code, we modified the code from https://github.com/GMvandeVen/continual-learning
    '''

    def __init__(self):
        super().__init__()
        # -SI:
        self.si_c = 0           #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # -EWC:
        self.gamma = 1.         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.emp_FI = False     #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

    #----------------- EWC-specifc functions -----------------#


    def estimate_fisher_update_by_policy(self, inputs, targets, policy, allowed_classes=None, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        #data_loader = torch.utils.data.DataLoader(dataset, batch_size=policies[0].shape[0], shuffle=False)

        mask = [0.] * self.num_classes
        if allowed_classes is not None:
            for c in allowed_classes:
                mask[c] = 1.0
        else:
            mask = [1.] * self.num_classes

        #for index,(x,y) in tqdm.tqdm(enumerate(data_loader)):
        if True:
            x = inputs
            y = targets
            # run forward pass of model
            if allowed_classes is None:
                output, _ = self.forward(x, policy) 
            else: 
                out, _ = self.forward(x, policy)
                #output =  out[:, allowed_classes]
                output = out * Variable(torch.tensor(mask)).cuda(non_blocking=True)

            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad: #and arch_is_picked(n, policies):
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

        del est_fisher_info
        gc.collect()


    def ewc_loss(self, policies):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                #print('---------')
                #print(policies)
                for n, p in self.named_parameters():
                    if p.requires_grad :#and arch_is_picked(n, policies):
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())

            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            gc.collect() 
            return (1./2)*sum(losses)#, losses_dict
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            #return torch.tensor(0., device=self._device())
            return Variable(torch.tensor(0.)).cuda()

    #------------------MetaGraph-----------------------
    def forward(self, x, policy, drop_path_prob=0):

        x = F.relu(self.norm1(self.conv1(x)))
        t = 0
        lat = Variable(torch.zeros(x.size(0)), requires_grad=False).cuda().float()

        for out_planes, num_blocks, stride in self.cfg:
            for idx in range(num_blocks):
                action = policy[:, t, :].contiguous()

                # early termination if all actions in the batch are zero
                if action[:, :].data.sum() == 0:
                    if idx != 0:
                        t += 1
                        continue
                    else:
                        feature_map = [self.layers[t].forward(x, 0)]
                        lat_in_this_block = [1. * (action[:, 0]+1)]
                        x = sum(feature_map)
                        lat += sum(lat_in_this_block).float()
                else:
                    prev_x = x
                    action_mask     = [action[:, i].contiguous().float().view(-1, 1, 1, 1) for i in range(action.size(1))]
                    feature_map_raw = [self.layers[t].forward(x, i) for i in range(action.size(1))]
                    feature_map     = [feature_map_raw[i] * action_mask[i] for i in range(action.size(1))]
                    lat_in_this_block = [1. * action[:, i].float() for i in range(action.size(1))]
                    x = sum(feature_map)
                    lat_delta = sum(lat_in_this_block).float()
                    is_no_action = (action.data==0).all(1)
                    if is_no_action.any():
                        if idx!=0:
                            # Exactly use previous feature map
                            x[is_no_action] = prev_x[is_no_action]
                            lat_delta[is_no_action] = 0
                        else:
                            # Feature map reshaped, use default layer
                            x[is_no_action] = feature_map_raw[2][is_no_action]
                            lat_delta[is_no_action] = 1.
                    lat += lat_delta

                t += 1


        x = F.relu(x)
        if self.dset_name == 'Tiny':
            x = F.avg_pool2d(x, 7)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x, lat 

    def store_back(self):
        for t in range(self.num_of_blocks):
            self.layers[t].store_back()

    def assign_mask(self):
        for t in range(self.num_of_blocks):
            self.layers[t].assign_mask()

    def count_real_params(self):
        num_param = 0
        for t in range(self.num_of_blocks):
            num_param += self.layers[t].count_parameters()

        print(' [*] Total num of real parameter: %.2f M' % (num_param/1e6))



class ResNet18(MetaGraph):
    # out_planes, num_blocks, stride
    cfg = [
           (64, 2, 1),
           (128, 2, 2),
           (256, 2, 2),
           (512, 2, 2)]
    def __init__(self, config=None, num_classes=10, detailed=True, norm_type='GroupNorm'):
        super(ResNet18, self).__init__()
        self.detailed = detailed
        self.dset_name = 'cifar'
        self.num_of_actions = 4
        self.num_of_blocks = sum([num_blocks for out_planes, num_blocks, stride in self.cfg])
        self.num_classes = num_classes
        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_type = norm_type
        if norm_type == 'GroupNorm':
            self.norm1 = nn.GroupNorm(2, 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)

        self.layers = self._make_layers(in_planes=64)
        self.linear = nn.Linear(512, num_classes)

        if self.detailed:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
            '''
            For easier implementation, we duplicate the weight from the superkernel to create different options during forwarding and backwarding.
            One may implemented it by reusing the same weight without creating dummy weights.
            '''
            print(' [*] Total num of parameters with dummy weights: %.2f M' % (pytorch_total_params/1e6)) 
            self.count_real_params()

 
    def _make_layers(self, in_planes):
        layers = []
        for out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                block = self._make_action(in_planes, out_planes, stride)
                layers.append(block)
                in_planes = out_planes

        self.num_of_layers = len(layers)
        if self.detailed:
            print(" [*] Total num of layers : ", self.num_of_layers)
        return nn.Sequential(*layers)

    def _make_action(self, inp, oup, stride):
        action = base_single.GroupBasicBlock(inp, oup, stride, 7, self.num_of_actions, self.num_of_blocks, groups=4, norm_type=self.norm_type)
        return action
        
class ResNet18_64(MetaGraph):
    # out_planes, num_blocks, stride
    cfg = [
           (64, 2, 1),
           (128, 2, 2),
           (256, 2, 2),
           (512, 2, 2)]
    def __init__(self, config=None, num_classes=200, detailed=True, norm_type='GroupNorm'):
        super(ResNet18_64, self).__init__()
        self.num_classes = num_classes
        self.dset_name = 'Tiny'
        self.num_of_actions = 4
        self.num_of_blocks = sum([num_blocks for out_planes, num_blocks, stride in self.cfg])
        self.detailed = detailed
        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if norm_type == 'GroupNorm':
            self.norm1 = nn.GroupNorm(2, 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)
        self.layers = self._make_layers(in_planes=64)
        self.linear = nn.Linear(512, num_classes)

        if self.detailed:
            '''
            For easier implementation, we duplicate the weight from the superkernel to create different options during forwarding and backwarding.
            One may implemented it by reusing the same weight without creating dummy weights.
            '''
            pytorch_total_params = sum(p.numel() for p in self.parameters())
            print(' [*] Total num of parameters with dummy weights: %.2f M' % (pytorch_total_params/1e6))
            self.count_real_params()

    def _make_layers(self, in_planes):
        layers = []
        for out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                block = self._make_action(in_planes, out_planes, stride)
                layers.append(block)
                in_planes = out_planes
        self.num_of_layers = len(layers)
        if self.detailed:
            print(" [*] Total num of layers: ", self.num_of_layers)
        return nn.Sequential(*layers)

    def _make_action(self, inp, oup, stride):
        action = base_single.GroupBasicBlock(inp, oup, stride, 7, self.num_of_actions, self.num_of_blocks, groups=4, norm_type=self.norm_type)

        return action
