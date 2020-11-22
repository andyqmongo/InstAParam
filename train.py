import argparse
import os
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli
import sys
import math
import pandas as pd
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_num_threads(1)
import random
import warnings
warnings.filterwarnings("ignore")

import wandb

parser = argparse.ArgumentParser(description='InstAParam')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--net_lr', type=float, default=None, help='learning rate for net, use `args.lr` if not set')
parser.add_argument('--beta', type=float, default=0.8, help='entropy multiplier')
parser.add_argument('--model', default='InstAParam-single', choices=['InstAParam', 'InstAParam-single'])
parser.add_argument('--dset_name', default='C10', required=False, choices=['C10', 'C100', 'Tiny', 'Fuzzy-C10'])
parser.add_argument('--data_dir', default='./data', help='data directory')
parser.add_argument('--load_graph', default=None, help='checkpoint to load meta-graph from')
parser.add_argument('--load_controller', default=None, help='checkpoint to load controller from')
parser.add_argument('--cv_dir', default='./result', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--iter_per_batch', type=int, default=10, help='iterations for each batch of data')
parser.add_argument('--alpha', type=float, default=0.80, help='probability bounding factor')
parser.add_argument('--pos_w', type=float, default=30.)
parser.add_argument('--neg_w', type=float, default=0.)
parser.add_argument('--lat_exp', type=float, default=1.)
#for continual learning
parser.add_argument('--pretrain_epochs', type=int, default=0)
parser.add_argument('--ewc_lambda', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--net_optimizer', default='sgd', choices=['adam', 'sgd', 'sgdm'])
parser.add_argument('--k', default=10., type=float, help='value to reshape sigmoid function')
parser.add_argument('--shift', default=0.5, type=float, help='the amount of sigmoid function shift')
parser.add_argument('--mu', type=float, default=0.5, help='thresholds to determine picked or not')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

args = parser.parse_args()


np.set_printoptions(suppress=True)
tanh = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()

if args.net_lr is None:
    args.net_lr = args.lr

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)

hyperparam = 'lr_{:.6f}_netlr{:.6f}_iter{}_{}_mu{}_gamma{:.4f}_ewc{}'.format(
    args.lr, args.net_lr, args.iter_per_batch, args.net_optimizer, args.mu, args.gamma, args.ewc_lambda)
save_dir = os.path.join(args.cv_dir, hyperparam)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

wandb.init(project='NIPS_{}_{}'.format(args.model, args.dset_name), config=args, name=hyperparam)
args.cv_dir = save_dir

if args.dset_name.find('C100') >= 0:
    task_length = 10
    num_tasks = 10
    num_classes = task_length * num_tasks
elif args.dset_name.find('C10') >= 0:
    task_length = 2
    num_tasks = 5
    num_classes = task_length * num_tasks
elif args.dset_name.find('Tiny') >= 0:
    task_length = 20
    num_tasks = 10
    num_classes = task_length * num_tasks
else:
    raise NotImplementedError

def get_reward(preds, targets, policy, elasped):

    sparse_reward = torch.log( policy.shape[1]*policy.shape[2] - elasped.cuda().data )
    sparse_reward = torch.clamp(sparse_reward, min=0.)

    _, pred_idx = preds.max(1)
    match = (pred_idx == targets).data

    reward = sparse_reward ** args.lat_exp

    reward[match]   *= args.pos_w
    reward[match==0] = args.neg_w

    reward = reward.unsqueeze(1)
    reward = reward.unsqueeze(2)

    return reward, match.float()


def train_online_and_test(trainLoaders, testLoaders):

    controller.train()
    meta_graph.eval()
    
    policies_real_sum = []
    
    iteration = 0
    for task in range(num_tasks):
        policies_ = []

        # pretrained if needed
        if args.dset_name == 'C10':
            #Do not need pretrain
            pass
        elif args.load_graph and task==0:
            #Load pretrained weights for the meta-graph
            matches = []
            policies = []
            print('task 0, load pretrained for meta-graph')
            if args.model == 'InstAParam-single':
                meta_graph.assign_mask()
            meta_graph.eval()
            policy_zero = Variable(torch.zeros(args.batch_size, meta_graph.num_of_blocks, meta_graph.num_of_actions)).cpu()
            for _, (inputs, targets) in tqdm.tqdm(enumerate(testLoaders[0]), total=len(testLoaders[0])):
                
                with torch.no_grad():
                    inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
                #--------------------------------------------------------------------------------------------#
                with torch.no_grad():
                    policy = Variable(torch.ones(inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)).cuda()
                    preds, _  = meta_graph.forward(inputs, policy.data.squeeze(0))
                #--------------------------------------------------------------------------------------------#
                
                curSubclass = range(0*task_length, 0*task_length+task_length)
                mask = [0.] * task_length * num_tasks
                for sub in curSubclass:
                    mask[sub] = 1.0
                preds = preds[:, curSubclass]

                _, pred_idx = preds.max(1)

                
                b = Variable(torch.Tensor([0*task_length]).long()).cuda()
                match = (pred_idx == (targets-b.expand(targets.size()))).data.float()

                matches.append(match)
                policies.append(policy_zero)

            accuracy = torch.cat(matches, 0).mean()
            print('Accuracy for loaded pretrain meta-graph :{:.4f}'.format(accuracy))
            
            policies_real_sum.append(torch.sum(torch.sum(torch.stack(policies), dim=0, keepdim=True)[0], dim=0, keepdim=True)[0])
            continue
        elif args.pretrain_epochs > 0 and task==0:
            ## Pretrain the meta-graph if needed
            for epoch in range(args.pretrain_epochs):
                for _, (inputs, targets) in tqdm.tqdm(enumerate(trainLoaders[0]), total=len(trainLoaders[0])):
                    inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)
                    meta_graph.train()
                    if args.model == 'InstAParam-single':
                        meta_graph.assign_mask()

                    if epoch >= args.pretrain_epochs//2:
                        policy_shape = (inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)
                        policy = Variable(torch.from_numpy(np.random.binomial(1, 0.5, policy_shape))).long().cuda()
                    else:
                        policy = Variable(torch.ones(args.batch_size, meta_graph.num_of_blocks, meta_graph.num_of_actions)).cuda()

                    pm, _ = meta_graph.forward(inputs, policy)
                    pm[:, (task+1)*task_length:] = 0.0

                    net_loss = F.cross_entropy(pm, targets)

                    optimizer_net.zero_grad()
                    net_loss.backward()
                    optimizer_net.step()
                    if args.model == 'InstAParam-single':
                        meta_graph.store_back()

            print('Pretrain done')
            policy_zero = Variable(torch.zeros(args.batch_size, meta_graph.num_of_blocks, meta_graph.num_of_actions)).cpu()
            policies_real_sum.append(torch.sum(torch.sum(torch.stack(policy_zero), dim=0, keepdim=True)[0], dim=0, keepdim=True)[0])
            continue

        if task > 1 or (task > 0 and args.dset_name=='C10'):
            #Calculate history matrix H to encourage exploration 
            history_policies = Variable(torch.sum(torch.stack(policies_real_sum[:task]), dim=0, keepdim=True)[0]).cuda(non_blocking=True)
            sqrt_hist = Variable(torch.sqrt(history_policies)).float().cuda()
            policies_reg = sigmoid( args.k*(sqrt_hist / sqrt_hist.max() - args.shift) ) * args.gamma
        else:
            policies_reg = 0
        
        matches, policies = [], []
        policies_ = []

        for _, (inputs, targets) in tqdm.tqdm(enumerate(trainLoaders[task]), total=len(trainLoaders[task])):
            iteration += 1
            for _ in range(args.iter_per_batch):
                if args.model == 'InstAParam-single':
                    meta_graph.assign_mask()

                inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)
                probs, _ = controller(inputs)
                #---------------------------------------------------------------------#

                policy_map = probs.data.clone()
                
                policy_map[policy_map < args.mu] = 0.0
                policy_map[policy_map >= args.mu] = 1.0


                policy_map = Variable(policy_map)

                probs = probs*args.alpha + (1-probs)*(1-args.alpha) - policies_reg
                probs = probs.clamp(0.0, 1.0)
                distr = Bernoulli(probs)
                policy = distr.sample()

                with torch.no_grad():
                    v_inputs = Variable(inputs.data)
                
                meta_graph.eval()
                preds_map, lat_map = meta_graph.forward(v_inputs, policy_map)

                preds_sample, lat = meta_graph.forward(v_inputs, policy)
                
                #mask for incremental task learning
                mask = [0.] * num_classes

                curSubclass = range(task*task_length, (task+1)*task_length)

                for sub in curSubclass:
                    mask[sub] = 1.0

                preds_map = preds_map * Variable(torch.tensor(mask)).cuda(non_blocking=True)
                
                preds_sample = preds_sample * Variable(torch.tensor(mask)).cuda(non_blocking=True)


                reward_map, _ = get_reward(preds_map, targets, policy_map.data, lat_map)
                reward_sample, match = get_reward(preds_sample, targets, policy.data, lat)
                
                
                advantage = reward_sample - reward_map

                loss = -distr.log_prob(policy)
                loss = loss * Variable(advantage).expand_as(policy)

                loss = loss.sum()

                probs = probs.clamp(1e-15, 1-1e-15)
                entropy_loss = -probs*torch.log(probs)
                entropy_loss = args.beta*entropy_loss.sum()

                loss = (loss - entropy_loss)/inputs.size(0)
                #------------------------Backprop Controller Loss---------------------#    
                #---------------------------------------------------------------------#
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #---------------------------------------------------------------------#

                #------------------------Backprop Meta-Graph Loss---------------------#    
                #---------------------------------------------------------------------#
                meta_graph.train()
                perm_policy = policy
                pm, _ = meta_graph.forward(v_inputs, perm_policy)


                mask = [0.] * num_classes

                curSubclass = range(task*task_length, (task+1)*task_length)
                for sub in curSubclass:
                    mask[sub] = 1.0
                pm = pm * Variable(torch.tensor(mask)).cuda(non_blocking=True)

                _, pred_idx = pm.max(1)

                match = (pred_idx == targets).data.float()

                
                with torch.no_grad():
                    pol = perm_policy
                if args.ewc_lambda > 0:
                    if task > 0 and args.dset_name == 'C10':
                        ewc_loss= meta_graph.ewc_loss(pol)
                    elif task > 1:
                        ewc_loss= meta_graph.ewc_loss(pol)
                    else:
                        ewc_loss = torch.zeros(1).cuda()
                else:
                    ewc_loss = torch.zeros(1).cuda()
                    
                entropy_loss = F.cross_entropy(pm, targets)
                net_loss = entropy_loss + ewc_loss * args.ewc_lambda

                
                optimizer_net.zero_grad()
                net_loss.backward()
                optimizer_net.step()
                #---------------------------------------------------------------------#
                if args.model == 'InstAParam-single':
                    meta_graph.store_back()

            if args.ewc_lambda > 0:
                allowed_classes = range(task*task_length, (task+1)*task_length)  
                meta_graph.estimate_fisher_update_by_policy(inputs, targets, policy=pol, allowed_classes=allowed_classes)

            policies.append(policy.data)
            matches.append(match)
            
            policies_.append(policy_map.data.cpu())

            # save the model
            controller_state_dict = controller.state_dict()
            net_state_dict = meta_graph.state_dict()


            if args.dset_name == 'Tiny':
                assert task != 0
            torch.save(controller_state_dict, os.path.join(args.cv_dir, 'controller_{}.t7'.format(task)))
            if args.model == 'InstAParam-single':
                utils.save_real_param(net_state_dict, os.path.join(args.cv_dir, 'meta_grpah_{}.t7'.format(task)))
            else:
                torch.save(net_state_dict, os.path.join(args.cv_dir, 'meta_grpah_{}.t7'.format(task)))
            #------ wandb --------
            if iteration%20 == 0:
                test_avg_acc, test_accs = test(testLoaders, cur_task=task)    
                if args.dset_name == 'C10' or args.dset_name=='Fuzzy-C10':
                    wandb.log({
                        'loss': np.mean(entropy_loss.detach().cpu().numpy()),
                        'ewc loss': np.mean(ewc_loss.detach().cpu().numpy()),
                        'online train acc': torch.cat(matches, 0).mean().cpu().numpy(),
                        #'cur train acc': train_acc,
                        'avg test acc':test_avg_acc,
                        'test acc 0':test_accs[0],
                        'test acc 1':test_accs[1],
                        'test acc 2':test_accs[2],
                        'test acc 3':test_accs[3],
                        'test acc 4':test_accs[4]
                    },step=iteration)
                else:
                    wandb.log({
                        'loss': np.mean(entropy_loss.detach().cpu().numpy()),
                        'ewc loss': np.mean(ewc_loss.detach().cpu().numpy()),
                        'online train acc': torch.cat(matches, 0).mean().cpu().numpy(),
                        #'cur train acc': train_acc,
                        'avg test acc':test_avg_acc,
                        'test acc 0':test_accs[0],
                        'test acc 1':test_accs[1],
                        'test acc 2':test_accs[2],
                        'test acc 3':test_accs[3],
                        'test acc 4':test_accs[4],
                        'test acc 5':test_accs[5],
                        'test acc 6':test_accs[6],
                        'test acc 7':test_accs[7],
                        'test acc 8':test_accs[8],
                        'test acc 9':test_accs[9],
                    },step=iteration)


            #-------



            if math.isnan(np.mean(entropy_loss.detach().cpu().numpy())) or math.isnan(np.mean(ewc_loss.detach().cpu().numpy())):
                print('loss is nan')
                sys.exit()

        policies_real_sum.append(torch.sum(torch.sum(torch.stack(policies_), dim=0, keepdim=True)[0], dim=0, keepdim=True)[0])

def test(testLoaders, repro_oneshot=False, test_task=-1, cur_task=-1):
    assert test_task < num_tasks
    accs = np.zeros(num_tasks)
    meta_graph_, controller_ = utils.get_model(args.model, args.dset_name, detailed=False)

    loop_through_task = list(range(num_tasks)) if test_task < 0 else [test_task]

    assert cur_task >= 0
    if os.path.exists('{}/meta_grpah_{}.t7'.format(args.cv_dir, cur_task)):
        ckpt_net = torch.load('{}/meta_grpah_{}.t7'.format(args.cv_dir, cur_task))
        meta_graph_.load_state_dict(ckpt_net, strict=False)
        meta_graph_.eval().cuda()
        if args.model == 'InstAParam-single':
            meta_graph_.assign_mask()
    else:
        raise NotImplementedError("Wrong Cur task for testing")
    for task in loop_through_task:
        matches, policies = [], []
        if os.path.exists('{}/controller_{}.t7'.format(args.cv_dir, task)):
            ckpt = torch.load('{}/controller_{}.t7'.format(args.cv_dir, task))
            controller_.load_state_dict(ckpt)
            controller_.eval().cuda()

            for _, (inputs, targets) in enumerate(testLoaders[task]):

                with torch.no_grad():
                    inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)

                    probs, _ = controller_(inputs)

                    policy = probs.data.clone()
                    
                    policy[policy < args.mu] = 0.0
                    policy[policy >= args.mu] = 1.0

                    if repro_oneshot:
                        policy = Variable(torch.ones(inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)).float().cuda()
                    else:
                        policy = Variable(policy)

                    preds, _ = meta_graph_.forward(inputs, policy)

                curSubclass = range(task*task_length, (task+1)*task_length)

                mask = [0.] * num_classes
                for sub in curSubclass:
                    mask[sub] = 1.0

                preds = preds * Variable(torch.tensor(mask)).cuda(non_blocking=True)

                _, pred_idx = preds.max(1)
                match = (pred_idx == targets).data.float()


                matches.append(match)

            accuracy = torch.cat(matches, 0).mean()
            accs[task] = accuracy
        
    
    if test_task >= 0:
        assert accs[test_task] == accuracy
        return accs[test_task]

    if args.dset_name=='C10':
        return np.mean(accs), accs
    else:
        return np.mean(accs[1:]), accs


from dataloader import getDataloaders
trainLoaders, testLoaders = getDataloaders(dset_name=args.dset_name, shuffle=True, splits=['train', 'test'], 
        data_root=args.data_dir, batch_size=args.batch_size, num_workers=0, num_tasks=num_tasks, raw=False)

meta_graph, controller = utils.get_model(args.model, args.dset_name)


if args.load_graph is not None:
    checkpoint = torch.load(args.load_graph)
    new_state = meta_graph.state_dict()
    new_state.update(checkpoint)
    meta_graph.load_state_dict(new_state, strict=False)

meta_graph.cuda()
controller.cuda()


if args.net_optimizer == 'sgd':
    optimizer_net = optim.SGD(meta_graph.parameters(), lr=args.net_lr, weight_decay=args.wd)
elif args.net_optimizer == 'adam':
    optimizer_net = optim.Adam(meta_graph.parameters(), lr=args.net_lr, weight_decay=args.wd)
elif args.net_optimizer == 'sgdm':
    optimizer_net = optim.SGD(meta_graph.parameters(), lr=args.net_lr, weight_decay=args.wd, momentum=0.9)
optimizer = optim.Adam(controller.parameters(), lr=args.lr, weight_decay=args.wd)


train_online_and_test(trainLoaders, testLoaders)
