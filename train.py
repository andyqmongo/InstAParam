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

parser = argparse.ArgumentParser(description='InstaNas Search Stage')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--net_lr', type=float, default=None, help='learning rate for net, use `args.lr` if not set')
parser.add_argument('--beta', type=float, default=0.8, help='entropy multiplier')
parser.add_argument('--model', default='InstAParam-single', choices=['InstAParam', 'InstAParam-single'])
parser.add_argument('--dset_name', default=None, required=True, choices=['C10', 'C100', 'Tiny', 'Fuzzy-C10'])
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
parser.add_argument('--wd', type=float, default=1.0, help='weight decay')

args = parser.parse_args()

np.set_printoptions(suppress=True)
tanh = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()

if args.net_lr is None:
    args.net_lr = args.lr

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)

if args.dset_name == 'C10':
    task_length = 2
    num_tasks = 5
elif args.dset_name == 'C100':
    task_length = 10
    num_tasks = 10
elif args.dset_name == 'Tiny':
    task_length = 20
    num_tasks = 10

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

    agent.train()
    instanet.eval()
    
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
            instanet.assign_mask()
            instanet.eval()
            policy_zero = Variable(torch.zeros(args.batch_size, instanet.num_of_blocks, instanet.num_of_actions)).cpu()
            for _, (inputs, targets) in tqdm.tqdm(enumerate(testLoaders[0]), total=len(testLoaders[0])):
                
                with torch.no_grad():
                    inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
                #--------------------------------------------------------------------------------------------#
                with torch.no_grad():
                    policy = Variable(torch.ones(inputs.shape[0], instanet.num_of_blocks, instanet.num_of_actions)).cuda()
                    preds, _  = instanet.forward(inputs, policy.data.squeeze(0))
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
                    instanet.train()
                    instanet.assign_mask()

                    if epoch >= args.pretrain_epochs//2:
                        policy_shape = (inputs.shape[0], instanet.num_of_blocks, instanet.num_of_actions)
                        policy = Variable(torch.from_numpy(np.random.binomial(1, 0.5, policy_shape))).long().cuda()
                    else:
                        policy = Variable(torch.ones(args.batch_size, instanet.num_of_blocks, instanet.num_of_actions)).cuda()

                    pm, _ = instanet.forward(inputs, policy)
                    pm[:, (task+1)*task_length:] = 0.0

                    net_loss = F.cross_entropy(pm, targets)

                    optimizer_net.zero_grad()
                    net_loss.backward()
                    optimizer_net.step()
                    instanet.store_back()

            print('Pretrain done')
            policy_zero = Variable(torch.zeros(args.batch_size, instanet.num_of_blocks, instanet.num_of_actions)).cpu()
            policies_real_sum.append(torch.sum(torch.sum(torch.stack(policy_zero), dim=0, keepdim=True)[0], dim=0, keepdim=True)[0])
            continue

        if task > 1 or (task > 0 and args.dset_name=='C10'):
            #Calculate history matrix H to encourage exploration 
            history_policies = Variable(torch.sum(torch.stack(policies_real_sum[:task]), dim=0, keepdim=True)[0]).cuda(non_blocking=True)
            #print(history_policies)
            #df = pd.DataFrame(history_policies.cpu().numpy())
            #df.to_excel('{}/history_policies_{}.xlsx'.format(args.cv_dir, task))
            sqrt_hist = Variable(torch.sqrt(history_policies)).float().cuda()
            
            policies_reg = sigmoid( args.k*(sqrt_hist / sqrt_hist.max() - args.shift) ) * args.gamma
            #print('----------policy reg---------------')
            #print(policies_reg)
            #df = pd.DataFrame(policies_reg.cpu().numpy())
            #df.to_excel('{}/policies_reg_task{}.xlsx'.format(args.cv_dir, task))
        else:
            policies_reg = 0
        
        matches, policies = [], []
        policies_ = []

        for _, (inputs, targets) in tqdm.tqdm(enumerate(trainLoaders[task]), total=len(trainLoaders[task])):
            iteration += 1
            for iter_ in range(args.iter_per_batch):
                instanet.assign_mask()

                inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)
                probs, _ = agent(inputs)
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
                
                instanet.eval()
                preds_map, lat_map = instanet.forward(v_inputs, policy_map)

                preds_sample, lat = instanet.forward(v_inputs, policy)
                
                #mask for incremental task learning
                mask = [0.] * (num_tasks * task_length)


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
                instanet.train()
                perm_policy = policy
                pm, _ = instanet.forward(v_inputs, perm_policy)


                mask = [0.] * (num_tasks * task_length)

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
                        ewc_loss= instanet.ewc_loss(pol)
                    elif task > 1:
                        ewc_loss= instanet.ewc_loss(pol)
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
                instanet.store_back()

            if args.ewc_lambda > 0:
                allowed_classes = range(task*task_length, (task+1)*task_length)  
                instanet.estimate_fisher_update_by_policy(inputs, targets, policy=pol, allowed_classes=allowed_classes)

            policies.append(policy.data)
            matches.append(match)
            
            policies_.append(policy_map.data.cpu())

            # save the model
            agent_state_dict = agent.state_dict()
            net_state_dict = instanet.state_dict()


            if args.dset_name == 'Tiny':
                assert task != 0
            torch.save(agent_state_dict, os.path.join(args.cv_dir, 'controller_{}.t7'.format(task)))
            if args.model == 'InstAParam-single':
                utils.save_real_param(net_state_dict, os.path.join(args.cv_dir, 'meta_grpah_{}.t7'.format(task)))
            else:
                torch.save(net_state_dict, os.path.join(args.cv_dir, 'meta_grpah_{}.t7'.format(task)))


            if math.isnan(np.mean(entropy_loss.detach().cpu().numpy())) or math.isnan(np.mean(ewc_loss.detach().cpu().numpy())):
                print('loss is nan')
                sys.exit()

        #----

        policies_real_sum.append(torch.sum(torch.sum(torch.stack(policies_), dim=0, keepdim=True)[0], dim=0, keepdim=True)[0])
        #print('-- policies_ mean ---')
        #cur_prm = policies_real_sum[task].cpu().numpy()
        #print(cur_prm)

        #df = pd.DataFrame(cur_prm)
        #df.to_excel('{}/policies_real_sum_task{}.xlsx'.format(args.cv_dir, task))


def test(testLoaders, repro_oneshot=False, test_task=-1, cur_task=-1):
    assert test_task < num_tasks
    accs = np.zeros(num_tasks)
    instanet_, agent_ = utils.get_model(args.model, detailed=False)

    loop_through_task = list(range(num_tasks)) if test_task < 0 else [test_task]

    assert cur_task >= 0
    if os.path.exists('{}/meta_grpah_{}.t7'.format(args.cv_dir, cur_task)):
        ckpt_net = torch.load('{}/meta_grpah_{}.t7'.format(args.cv_dir, cur_task))
        instanet_.load_state_dict(ckpt_net, strict=False)
        instanet_.eval().cuda()
        instanet_.assign_mask()
    else:
        raise NotImplementedError("Wrong Cur task for testing")
    for task in loop_through_task:
        matches, policies = [], []
        if os.path.exists('{}/controller_{}.t7'.format(args.cv_dir, task)):
            ckpt = torch.load('{}/controller_{}.t7'.format(args.cv_dir, task))
            agent_.load_state_dict(ckpt)
            agent_.eval().cuda()

            for _, (inputs, targets) in enumerate(testLoaders[task]):

                with torch.no_grad():
                    inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)

                    probs, _ = agent_(inputs)

                    policy = probs.data.clone()
                    
                    policy[policy < args.mu] = 0.0
                    policy[policy >= args.mu] = 1.0

                    if repro_oneshot:
                        policy = Variable(torch.ones(inputs.shape[0], instanet.num_of_blocks, instanet.num_of_actions)).float().cuda()
                    else:
                        policy = Variable(policy)

                    preds, _ = instanet_.forward(inputs, policy)

                curSubclass = range(task*task_length, (task+1)*task_length)

                mask = [0.] * (num_tasks * task_length)
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

instanet, agent = utils.get_model(args.model, args.dset_name)


if args.load_graph is not None:
    checkpoint = torch.load(args.load_graph)
    new_state = instanet.state_dict()
    new_state.update(checkpoint)
    instanet.load_state_dict(new_state, strict=False)

instanet.cuda()
agent.cuda()


if args.net_optimizer == 'sgd':
    optimizer_net = optim.SGD(instanet.parameters(), lr=args.net_lr, weight_decay=args.wd)
elif args.net_optimizer == 'adam':
    optimizer_net = optim.Adam(instanet.parameters(), lr=args.net_lr, weight_decay=args.wd)
elif args.net_optimizer == 'sgdm':
    optimizer_net = optim.SGD(instanet.parameters(), lr=args.net_lr, weight_decay=args.wd, momentum=0.9)
optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.wd)


train_online_and_test(trainLoaders, testLoaders)
