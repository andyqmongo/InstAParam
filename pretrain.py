import argparse 
import os 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import torch.optim as optim
import numpy as np
import tqdm 
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.set_num_threads(1)

import warnings
warnings.filterwarnings("ignore")
import wandb

"""
Set seed
"""
np.random.seed = 9487
torch.manual_seed(9487)
torch.cuda.manual_seed(9487)
torch.cuda.manual_seed_all(9487)

parser = argparse.ArgumentParser(description='InstaNas Search Stage')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='InstAParam-single', choices=['InstAParam', 'InstAParam-single'])
parser.add_argument('--dset_name', default=None, required=True, choices=['C100', 'Tiny'])
parser.add_argument('--data_dir', default='./data/', help='data directory')
parser.add_argument('--cv_dir', default='./result', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--task', type=int, default=0, help="task to pre-train")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--net_optimizer', default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

args = parser.parse_args()


hyperparam = 'lr_{:.6f}_b_{}_e_{}_{}'.format(args.lr, args.batch_size, args.epochs, args.net_optimizer)
args.cv_dir = os.path.join(args.cv_dir, hyperparam)

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)

if not os.path.exists('./{}/models'.format(args.cv_dir)):
    import shutil
    shutil.copytree('./models', './{}/models'.format(args.cv_dir))

utils.save_args(__file__, args)

wandb.init(project='NIPS_InstAParam-single-{}-pretrain'.format(args.dset_name), config=args, name=hyperparam)

if args.dset_name == 'C10' or 'Fuzzy-C10':
    task_length = 2
    num_tasks = 5
elif args.dset_name == 'C100':
    task_length = 10
    num_tasks = 10
elif args.dset_name == 'Tiny':
    task_length = 20
    num_tasks = 10

assert args.task < num_tasks

def train(trainloader, testloader, task):

    
    for epoch in range(args.epochs):
        matches = []
        matches_te = []

        net_losses = []
        #Training 
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
            meta_graph.train()
            if args.model == 'InstAParam-single':
                meta_graph.assign_mask()
            inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)

            if epoch > args.epochs//2:
                drop_rate = 1-(0.5/(args.epochs//2))*(epoch-args.epochs//2)
                policy_shape = (inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)
                policy = Variable(torch.from_numpy(np.random.binomial(1, drop_rate, policy_shape))).long().cuda()
            else:
                policy = Variable(torch.ones(inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)).cuda()

            
            pm, _ = meta_graph.forward(inputs, policy)
            pm[:, (task+1)*task_length:] = 0.0

            net_loss = F.cross_entropy(pm, targets)

            optimizer_net.zero_grad()
            net_loss.backward()
            optimizer_net.step()
            net_losses.append(net_loss.item())

            #Acc
            _, pred_idx = pm.max(1)
            match = (pred_idx == targets).data.float()
            matches.append(match)
            if args.model == 'InstAParam-single':
                meta_graph.store_back()


        # Check train acc with assign_mask()
        if args.model == 'InstAParam-single':
            meta_graph.assign_mask()
        #Testing accuracy with assign_mask()
        for idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            meta_graph.eval()
            with torch.no_grad():
                inputs, targets = Variable(inputs).cuda(non_blocking=True), Variable(targets).cuda(non_blocking=True)

                policy = Variable(torch.ones(inputs.shape[0], meta_graph.num_of_blocks, meta_graph.num_of_actions)).cuda()

                preds, _ = meta_graph.forward(inputs, policy)
                preds[:, (task+1)*task_length:] = 0.0
                
                _, pred_idx = preds.max(1)
                match = (pred_idx == targets).data.float()
                matches_te.append(match)

        accuracy = torch.cat(matches, 0).mean()
        testing_acc = torch.cat(matches_te, 0).mean()


        if epoch == args.epochs // 2:
            net_state_dict = meta_graph.state_dict()
            torch.save(net_state_dict, os.path.join(args.cv_dir, 'ckpt_pretrain.t7'))
        elif epoch >= args.epochs-1:
            net_state_dict = meta_graph.state_dict()
            torch.save(net_state_dict, os.path.join(args.cv_dir, 'ckpt_pretrain_dropout.t7'))
        wandb.log({
            'train acc': accuracy,
            'test acc': testing_acc
        })


meta_graph, _ = utils.get_model(args.model, args.dset_name)
meta_graph.cuda()


if args.net_optimizer == 'sgd':
    optimizer_net = optim.SGD(meta_graph.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.net_optimizer == 'adam':
    optimizer_net = optim.Adam(meta_graph.parameters(), lr=args.lr, weight_decay=args.wd)

from dataloader import getDataloaders
trainLoaders, testLoaders = getDataloaders(dset_name=args.dset_name, shuffle=True, splits=['train', 'test'], 
        data_root=args.data_dir, batch_size=args.batch_size, num_workers=0, num_tasks=num_tasks, raw=False)

train(trainLoaders[args.task], testLoaders[args.task], args.task)
