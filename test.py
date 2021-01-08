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

from torch.utils.tensorboard import SummaryWriter

"""
Set seed
"""
np.random.seed = 9487
torch.manual_seed(9487)
torch.cuda.manual_seed(9487)
torch.cuda.manual_seed_all(9487)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='InstAParam-single', choices=['InstAParam', 'InstAParam-single'])
parser.add_argument('--dset_name', default=None, required=True, choices=['C10', 'C100', 'Tiny', 'Fuzzy-C10'])
parser.add_argument('--data_dir', default='./data/')
parser.add_argument('--load_dir', required=True, default=None)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--mu', default=0.5, type=float, help='threshold for picking the block')
parser.add_argument('--norm_type', type=str, default='GroupNorm', choices=['GroupNorm', 'BatchNorm'])

args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.load_dir, 'log'))

if args.dset_name == 'C10' or args.dset_name == 'Fuzzy-C10':
    task_length = 2
    num_tasks = 5
elif args.dset_name == 'C100':
    task_length = 10
    num_tasks = 10
elif args.dset_name == 'Tiny':
    task_length = 20
    num_tasks = 10



def test(testloader, test_task, agent, meta_graph):

    matches_te = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        
        with torch.no_grad():
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()

        #--------------------------------------------------------------------------------------------#
        with torch.no_grad():
            probs, _ = agent(inputs)
            policy = probs.clone()

            policy[policy < args.mu] = 0.0
            policy[policy >= args.mu] = 1.0

            preds, _  = meta_graph.forward(inputs, policy.data.squeeze(0))
        #--------------------------------------------------------------------------------------------#

        curSubclass = range(test_task*task_length, test_task*task_length+task_length)
        mask = [0.] * (task_length * num_tasks)
        for sub in curSubclass:
            mask[sub] = 1.0
        preds = preds[:, curSubclass]

        _, pred_idx = preds.max(1)

        match = (pred_idx == targets).data.float()

        matches_te.append(match)

    testing_acc = torch.cat(matches_te, 0).mean()

    return testing_acc.cpu().detach().numpy()


from dataloader import getDataloaders
_, testLoaders = getDataloaders(dset_name=args.dset_name, shuffle=True, splits=['test'], 
        data_root=args.data_dir, batch_size=args.batch_size, num_workers=0, num_tasks=num_tasks, raw=False)

meta_graph, controller = utils.get_model(args.model, args.dset_name, norm_type=args.norm_type)
meta_graph.cuda()
controller.cuda()

accs = np.zeros((num_tasks, num_tasks))
start_task = 0
if args.dset_name =='C100' or args.dset_name =='Tiny':
    start_task = 1

for t in range(start_task, num_tasks):
    ckpt_net = torch.load('{}/meta_grpah_{}.t7'.format(args.load_dir, t))
    meta_graph.load_state_dict(ckpt_net, strict=False)
    meta_graph.eval().cuda()
    if args.model == 'InstAParam-single':
        meta_graph.assign_mask()

    for tt in range(t):
        if os.path.exists('{}/controller_{}.t7'.format(args.load_dir, tt)):
            ckpt = torch.load('{}/controller_{}.t7'.format(args.load_dir, tt))
            controller.load_state_dict(ckpt)
            controller.eval().cuda()

        accs[t][tt] = test(testLoaders[tt], tt, controller, meta_graph)
        

print(accs)