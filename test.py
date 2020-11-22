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

args = parser.parse_args()

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
        mask = [0.] * task_length * num_tasks
        for sub in curSubclass:
            mask[sub] = 1.0
        preds = preds[:, curSubclass]

        _, pred_idx = preds.max(1)

        b = Variable(torch.Tensor([test_task*task_length]).long()).cuda()
        match = (pred_idx == (targets-b.expand(targets.size()))).data.float()

        matches_te.append(match)

    testing_acc = torch.cat(matches_te, 0).mean()

    return testing_acc


from dataloader import getDataloaders
_, testLoaders = getDataloaders(dset_name=args.dset_name, shuffle=True, splits=['test'], 
        data_root=args.data_dir, batch_size=args.batch_size, num_workers=0, num_tasks=num_tasks, raw=False)

meta_graph, agent = utils.get_model(args.model, args.dset_name)
meta_graph.cuda()
agent.cuda()

accs = []
for t in range(num_tasks):
    for tt in range(t):
        acc = test(testLoaders[tt], tt, agent, meta_graph)
        accs.append(acc)

print(accs)