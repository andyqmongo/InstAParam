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
import pandas as pd
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
parser.add_argument('--load_dir', default=None)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--mu', default=0.5, type=float, help='threshold for picking the block')
parser.add_argument('--norm_type', type=str, default='GroupNorm', choices=['GroupNorm', 'BatchNorm'])
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



def inference(testloader, test_task, agent, meta_graph):

    matches_te = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
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
        #preds = preds[:, curSubclass]
        preds = preds * Variable(torch.tensor(mask)).cuda(non_blocking=True)

        _, pred_idx = preds.max(1)

        match = (pred_idx == targets).data.float()

        matches_te.append(match)

    testing_acc = torch.cat(matches_te, 0).mean()

    return testing_acc.cpu().detach().numpy()



def test_one_by_one(testLoaders, model, dset_name, norm_type, load_dir, detailed=False):
    writer = SummaryWriter(os.path.join(load_dir, 'log'))

    meta_graph, controller = utils.get_model(model, dset_name, norm_type=norm_type, detailed=detailed)
    meta_graph.cuda()
    controller.cuda()

    accs = np.zeros((num_tasks, num_tasks))
    start_task = 0
    if dset_name =='C100' or dset_name =='Tiny':
        start_task = 1
    if detailed:
        print("START TASK:{}".format(start_task))

    for t in range(start_task, num_tasks):#meta-graph
        if not os.path.exists('{}/meta_grpah_{}.t7'.format(load_dir, t)):
            return None
        try:
            ckpt_net = torch.load('{}/meta_grpah_{}.t7'.format(load_dir, t))
        except:
            return None
        meta_graph.load_state_dict(ckpt_net, strict=False)
        meta_graph.eval().cuda()
        if model == 'InstAParam-single':
            meta_graph.assign_mask()

        for tt in range(start_task, t+1):#controller
            if not os.path.exists('{}/controller_{}.t7'.format(load_dir, tt)):
                return None
            try:
                ckpt = torch.load('{}/controller_{}.t7'.format(load_dir, tt))
            except:
                return None
            controller.load_state_dict(ckpt)
            controller.eval().cuda()
            
            acc = inference(testLoaders[tt], tt, controller, meta_graph)
            #print("t:{}, tt:{}, acc:{}".format(t, tt, acc))
            accs[t][tt] = acc
            
    if dset_name == 'C10':
        avg_acc = np.mean(accs[-1])
    else:
        avg_acc = np.mean(accs[-1][1:])
    dir_ = os.path.join(load_dir, 'test_{:.4f}.csv'.format(avg_acc))
    
    df = pd.DataFrame(accs)
    #if saved:
    df.to_csv(dir_)
    
    if detailed:
        print(accs)
        print('avg acc:{:.4f}'.format(avg_acc))

    return avg_acc

if __name__ == '__main__':
    from dataloader import getDataloaders
    _, testLoaders = getDataloaders(dset_name=args.dset_name, shuffle=True, splits=['test'], 
            data_root=args.data_dir, batch_size=args.batch_size, num_workers=0, num_tasks=num_tasks, raw=False)
    test_one_by_one(testLoaders, args.model, args.dset_name, args.norm_type, args.load_dir, detailed=True)
