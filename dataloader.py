import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

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

def getDataloaders_CL(data_set, dataset_name, train, shuffle, subclass=None, num_class=5, batch_size=64):
    class_length = -1
    if dataset_name =="C10":
        class_length = 5000 if train else 1000
    elif dataset_name == 'C100':
        class_length = 500 if train else 100
    elif dataset_name=="Tiny":
        class_length = 6000 if train else 1000
    data_loaders = []

    for s in subclass:
        dataset_prime = None
        count = 0
        for i in s:
            if not dataset_prime:
                dataset_prime = (data_set[i*class_length:(i+1)*class_length])
            else:
                dataset_prime = torch.utils.data.ConcatDataset((dataset_prime, (data_set[i*class_length:(i+1)*class_length])))
            count += 1

        data_loader = torch.utils.data.DataLoader(dataset_prime, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True,pin_memory=True)
        data_loaders.append(data_loader)
    return data_loaders

def dataset_to_loader(dataset, dset_name, num_tasks, isTrain, batch_size, shuffle, num_workers):
    class_length = -1
    if dset_name =="C10":
        class_length = 5000 if train else 1000
    elif dset_name == 'C100' or dset_name=='Tiny':
        class_length = 500 if train else 100
    else:
        raise NotImplementedError

    data_loaders = []
    task_length = num_tasks * class_length

    for i in range(num_tasks):
        loader = torch.utils.data.DataLoader(dataset[i*task_length:(i+1)*task_length] , batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True,pin_memory=True)
        data_loaders.append(loader)
    
    return data_loaders

def getDataloaders(dset_name, shuffle=True, splits=['train', 'test'],
                   data_root='./data', batch_size=10, 
                   num_workers=0, num_tasks=5, raw=False, **kwargs):

    train_loaders, test_loaders = None, None, None
    print('loading ' + dataset)
    if dset_name.find('C10') >= 0:
        if dataset.find('C100') >= 0:
            d_func = dset.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        else:
            d_func = dset.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        aug_trans = []
        common_trans = [transforms.ToTensor()]
        if not raw:
            common_trans.append(normalize)

        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        if 'train' in splits:
            train_set = d_func(data_root, train=True, transform=train_compose, download=True)
            train_set = sorted(train_set, key=lambda s:s[1])
            
            train_loaders = dataset_to_loader(train_set, dset_name=dset_name, num_tasks=num_tasks, isTrain=True, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if 'test' in splits:
            test_set = d_func(data_root, train=False, transform=test_compose)
            test_set = sorted(test_set, key=lambda s:s[1])

            test_loaders = dataset_to_loader(test_set, dset_name=dset_name, num_tasks=num_tasks, isTrain=True, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    elif dset_name.find('Tiny') >= 0:
        pass
    else:
        raise NotImplemented

    return train_loaders, test_loaders
