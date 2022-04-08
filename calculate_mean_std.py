#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from src.dataset import FaceDataset
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


# In[ ]:


def calc_dataset_mean_std(train_tensor_dataset, batch_size=32, num_workers=0):
   
    loader = DataLoader(train_tensor_dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)

    mean = 0.0
    for batch in tqdm(loader):
        images = batch['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    image_info = train_tensor_dataset[0]
    sample_img = image_info['image']
    # for std calculations
    h, w = sample_img.shape[1], sample_img.shape[2]

    for batch in tqdm(loader):
        images = batch['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset)*w*h))

    return mean, std


# In[ ]:


if __name__ == '__main__':
    path = 'dataset/celebrity-256/custom'
    split = 'train'
    dataset = FaceDataset(path,split,transforms = ToTensorV2())
    mean,std = calc_dataset_mean_std(dataset, batch_size=32, num_workers=0)
    print(mean/255,std/255)


# In[ ]:




