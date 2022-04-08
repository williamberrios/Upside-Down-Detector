import torch
import os
import cv2
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,path,split = 'train',transforms = None):
        self.img_files  = os.listdir(os.path.join(path,split))
        self.transforms = transforms
        self.path       = path
        self.split      = split
    
    def __getitem__(self,index):
        img_path = os.path.join(self.path,self.split,self.img_files[index])
        img      = cv2.imread(img_path)
        img      = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label    = int(os.path.join( self.img_files[index]).split('_')[1][0])
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img, #torch.tensor(img, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
    def __len__(self):
        return len(self.img_files)
if __name__ == '__main__':
    DATA_PATH  = '../dataset/celebrity-256/custom'
    dataset = FaceDataset(DATA_PATH,'train',transforms = None)
    print(dataset.__getitem__(0)['image'].mean())