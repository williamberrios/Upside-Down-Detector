import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
from PIL import Image
import glob
from joblib import Parallel, delayed
import shutil
# Seed random
## Create Dataset
# 0: Down
# 1: Top
SEED = 0
WORKERS = 16
SOURCE_PATH = "dataset/celebrity-256/original/celeba_hq_256/"  
OUTPUT_PATH = "dataset/celebrity-256/custom" 

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def rotate_save(img_path,orient,split):
    if orient == 'down':
        # Open Image
        img = Image.open(img_path)
        img = img.rotate(180)
        
        # Save Image
        img.save(os.path.join(OUTPUT_PATH,split,img_path.split('/')[-1].split('.jpg')[0]+'_0' + '.jpg'))
    else:
        shutil.copy2(img_path,os.path.join(OUTPUT_PATH,split,img_path.split('/')[-1].split('.jpg')[0]+'_1' + '.jpg') ) 
        
if __name__ == '__main__':
    imgs = glob.glob(os.path.join(SOURCE_PATH,'*.jpg'))
    imgs_train, imgs_test, _, _ = train_test_split(imgs, imgs, test_size=0.2, random_state=SEED)
    print(f"Porcentaje {np.round(100*len(imgs_train)/(len(imgs_train) + len(imgs_test)),2)}%")
    imgs_train_up, imgs_train_down, _, _ = train_test_split(imgs_train, 
                                                        imgs_train, 
                                                        test_size=0.5, 
                                                        random_state=SEED)
    imgs_test_up, imgs_test_down, _, _ = train_test_split(imgs_test, 
                                                          imgs_test, 
                                                          test_size=0.5,
                                                          random_state=SEED)
    
    # Preprocessing Image train
    # Up
    print("**** TRAIN UP *****")
    Parallel(n_jobs = WORKERS,backend = "multiprocessing")(
        delayed(rotate_save)(img_path,orient,split) for img_path,orient,split in (zip(imgs_train_up,['up']*len(imgs_train_up),['train']*len(imgs_train_up)))
    )
    # Down
    print("**** TRAIN DOWN *****")
    Parallel(n_jobs = WORKERS,backend = "multiprocessing")(
        delayed(rotate_save)(img_path,orient,split) for img_path,orient,split in (zip(imgs_train_down,['down']*len(imgs_train_down),['train']*len(imgs_train_down)))
    )
    # Preprocessing Image test
    # Up
    print("**** TEST UP *****")
    Parallel(n_jobs = WORKERS,backend = "multiprocessing")(
        delayed(rotate_save)(img_path,orient,split) for img_path,orient,split in (zip(imgs_test_up,['up']*len(imgs_test_up),['test']*len(imgs_test_up)))
    )
    # Down
    # Preprocessing Image test
    print("**** TEST DOWN *****")
    Parallel(n_jobs = WORKERS,backend = "multiprocessing")(
        delayed(rotate_save)(img_path,orient,split) for img_path,orient,split in (zip(imgs_test_down,['down']*len(imgs_test_down),['test']*len(imgs_test_down)))
    )