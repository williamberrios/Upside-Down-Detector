import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
class Config():
    seed           = 42
    epochs         = 10
    device         = 'cuda'
    lr             = 5e-5
    batch_size     = 256
    num_workers    = 32
    feature_name   = 'image'
    target_name    = 'label' 
    model_path     = 'SavedModels'
    fp16           = True
    accumulation_steps = 1
    max_grad_norm  = None
    # Logging:
    logging        = True
    project_name   = 'Fatima-Fellowship'
    runname        = 'resnet18'

    # Early stopping:
    early_stopping =  100000
    mode           = 'max'
    es_metric      = 'accuracy'

    # Loss:
    loss_params    = {'name':'BCE'}        
    # Optimizer:
    optimizer_params = {'name':'Adam',
                        'WD'  : 1e-5}

    # Scheduler:
    scheduler_params = {'name':None,
                        'step_on': None,# mode: [epoch,batch]      
                        'step_metric':None,
                        'patience':None} 
    # Model Params
    arch = 'resnet18'
    pretrained   = False
    # Augmentations:
    MEAN = [0.5176, 0.4169, 0.3637]
    STD  = [0.3010, 0.2723, 0.2672]

    transformations = {'train':  A.Compose(
                                            [   

                                                A.Normalize(MEAN,STD),
                                                ToTensorV2(transpose_mask=False,p=1.0)
                                            ]
                                        ),

                        'test': A.Compose(
                                            [
                                                A.Normalize(MEAN,STD),
                                                ToTensorV2(transpose_mask=False,p=1.0)
                                            ])}