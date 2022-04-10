 
# Upside down detector


## Setup

1.  Install Python (>=3.7), PyTorch and other required python libraries with:
    ```
    pip install -r requirements.txt
    ```
2.  Download the CelebA-HQ-resized dataset from [Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) inside the datasets/celebrity-256/original folder

3. Create the Train-Test splits
    ```
    python preprocess_images.py 
    ```
4. Calculata the mean and std of the training data
   ```
   python calculate_mean_std.py
   ```

## Usage

+ Generate or choose a config file from "Configs" folder and run the Train notebook

## Hugging Face Space
+ You can interact with the model [here](https://huggingface.co/spaces/will33am/fatima-Upside-Down-Detector)
## Logging

You could see the logging and plotting of your metrics in [wandb]( https://wandb.ai/williamberrios/Fatima-Fellowship?workspace=user-williamberrios
)
