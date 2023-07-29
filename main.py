from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import glob
import os,sys
import matplotlib.pylab as plt
import joblib
from tqdm import tqdm
# import mlflow
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from dataclasses import asdict
from pytorch_lightning.callbacks import EarlyStopping
# import wandb

from model import * 
from dataio import *
from utils import *
from eval import *


@hydra.main(config_name='config')
def main(param):
    SEED = 2023
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)   # :bug it arises an unexpected bug of pytorch 
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Logging
    logger = WandbLogger(name=param.experiment_name, project="Singing technique classification")
    logger.log_hyperparams(param)
    print(hydra.utils.get_original_cwd())

    dir = hydra.utils.get_original_cwd() + "/mlruns"
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('load train_dataset...')
    audio_path = os.path.join(param.data_dir, "audio")

    train_dataset = AudioDataset(audio_path,param.data_dir,"train",3)
    valid_dataset = AudioDataset(audio_path,param.data_dir,"valid",3)
    test_dataset = AudioDataset(audio_path,param.data_dir,"test",3)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True,
                                            drop_last=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=param.batch_size, shuffle=True,
                                            drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                            drop_last=False, num_workers=4)
    
    pl.seed_everything(param.random_seed)

    # train
    print("start training...")
    class_weights = train_dataset.get_class_weights(alpha=param.alpha)
    class_weights = [float(x) for x in class_weights.values()] 
    class_weights = torch.from_numpy(np.array(class_weights)).float()
    model =PlModel(param=param,classes_num=10,class_weights=class_weights,retrain=False)
    model.train()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=param.epoch, default_root_dir='data/models/',precision=32, check_val_every_n_epoch=5, logger=logger, callbacks=[early_stop_callback])
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader)
    
    model.eval()
    feature_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
        drop_last=True, num_workers=4)
    macrof1, accuracy, balanced, top_2, top_3, df_cmx, report = evaluation_wandb(logger,
    plot_title="singing technique classification",
    test_loader=test_loader,
    model=model,
      random_state=2023,
      retrain=param.retrain,
      epoch=10, 
      retrain_loader=train_loader, 
      target_class=train_dataset.class2id, 
      target_class_inv=train_dataset.id2class)
    # try:   
    #     mlflow.log_artifact(plot_title + "_result.txt")
    # except:
    #     pass
    logger.log_metrics({"acc":accuracy, "bacc":balanced, "top2":top_2, "top3":top_3, "f1":macrof1})
    
if __name__ == "__main__":
    main()