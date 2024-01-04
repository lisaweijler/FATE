import torch
import wandb
import os
from typing import Dict
from .datastructures import WandBConfig

class WandBLogger:

    def __init__(self, config: WandBConfig, model: torch.nn.modules=None, run_config: Dict = None) -> None:
        

        if config is None:
            self.enabled = False
        else:
            self.enabled = config.enabled


        if self.enabled:
            wandb.init(entity=config.team, 
                        project=config.project_name, 
                        group=config.group,
                        notes=config.notes,
                        tags=config.tags,
                        config=run_config.dict())

            wandb.run.name = wandb.run.id      

            if model is not None:
                self.watch(model)         
            
    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
                
    def save_bestmodel(self, state):
        if self.enabled:
            torch.save(state, os.path.join(wandb.run.dir,"model_best.pth"))
 

    def finish(self):
        if self.enabled:
            wandb.finish()
