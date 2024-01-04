import torch
from typing import Any
from abc import abstractmethod
from numpy import inf
from pathlib import Path

from src.utils.wandb_logger import WandBLogger
from src.utils.loggingmanager import LoggingManager
from src.utils.datastructures import TrainerConfig


class BaseTrainer:
    """
    Base class for all trainers
    """
    # use propety as those are otherwise notused in base trainer.. and then i dont have to pass it over..
    @property
    def model(self) -> Any:#BaseModel:
        raise NotImplementedError

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def __init__(self, model, optimizer, base_trainer_config: TrainerConfig, 
                wandb_logger: WandBLogger, 
                training_save_dir: Path,
                gpu_id: int, 
                resume_path: Path=None):
    

        self.logger = LoggingManager.get_logger_by_name('train')


        # sofar only ever 1 is used
        self.device = self._prepare_device(gpu_id)
    
        self.epochs = base_trainer_config.epochs
        self.save_period = base_trainer_config.save_period
        self.resume_path = resume_path


        # configuration to monitor model performance and save best
        if base_trainer_config.monitor_config is None:
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            assert base_trainer_config.monitor_config.criterion in ['min', 'max']
            assert base_trainer_config.monitor_config.data_type in ['val', 'train']
            
            self.mnt_mode = base_trainer_config.monitor_config.criterion
            self.mnt_metric = base_trainer_config.monitor_config.data_type + "/" + base_trainer_config.monitor_config.metric_name

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = base_trainer_config.early_stop

        self.start_epoch = 1

        self.checkpoint_dir = training_save_dir
        self.wandb_logger = wandb_logger

        if self.resume_path is not None:
            self._resume_checkpoint(self.resume_path)

        

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    @abstractmethod
    def _val_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            #torch.cuda.empty_cache()
            log, log_median, log_plots = self._train_epoch(epoch)
            #with torch.no_grad():
            val_log, val_log_median, val_log_plots = self._val_epoch(epoch)

            # self.logger.info(f"Model size after epoch {epoch}: {profile.get_model_size(self.model)} bytes.")
            # get stuff we want to log
            result_log = {'epoch': epoch}
            for k,v in log.items():
                result_log.update({'train/' + k: v, 'train/median-' + k: log_median[k]})

            for k,v in val_log.items():
                result_log.update({'val/' + k: v, 'val/median-' + k: val_log_median[k]})

            # print logged informations to the screen
            for key, value in result_log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
  
            # add plots to be logged
            for plot in log_plots:
                result_log.update({"train/" + plot.name: plot.figure})
            for plot in val_log_plots:    
                result_log.update({"val/" + plot.name: plot.figure})
            
            self.wandb_logger.log(result_log)

            

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and result_log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and result_log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = result_log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)
        
        


        return self.mnt_best

    def dispose(self):
        self.wandb_logger.finish()

    def _prepare_device(self, gpu_id):
        """
        setup GPU device if available, move model into configured device
        For now training only on one is possible..
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            device = torch.device('cpu')
        else: 
            if gpu_id > n_gpu - 1:
                self.logger.warning("Warning: The GPU id given does not match any available GPU, set to default id = 0.")
                gpu_id = 0
            device = torch.device('cuda:' + str(gpu_id))

        return device


    def _save_checkpoint(self, epoch, save_best=False) -> None:
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        #only for DGI pretraining - save only encoder as model
        
        arch = type(self.model).__name__
        if arch == 'DeepGraphInfomax':
            state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
            #'config': self.config
        }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best
                #'config': self.config
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
            
            

    def _resume_checkpoint(self, resume_path: str) -> None:
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])


        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
