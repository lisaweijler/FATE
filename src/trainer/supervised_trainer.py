from typing import List
import torch
from tqdm import tqdm
from random import shuffle
from collections import namedtuple

from src.trainer.base_trainer import BaseTrainer
from src.utils.utils import MetricTracker
from src.utils.vis import mrd_plot


Log_plot = namedtuple('Log_plot', ['name', 'figure'])

class FATETrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                 loss_ftn,
                 metric_ftns,
                 optimizer, 
                 config, 
                 marker_dict,
                 wandb_logger, 
                 training_save_dir, 
                 gpu_id, 
                 data_loader, 
                 eval_data_loader=None, 
                 lr_scheduler=None, 
                 resume_path=None):

        
        self.config = config
        
        # used in baseclass so written as property to not have to pass it over
        self._model = model
        self._optimizer = optimizer
        super().__init__(model, 
                         optimizer, 
                         config, 
                         wandb_logger, 
                         training_save_dir, 
                         gpu_id, 
                         resume_path=resume_path)

        self.model.to(self.device)

        self.loss_ftn = loss_ftn
        self.metric_ftns = metric_ftns

        # Data
        self.data_loader = data_loader
        self.valid_data_loader = eval_data_loader

        self.len_epoch = len(self.data_loader)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.marker_dict = marker_dict

        # Metrics for training and validation
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.valid_freq =5
        


    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch. Starts by 1 not 0
        :return: A log that contains average loss and metric in this epoch.
        """
 
        self.model.train()
        self.train_metrics.reset()

        log_plots = []

        for batch_idx, batch in tqdm(enumerate(self.data_loader), desc='train', total=len(self.data_loader)):
           
            self.optimizer.zero_grad()
            batch_size = len(batch["data"])

            loss = 0


            
            for spl_idx in range(batch_size):
                sample_loss = 0
                current_sample = batch["data"][spl_idx].float().to(self.device)
                current_y = batch["target"][spl_idx].float().to(self.device).unsqueeze(-1)
                current_markerlist = batch["marker"][spl_idx]

                # Forward + backward pass
                m_list = [m for m in current_markerlist if m in self.marker_dict]
                n_events = current_sample.shape[0]
                shuffle(m_list)
                markers_idx = torch.tensor([current_markerlist.index(m) for m in m_list], device=self.device)
                _, markers_pos, _ = self._get_marker_ordering_and_position(n_events, m_list)
            
                prediction, latents = self.model(current_sample[:,markers_idx], markers_pos) # apply model 

                sup_sample_loss = self.loss_ftn(prediction, current_y)
                loss += sup_sample_loss
          


            # we are finished with looping over batch
            loss /= batch_size
   

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) 
            self.optimizer.step()
            

            self.train_metrics.update('loss', loss.item(), n=batch_size)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(prediction, current_y), n=batch_size)


            if batch_idx == self.len_epoch: 
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

  
        log, median_log = self.train_metrics.result()
        
        return log, median_log, log_plots
    
    def _get_marker_ordering_and_position(self, n_events, markerlist, p_removed: float = None):
        '''
        markerlist: list of markers that are in markerdict and in sample
                    from those it is chosen what to mask
        p_remove: percentage of markers to mask
        '''

        n_marker = len(markerlist)
        marker_mask = None
        markers_pos_masked = None
        # create matrix with position of markers in marker dict n_events x n_marker
        markers_pos = torch.tensor([self.marker_dict[m] for m in markerlist], device=self.device)
        markers_pos = markers_pos.repeat((n_events, 1))
        if p_removed is not None:
            # how many marker to mask based on percentage
            n_remove = int(len(markerlist)*p_removed)

            # create masking matrix n_events x n_marker, true are markers to be kept
            values = torch.rand((n_events, n_marker), device=self.device)
            _, indices = torch.topk(values, n_remove, largest=False, sorted=True)
            marker_mask = torch.ones((n_events, n_marker), dtype=torch.bool, device=self.device)
            marker_mask.scatter_(1, indices, False)
    
            markers_pos_masked = markers_pos[marker_mask].reshape((n_events, n_marker-n_remove)) # n_events x n_marker_not_masked

        return marker_mask, markers_pos, markers_pos_masked
    


    def _val_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        self.valid_metrics.reset()

        filenames = []
        log_plots = []

        if self.do_validation:
            with torch.no_grad():
                if epoch%self.valid_freq==0 or epoch == self.epochs or epoch == 1:
                    for batch_idx, batch in tqdm(enumerate(self.valid_data_loader), desc= 'eval', total=len(self.valid_data_loader)):
                        # Load data
                        batch["data"][0] = batch["data"][0].float()
                        x = batch["data"][0].to(self.device)
                        target = batch["target"][0].float().to(self.device).unsqueeze(-1)
      
                        filenames.append(batch["name"][0]) # because it is a batch so we have a list of names                
                    
                        # Forward + backward pass
                        m_list = [m for m in batch["marker"][0] if m in self.marker_dict]
                        markers_idx = torch.tensor([batch["marker"][0].index(m) for m in m_list], device=self.device)
                        _, markers_pos, _ = self._get_marker_ordering_and_position(x.shape[0], m_list)
                    
                        prediction, latents = self.model(x[:,markers_idx], markers_pos) # apply model 

                        loss = self.loss_ftn(prediction, target)        

                        self.valid_metrics.update('loss', loss.item())
                        for met in self.metric_ftns:
                            self.valid_metrics.update(met.__name__, met(prediction, target))
                            

                    # MRD figure
                    log_plots.append(self._log_mrd_plot(self.valid_metrics, filenames))
        log, median_log = self.valid_metrics.result()
        return log, median_log, log_plots
    
    def _log_mrd_plot(self, metrics : MetricTracker, filenames : List[str]) -> Log_plot:

        # MRD figure
        metric_data = metrics.data()
        mrd_fig = mrd_plot(mrd_list_gt=metric_data['mrd_gt'], mrd_list_pred=metric_data['mrd_pred'], f1_score=metric_data['f1_score'], filenames=filenames)

        return Log_plot("MRD", mrd_fig)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)