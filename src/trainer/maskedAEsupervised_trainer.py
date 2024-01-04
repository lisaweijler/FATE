from typing import List
import torch
from tqdm import tqdm
from random import shuffle
from collections import namedtuple

from src.trainer.base_trainer import BaseTrainer
from src.utils.utils import MetricTracker
from src.utils.vis import mrd_plot

Log_plot = namedtuple('Log_plot', ['name', 'figure'])

class MaskedAESupervisedTrainer(BaseTrainer):
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
                 mask_ratio=0.5,
                 eval_data_loader=None, 
                 lr_scheduler=None, 
                 resume_path=None):

        
        self.config = config
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
        self.mask_ratio = mask_ratio

        # Metrics for training and validation
        self.train_metrics = MetricTracker('loss', 'supervised_loss','ae_loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', 'supervised_loss', 'ae_loss', *[m.__name__ for m in self.metric_ftns])

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
            supervised_loss = 0
            ae_loss = 0
            # could and should be optimized:
            for spl_idx in range(batch_size):
                sample_loss = 0
                current_sample = batch["data"][spl_idx].float().to(self.device)
                current_y = batch["target"][spl_idx].float().to(self.device).unsqueeze(-1)
                current_markerlist = batch["marker"][spl_idx]

                # Forward + backward pass
                m_list = [m for m in current_markerlist if m in self.marker_dict] # list of list containing marker present in samples in batch
                n_events = current_sample.shape[0]
                shuffle(m_list)
                
                n_marker = len(m_list)
                n_removed = int(n_marker*self.mask_ratio)
                markers_idx = [current_markerlist.index(m) for m in m_list]
                markers_mask, markers_pos, markers_pos_masked = self._get_marker_ordering_and_position(n_events, m_list, p_removed=self.mask_ratio)

                # prepare input sample
                #   1. reorder marker according to m_list
                input_sample = current_sample[:,markers_idx]
                #   2. remove masked marker
                input_sample_masked = input_sample[markers_mask].reshape((n_events, n_marker - n_removed))
                 
                latents, reconstruction, prediction = self.model(input_sample_masked, 
                                                                markers_pos_masked,
                                                                markers_pos)
                
                # calc loss
                reconstruction_for_loss = reconstruction[~markers_mask].reshape((n_events, n_removed))
                input_sample_for_loss = input_sample[~markers_mask].reshape((n_events,n_removed))
                sample_loss = self.model.ae_loss(reconstruction_for_loss, input_sample_for_loss)
                
                # update batch loss
                ae_loss += sample_loss
                loss += sample_loss
                
                if self.model.supervision:
                    sup_sample_loss = self.loss_ftn(prediction, current_y)
                    supervised_loss += sup_sample_loss
                    loss +=  (2*sup_sample_loss)
                               
            # we are finished with looping over batch
            loss /= batch_size
            ae_loss /= batch_size
            if self.model.supervision:
                supervised_loss /= batch_size

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) 
            self.optimizer.step()
            
            # save metrics
            self.train_metrics.update('loss', loss.item(), n=batch_size)
            self.train_metrics.update('ae_loss', ae_loss.item(), n=batch_size)
            if self.model.supervision:
                self.train_metrics.update('supervised_loss', supervised_loss.item(), n=batch_size)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(prediction, current_y), n=batch_size)


            if batch_idx == self.len_epoch: 
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log, median_log = self.train_metrics.result()
        
        return log, median_log, log_plots
    
    def _get_marker_ordering_and_position(self, n_events, markerlist, p_removed: float = 0.5):
        '''
        markerlist: list of markers that are in markerdict and in sample
                    from those it is chosen what to mask
        p_remove: percentage of markers to mask
        '''

        n_marker = len(markerlist)
        # how many marker to mask based on percentage
        n_remove = int(len(markerlist)*p_removed)

        # create masking matrix n_events x n_marker, true are markers to be kept
        values = torch.rand((n_events, n_marker))
        _, indices = torch.topk(values, n_remove, largest=False, sorted=True)
        marker_mask = torch.ones((n_events, n_marker), dtype=torch.bool)
        marker_mask.scatter_(1, indices, False)
    
        # create matrix with position of markers in marker dict n_events x n_marker
        markers_pos = torch.tensor([self.marker_dict[m] for m in markerlist])
        markers_pos = markers_pos.repeat((n_events, 1))
        markers_pos_masked = markers_pos[marker_mask].reshape((n_events, n_marker-n_remove)) # n_events x n_marker_not_masked

        return marker_mask, markers_pos, markers_pos_masked
    
    def _val_epoch(self, epoch):
        """
        Validate between training epochs

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
                    for batch_idx, batch in tqdm(enumerate(self.valid_data_loader), 
                                                 desc= 'eval', 
                                                 total=len(self.valid_data_loader)):
                        # Load data
                        batch["data"][0] = batch["data"][0].float()
                        x = batch["data"][0].to(self.device)
                        target = batch["target"][0].float().to(self.device).unsqueeze(-1)
      
                        filenames.append(batch["name"][0]) # because it is a batch so we have a list of names                
                    
                        # Forward + backward pass
                        m_list = [m for m in batch["marker"][0] if m in self.marker_dict]
                        markers_idx = [batch["marker"][0].index(m) for m in m_list]
                        _, markers_pos, _ = self._get_marker_ordering_and_position(x.shape[0], m_list)
                    
                        latents, reconstruction, prediction = self.model(x[:,markers_idx], markers_pos, markers_pos) # apply model 

                        ae_loss = self.model.ae_loss(reconstruction, x[:,markers_idx])

                        loss = ae_loss
                        if self.model.supervision:
                            supervised_loss = self.loss_ftn(prediction, target)
                            loss += (2*supervised_loss)
             
                        self.valid_metrics.update('loss', loss.item())
                        self.valid_metrics.update('ae_loss', ae_loss.item())
                        if self.model.supervision:
                            self.valid_metrics.update('supervised_loss', supervised_loss.item())
                            for met in self.metric_ftns:
                                self.valid_metrics.update(met.__name__, met(prediction, target))

                    # MRD figure
                    if self.model.supervision:
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