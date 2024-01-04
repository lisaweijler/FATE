import argparse
import json
import torch
from torch.utils.data import DataLoader, default_collate
from typing import Sequence
from collections.abc import Mapping

from src.data_loader.dataset_base import FlowDataset
from src.models.supervised_model import SupervisedModel
from src.trainer.supervised_trainer import FATETrainer
from src.utils.configparser import ConfigParser
from src.utils.datastructures import SupervisedFATEConfig
from src.utils.dynamictypeloader import init_obj, init_ftn
from src.utils.loggingmanager import LoggingManager
from src.utils.wandb_logger import WandBLogger

MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)

def custom_collate(batch):
    '''
    Since there can be different number of markers and events just collate as lists.
    funciton works recursive. 
    whenever just batch is returned - it is simply a list.
    '''
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return batch # not using default collate due to different number of markers
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, Mapping):
        return {key: custom_collate([data[key] for data in batch]) for key in elem}
    elif isinstance(elem, Sequence) and not isinstance(elem, str):
        return batch # for marker lists - just append to list.. 
        #return [custom_collate(s) for s in zip(*batch)]

    raise TypeError(f'DataLoader found invalid type: {type(elem)}')


def train(config: SupervisedFATEConfig, wandb_logger=None):

    # python logger
    logging_manager = LoggingManager(config.logging_config)
    logging_manager.register_handlers(name='train', save_path=config.output_save_dir/'log.txt')
    logger = logging_manager.get_logger_by_name(name='train')

    # create dataset and loader
    logger.info('-'*20 + "Creating data_loader instance..")
    train_data = FlowDataset.init_from_config(config.data_loader_config, 
                                              dataset_type='train')
    
    train_dataloader = DataLoader(train_data, 
                                    batch_size = config.data_loader_config.batch_size, 
                                    num_workers = config.data_loader_config.num_workers, 
                                    shuffle = config.data_loader_config.shuffle,
                                    pin_memory = config.data_loader_config.pin_memory,
                                    collate_fn=custom_collate)
    eval_data = None
    if config.do_eval:
        eval_data = FlowDataset.init_from_config(config.data_loader_config, 
                                                 dataset_type='eval')
        eval_dataloader = DataLoader(eval_data, 
                                        batch_size = 1, 
                                        num_workers = 1, 
                                        shuffle = False,
                                        pin_memory = False,
                                        collate_fn=custom_collate)
    
    logger.info('-'*20 +'Done!')

    # model loading
    logger.info('-'*20 + "Loading your hot shit model..")
    encoder = init_obj(config.supervised_model.encoder)
    pred_head = init_obj(config.supervised_model.pred_head)


    model = SupervisedModel(encoder,
                            pred_head,
                            config.supervised_model.n_marker,
                            config.supervised_model.pos_encoding_dim,
                            config.supervised_model.encoder_out_dim,
                            config.supervised_model.latent_dim)

    logger.info(model) 
    logger.info('-'*20 +'Done!')

    # load pretrained model
    if config.pretrained_model_path is not None:
        logger.info("Loading pretrained model weights: {} ...".format(str(config.pretrained_model_path)))
        checkpoint = torch.load(config.pretrained_model_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        for key in pretrained_dict.keys():
            if key in model_dict.keys():
                model_dict[key]= pretrained_dict[key]
 
        model.load_state_dict(model_dict)

    # load markerdict
    with open(str(config.marker_dict_path), 'r') as fp:
        marker_dict = json.load(fp)

    # push model to gpu and get trainable params
    model.to(f'cuda:{config.gpu_id}')
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # loss, metrics, optimizer, learning rate scheduler
    logger.info('-'*20 + "Initializing loss, metrics, optimizer, lr scheduler.." )
    loss_ftn = init_ftn(config.loss)
    metric_ftns = [init_ftn(config_met) for config_met in config.train_metrics_list]
    optimizer = init_obj(config.optimizer, trainable_params)
    lr_scheduler = init_obj(config.lr_scheduler, optimizer) # returns none if obtj to be initialized (e.g. lr_scheduler) is none
    logger.info('-'*20 +'Done!')

    #WandB logger
    if wandb_logger is None:
        logger.info('-'*20 + "Initializing W&B Logger.." )
        wandb_logger = WandBLogger(config.wandb_config, model, run_config=config)
        logger.info('-'*20 +'Done!')
        dispose = True
    else:
        wandb_logger.watch(model)
        dispose = False

    # get the trainer class
    logger.info('-'*20 + "Initializing your marvellous trainer..")
    trainer = FATETrainer(model, 
                            loss_ftn,
                            metric_ftns,
                            optimizer,
                            config.trainer_config,
                            marker_dict,
                            wandb_logger,
                            config.output_save_dir,
                            config.gpu_id,
                            train_dataloader,
                            eval_data_loader=eval_dataloader, 
                            lr_scheduler=lr_scheduler,
                            resume_path=config.resume_path)
    
    logger.info('-'*20 +'Done!')

    trainer.train()
    if dispose:
        trainer.dispose() #finish WandBLogger

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Training ')
    configParser = ConfigParser(mode="train")
    config_default = "config_templates/train_FATE_config.json"
    args.add_argument('-c', '--config', default=config_default, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='index of which GPU to use')
    
    config = configParser.parse_config_from_args(args)
    train(config)