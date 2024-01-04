import torch
from pathlib import Path
from typing import Dict

from src.utils.dynamictypeloader import init_ftn, load_type_dynamically_from_fqn
from src.utils.standardizationhandler import StandardizationHandler
from src.utils.utils import get_project_root
from src.utils.datastructures import DataLoaderConfig, FlowSampleConfig
from src.data_transforms.data_transforms import TransformStack
from src.utils.markercollection import MarkerCollection

class FlowDataset(torch.utils.data.Dataset):
    """
    Flow cytometry pytorch dataset. Processing preloaded files saved as pkl.
    """

    def __init__(self, preload_dir: Path,
                 data_splits_dir: Path,
                 dataset_type: str,
                 enable_standardization: bool,
                 flowsample_config: FlowSampleConfig,
                 transform: TransformStack,
                 ):  # for data augmentation stuff

        assert dataset_type in ['train', 'eval', 'test'], \
            f'dataset_type must be either "train", "eval" or "test" but got "{dataset_type}"'

        self.dataset_type = dataset_type  # train, test, val  check if in allowed dataset types
        self.data_splits_root = data_splits_dir
        self.flowsample_config = flowsample_config
        self.enable_standardization = enable_standardization
        self.preload_dir = preload_dir

        self.transform = transform #if transform is not None else (lambda x, y: (x, y))
        self.fcm_sample_type = load_type_dynamically_from_fqn(flowsample_config.sample_type)

        self.data_list_path = self.data_splits_root / Path(self.dataset_type + '.txt')
        self.filepaths = self._get_filepath_list()



        if self.enable_standardization:
            self.strdhandler = StandardizationHandler(self.preload_dir, self.dataset_type, self.flowsample_config.markers)
            self.strdhandler.prepare_stats(self.filepaths)

    @classmethod
    def init_from_config(cls,
                         config:  DataLoaderConfig,
                         dataset_type: str):

        if not isinstance(config,  DataLoaderConfig):
            raise TypeError(f"config must be from Type 'DataLoaderConfig' given {type(config)}")

        preload_dir = config.preloaded_data_dir
        data_splits_dir = get_project_root() / config.data_splits_dir

        # setup transforms
        if config.transforms_config is not None:
            transform_stack = TransformStack([init_ftn(config_tform) for config_tform in config.transforms_config.transforms_list],
                                             execution_probability=config.transforms_config.execution_probability, # todo each on has seperate execution probs?
                                             dataset_type=dataset_type)
        else:
            transform_stack = None

        return cls(preload_dir=preload_dir,
                   data_splits_dir=data_splits_dir,
                   dataset_type=dataset_type,
                   flowsample_config=config.flow_sample_config,
                   enable_standardization=config.enable_standardization,
                   transform=transform_stack)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx) -> Dict:
        filepath = self.filepaths[idx]
        spl = self.fcm_sample_type.load_from_pickle_files(Path(self.preload_dir), 
                                                          Path(filepath).with_suffix('').name, 
                                                          self.flowsample_config, 
                                                          apply_marker_filter=True, 
                                                          apply_gate_filter=True)

        n_events = len(spl.events.index)
        marker = spl.get_marker_list()
        #marker = MarkerCollection.renameMarkers_after_preload(marker) # in case markerdict has changed after preloading - but actually should not be necessary
        data = torch.tensor(spl.events.values)
        target_values = spl.get_class_labels()
        target = target_values if isinstance(target_values, torch.Tensor) else torch.tensor(target_values.values)

        if self.enable_standardization:
            data = self.strdhandler.standardize_tensor(data)

        if self.transform is not None:
            data, target = self.transform(data, target)  # only applies transform for train data except minmaxscale (see transformstack class)
        name = spl.name

        return {'data': data.float(),
                'marker': marker,
                'target': target.float(),
                'name': name,
                'n_events': n_events,
                'filepath': str(filepath)}

    def _get_filepath_list(self):
        """
        Get list of experiment files (*.xml and *.analysis) from list self.data_list_path.
        """
        with open(self.data_list_path, 'r') as file:
            files = file.readlines()
        files = [f.strip() for f in files]
        files = [Path(f) for f in files] #if Path(f).is_file()]

        return files