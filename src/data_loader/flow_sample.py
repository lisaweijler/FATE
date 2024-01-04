from __future__ import annotations
from pathlib import Path
from typing import List,  Any
from xmlrpc.client import Boolean
import numpy as np

from src.utils.datastructures import FlowSampleConfig
from src.utils.gatecollection import GateCollection
from src.utils.markercollection import MarkerCollection


import pandas as pd
import warnings



def get_gate_filter_mask(gates: pd.DataFrame, filter_gates: List[str]):
        '''
        return the mask that specifies the events( = rows) 
        that are within the specified filter_gates - OR connection
        if it is in any of the specified gates, event will be included
        returns a boolean vector, specifying what event will be filtered out (= False)
        '''
        mask = gates[filter_gates] > 0
        mask = mask.any(axis=1) # check if row (= event) is in any of the gates

        return mask


class FlowSampleFile:

    def __init__(self, 
                 name: str, config: FlowSampleConfig, 
                 events: pd.DataFrame, labels: pd.DataFrame,
                 apply_marker_filter: Boolean, # if true then when property events/labels are called the filtered version is returned
                 apply_gate_filter: Boolean # if true then when property events/labels are called the filtered version is returned
                 ):
        self.config = config
        self._name = name
        self._events = events.loc[:,~events.columns.duplicated()].copy() # deduplicate columns
        self._events = MarkerCollection.renameMarkers(self._events) # standardize marker names
        self._labels = GateCollection.renameGates(labels) # standardize gate names
        self._apply_marker_filter = apply_marker_filter
        self._apply_gate_filter = apply_gate_filter

    @staticmethod
    def preloading_suffixes(config : FlowSampleConfig) -> List[str]:
        return [".pkl", "_labels.pkl"]

    @property
    def name(self) -> Any:
        return self._name

    @property
    def events(self) -> Any:
        if not (self._apply_marker_filter or self._apply_gate_filter):
            return self._events

        # default stuff
        gate_filter_mask = self._labels > -10 # quick and dirty create all True mask
        markers_used = self._events.columns

        if self._apply_gate_filter:
            gate_filter_mask = get_gate_filter_mask(self._labels, self.config.filter_gates)
        if self._apply_marker_filter:
            markers_used = self.config.markers
            if len(markers_used) == 0:
                return self._events.loc[gate_filter_mask].reset_index(drop=True) # use all marker -feature transfomre e.g.

        
        return self._events.loc[gate_filter_mask].reset_index(drop=True).reindex(columns=markers_used, fill_value=0)
        
        
    @property
    def events_all_markers(self) -> Any:
        if not self._apply_gate_filter:
            return self._events

        # default stuff
        gate_filter_mask = self._labels > -10 # quick and dirty create all True mask


        if self._apply_gate_filter:
            gate_filter_mask = get_gate_filter_mask(self._labels, self.config.filter_gates)


        return self._events.loc[gate_filter_mask, :].reset_index(drop=True)

    @property
    def labels(self) -> Any:
        # add blast columns if not there
        if not [blastgate for blastgate in GateCollection.getBlastGates() if blastgate in self._labels.columns]:
            self._labels['blast'] = 0
        if self._apply_gate_filter:
            gate_filter_mask = get_gate_filter_mask(self._labels, self.config.filter_gates)
            return self._labels[gate_filter_mask].reset_index(drop=True)
        return self._labels
    
    @classmethod
    def load_from_pickle_files(cls, folder: Path, sample_name: str, config: FlowSampleConfig,  apply_marker_filter: bool = False,
                               apply_gate_filter: bool = False) -> FlowSampleFile:
        filepath_events = folder / Path(sample_name + cls.preloading_suffixes(config)[0])
        filepath_labels = folder / Path(sample_name + cls.preloading_suffixes(config)[1])
        return cls.load_from_pickle(filepath_events, filepath_labels, config,
                                               apply_marker_filter=apply_marker_filter, apply_gate_filter=apply_gate_filter)

    @classmethod
    def load_from_pickle(cls, filepath_events: Path, filepath_labels: Path, config: FlowSampleConfig,
                         apply_marker_filter: bool = False, 
                         apply_gate_filter: bool = False) -> FlowSampleFile:
        ''' 
        Load an flowsample from .pkl 
        used to loade the previously pickled pandas DF in Raw data dir. 
        to further process into a graph and save in processed dir.
        '''

        if not isinstance(config,  FlowSampleConfig):
            raise TypeError(f"config must be from Type 'DataSetConfig' given {type(config)}")

        if filepath_events.suffix != '.pkl':
            warnings.warn(
                "The filepath suffix of {filepath_events} is not .pkl. "
                "Are you sure it is the right path you gave me? ")
        if filepath_labels.suffix != '.pkl':
            warnings.warn(
                "The filepath suffix of {filepath_labls} is not .pkl. "
                "Are you sure it is the right path you gave me? ")

        events = pd.read_pickle(filepath_events)
        labels = pd.read_pickle(filepath_labels)
        name = filepath_events.stem 
        return cls(name, config, events, labels, apply_marker_filter, apply_gate_filter)

        
        
    def get_marker_list(self) -> List[str]:

        return list(self.events.columns)

    def get_all_markers_list(self) -> List[str]:

        return list(self.events_all_markers.columns)

    def get_gate_labels(self) -> List[str]:
        return list(self.labels.columns)
    
    def get_marker_list_idx(self) -> List[int]:
        '''returns the idx of the markers that are filtered for in the full markers list
            all markers. [a,b,c,d,e], markers: [b,e] -> this function returns [1,4]
        '''

        # this shouldn't happen at all but just to be sure
        assert len(self.get_all_markers_list()) == list(set(self.get_all_markers_list()))
        return [self.get_all_markers_list().index(m) for m in self.get_marker_list()]


    def get_class_labels(self) -> pd.Series:
        '''
        This goes through the specified order, and only assigns this label if this event hasnt been used before..
        This means that the order should be {blast, cd19, other} for example
        '''

        class_labels = pd.Series(self.config.classes_dict['other'], 
                                 index=np.arange(len(self.events.index)))

        mask_temp = pd.Series(False, index=np.arange(len(self.events.index)))
        for key in self.config.classes_dict.keys():
            if key != 'other':
                if key not in self.get_gate_labels():
                    mask_key = pd.Series(data=False, index=self.labels.index)
                    # raise ValueError(f"Specified class: {key} is not present in gate labels of sample {self.name}.")
                else:
                    mask_key = self.labels[key] > 0
                mask = (mask_key & ~(mask_temp)) # have 1 for the current key column but have not been assigned a label before already!
                class_labels[mask] = self.config.classes_dict[key]
                mask_temp = (mask_key | mask_temp)

        return class_labels




