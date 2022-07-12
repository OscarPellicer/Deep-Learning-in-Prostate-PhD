import os, glob, sys, numpy as np, abc
import torch, pytorch_lightning as pl
from collections import defaultdict
from copy import copy
import torchio as tio

from .ImageDataLoader import MedicalImageDataset

class _RepeatSampler():
    ''' Sampler that repeats forever '''
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class ContinuousDataLoader(torch.utils.data.DataLoader):
    '''
        Dataloader that does not get reloaded after every epoch.
        It is a drop-in replacement for torch.utils.data.DataLoader
        that minimizes between-epoch idle times and allows for data caching
        within Datasets
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
        
class SegmentationDataModule(pl.LightningDataModule, metaclass=abc.ABCMeta):
    def __init__(self, path, batch_size=4, num_workers=1, 
                 seed=42, keep_condition= lambda a: True, 
                 **data_loader_kwargs):
        '''
            General segmentation (and otherwise) data module to be used for loading medical images
            from paths. The classes using it must fulfill the contract of having a method _get_subset_ids
            that creates self.data_dict, self.train_ids, self.val_ids and self.test_ids
            
            Parameters
            ----------
            path: str
                First path to the data (in case of multiple)
            batch_size: int, default 4
                Batch size
            num_workers: int, default 1
                Number of working threads for loading the data (set to at least 2 in general)
            seed: int, default 42
                RNG seed
            keep_condition: function, default lambda a: True
                Funtion that takes a patient ID as input (the asterisk * in sequence_equivalences)
                and returns True if the patient is to be kept, and False if they must be discarded
            **data_loader_kwargs: **dict
                Other LightningDataModule keyword parameters
        '''
        super().__init__()
        self.path= path
        self.rng= np.random.default_rng(seed)
        self.data_loader_kwargs= data_loader_kwargs
        self.batch_size, self.num_workers= batch_size, num_workers
        self.keep_condition= keep_condition
        
        #Attributes to fill by _get_subset_ids
        self.data_dict= defaultdict(dict)
        self.train_ids, self.val_ids, self.test_ids, self.predict_ids= [], [], [], []
        
    def setup(self, stage=None):
        #Call the building method defined by the subclass
        self._get_subset_ids()
        
        #Perform some subset independence checks
        assert not set(self.train_ids).intersection(set(self.val_ids))
        assert not set(self.train_ids).intersection(set(self.test_ids))
        assert not set(self.val_ids).intersection(set(self.test_ids))
        
        #Get a Dataset instance for each subset
        self.train_dataset= MedicalImageDataset(self.data_dict, ids=self.train_ids, **self.data_loader_kwargs)
        self.val_dataset= MedicalImageDataset(self.data_dict, ids=self.val_ids, **self.data_loader_kwargs)
        
        #The test dataset has some special properties
        test_dataloader_kwargs= copy(self.data_loader_kwargs)
        test_dataloader_kwargs['cache']= False
        test_dataloader_kwargs['save_extra_metadata']= True
        self.test_dataset= MedicalImageDataset(self.data_dict, ids=self.test_ids, **test_dataloader_kwargs)
        
        #We create a special predict dataset with all the data
        self.predict_dataset= MedicalImageDataset(self.data_dict, ids=self.predict_ids, **test_dataloader_kwargs)

    @abc.abstractmethod
    def _get_subset_ids(self):
        '''
            This method must create the following class attributes: 
            data_dict, train_ids, val_ids and test_ids
        '''
        pass

    def train_dataloader(self):
        return ContinuousDataLoader(self.train_dataset, batch_size=self.batch_size,
                                    drop_last=False, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return ContinuousDataLoader(self.val_dataset, batch_size=self.batch_size,
                                    drop_last=False, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return ContinuousDataLoader(self.test_dataset, batch_size=1,
                                    drop_last=False, num_workers=1, shuffle=False)

    def predict_dataloader(self):
        return ContinuousDataLoader(self.predict_dataset, batch_size=1,
                                    drop_last=False, num_workers=1, shuffle=False)

    def teardown(self, stage=None):
        pass
    
    def plot(self, *args, **kwargs):
        raise NotImplementedError('No plotting implemented into the dataloader as of yet')
            
class PathDataModule(SegmentationDataModule):
    def __init__(self, *args,
                 subset_percentages= {'train':0.7, 'val':0.15, 'test':0.15}, shuffle=True,
                 dataset='IVO', sorting_fn=lambda p: int(p.split('_')[1]), 
                 id_making_fn=lambda i: '_'.join(i.split('_')[:2]), **kwargs):
        '''
            Pattern-based DataModule. 
            It takes one (or several) paths, and one (or several) sequence_equivalences
            Provide all paths first and then all sequence_equivalences
            
            Parameters
            ----------
            *paths: *str
                Paths (must have corresponding sequence_equivalences).
            *sequence_equivalences: *dict
                Dictionaries keys like '*_T2.nrrd', where * corresponds to a variable string that 
                will make up the patient ID, and the corresponding value would be somehting like 'T2'
                Example: { '*_urethra_MR.seg.nrrd':'MR_UR', '*_urethra_US.seg.nrrd':'US_UR', 
                            '*_MR_img.nrrd':'MR', '*_US_img.nrrd':'US', 
                            '*_MR_msk.nrrd':'MR_mask', '*_US_msk.nrrd':'US_mask',
                            '*_MR_img_cor.nrrd': 'MR_cor', '*_MR_img_sag.nrrd': 'MR_sag'}
            subset_percentages: dict, default {'train':0.7, 'val':0.15, 'test':0.15}
                Percentages of the data to use for train, val and test. Must sum up to one
            shuffle: bool, default True
                Shuffle data before partitioning
            dataset: str, default 'IVO'
                Dataset identifier, only used for metadata purposes (see SegmentationDataModule)
            sorting_fn: callable, default lambda p: int(p.split('_')[1])
                Takes an ID as input, and makes it sortable
            id_making_fn: callable, lambda i: '_'.join(i.split('_')[:2])
                Takes an image name, and returns just the id
    
        '''
        super().__init__(args[0], **kwargs)
        
        self.paths= args[:len(args)//2]
        self.sequence_equivalences= args[len(args)//2:]
        self.sp= subset_percentages
        self.dataset= dataset
        self.shuffle= False
        self.sorting_fn= sorting_fn
        self.id_making_fn= id_making_fn
        
        #Perform some sanity cecks
        ss_sum= np.sum(list(self.sp.values()))
        assert ss_sum == 1, f'subset_percentages must sum up to 1 (sum: {ss_sum:.4f})'
        assert len(args) % 2 == 0, 'There must be as many paths as squence_equivalences provided'
        assert all([isinstance(se, dict) for se in self.sequence_equivalences]), 'All squence_equivalences must be dicts'
        assert all([isinstance(se, str) for se in self.paths]), 'All paths must be strings'

    def _get_subset_ids(self):
        #Recover PIDs and build data dict
        pids= sorted([self.id_making_fn(os.path.basename(os.path.normpath(path)))
                     for path in glob.glob(os.path.join(self.path, next(iter(self.sequence_equivalences[0].keys()))))], 
                     key=self.sorting_fn)
        self.predict_ids= pids
        self.data_dict= { pid : { **{name : os.path.join(path, pattern.replace('*', pid))
                             for se, path in zip(self.sequence_equivalences, self.paths) for pattern, name in se.items() },
                         'meta': {'dataset':self.dataset, 'pid':pid}} for pid in pids}
        
        #Split ids in train, val, test subsets
        pids= list(filter(self.keep_condition, pids))
        if self.shuffle: self.rng.shuffle(pids)
        N, first_subset= len(pids), next(iter(self.sp.keys()))
        if first_subset == 'train':
            self.train_ids= pids[:int(self.sp['train']*N)]
            self.val_ids= pids[int(self.sp['train']*N) : int((self.sp['train'] + self.sp['val'])*N)]
            self.test_ids= pids[int((self.sp['train'] + self.sp['val'])*N):]
        elif first_subset == 'test':
            self.test_ids= pids[:int(self.sp['test']*N)]
            self.val_ids= pids[int(self.sp['test']*N) : int((self.sp['test'] + self.sp['val'])*N)]
            self.train_ids= pids[int((self.sp['test'] + self.sp['val'])*N):]
        
class SubsetDataModule(SegmentationDataModule):
    def __init__(self, path, subsets_dict, load_by_id, modality, **kwargs):
        '''
            Function-based datamodule with pre-split subsets
            
            Parameters
            ----------
            path: str
                First path to the data (in case of multiple)
            subsets_dict: dict of dicts
                Build subsets_dict like {'IVO_1':{'MR': path, 'MR_mask': path}, Promise12_1:{'MR': ...}}
            load_by_id: callable
                Function that takes an ID and returns the path to that ID
            modality: str
                Modality of the image being loaded
        '''
        super().__init__(path, **kwargs)
        self.subets_dict= subsets_dict
        self.load= load_by_id
        self.modality= modality
        
    def _get_subset_ids(self):
        for modality, subsets in self.subets_dict.items():
            if modality == self.modality:
                for subset, datasets in subsets.items():
                    for dataset, pids in datasets.items():
                        for pid in pids:
                            img_path, mask_path= self.load(pid, self.path, modality)
                            self.data_dict[pid][modality]= img_path
                            self.data_dict[pid]['meta']= {'dataset':dataset, 'pid':pid}
                            self.data_dict[pid][f'{modality}_mask']= mask_path
                            if self.keep_condition(pid):
                                if subset == 'train': self.train_ids.append(pid)
                                elif subset == 'val': self.val_ids.append(pid)
                                elif subset == 'test': self.test_ids.append(pid)
                            self.predict_ids.append(pid)