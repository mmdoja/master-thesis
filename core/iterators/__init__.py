from pyhocon import ConfigTree
from torch.utils.data import DataLoader

import torch


class DataLoaderFactory:

    # change from other project
    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg
        self.num_gpus = 4

    def build(self, split='train'):
        dset = self.cfg.get_string('dataset.dset')
        print(f"dset: {dset}")
        if dset == 'urmp':
            from .urmp import URMPDataset
            ds = URMPDataset.from_cfg(self.cfg, split=split)
        elif dset == 'urmp_testingMIDI':
            from .urmp_testingMIDI import URMPtestingMIDIDataset
            ds = URMPtestingMIDIDataset.from_cfg(self.cfg, split=split)
        elif dset == 'Piano':
            from .urmp_transformer_decoder import URMPDataset
            ds = URMPDataset.from_cfg(self.cfg, split=split)
        elif dset == 'youtube_Piano':
            from .YT_data import YoutubeDataset
            ds = YoutubeDataset.from_cfg(self.cfg, split=split)
        elif dset == 'music21_segment':
            from .YT_data import YoutubeSegmentDataset
            ds = YoutubeSegmentDataset.from_cfg(self.cfg, split=split)
        elif dset == 'youtube_urmp':
            from .YT_data import YoutubeURMPDataset
            ds = YoutubeURMPDataset.from_cfg(self.cfg, split=split)
        else:
            raise Exception

        loader = DataLoader(
            ds,
            batch_size=self.cfg.get_int('batch_size') * self.num_gpus,
            num_workers=self.cfg.get_int('num_workers') * self.num_gpus,
            shuffle=(split == 'train')
        )

        print('Real batch size:', loader.batch_size)

        return loader