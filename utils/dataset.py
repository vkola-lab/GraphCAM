"""Dataset class for the graph classification task."""

import os
from warnings import warn
from typing import Any

import torch
from torch.utils import data
# import numpy as np
from PIL import ImageFile
# from torchvision import transforms
# import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


# def collate_features(batch):
#     img = torch.cat([item[0] for item in batch], dim=0)
#     coords = np.vstack([item[1] for item in batch])
#     return [img, coords]


# def eval_transforms(pretrained=False):
#     if pretrained:
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)

#     else:
#         mean = (0.5, 0.5, 0.5)
#         std = (0.5, 0.5, 0.5)

#     trnsfrms_val = transforms.Compose(
#         [
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#         ]
#     )

#     return trnsfrms_val


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 target_patch_size: int | None = None,
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "site/graph_name\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)

        The dataset directory structure should be as follows:
        root/
        └── {site}_features/
            └── simclr_files/
                ├── graph1/
                │   ├── features.pt
                │   └── adj_s.pt
                ├── graph2/
                │   ├── features.pt
                │   └── adj_s.pt
                └── ...
        """
        super(GraphDataset, self).__init__()
        self.root = root
        self.ids = ids
        # self.target_patch_size = target_patch_size

        if classdict is not None:
            self.classdict = classdict
        else:
            if site is None:
                warn('Neither site nor classdict provided. Assuming class labels are integers.')
                self.classdict = None
            elif site in {'LUAD', 'LSCC'}:
                self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
            elif site == 'NLST':
                self.classdict = {'normal': 0, 'tumor': 1}
            elif site == 'TCGA':
                self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
            else:
                raise ValueError(f'Site {site} not recognized and classdict not provided')
        self.site = site

        # self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Handle graph names with periods in them if site is None, otherwise preserve behavior
            graph_name = info.split('\t')[0].rsplit('.', 1)[0] if (self.site is not None) else \
                info.split('\t')[0]
            site, graph_name = graph_name.split('/')
            label = info.split('\t')[1]
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc

        if self.site is not None:
            assert self.site == site, f'ID {index} is of site {site}, not {self.site}'

        if site in {'LUAD', 'LSCC'}:
            site = 'LUNG'
            graph_path = os.path.join(self.root, 'CPTAC_{}_features'.format(site))
        elif site == 'NLST':
            graph_path = os.path.join(self.root, 'NLST_Lung_features')
        elif site == 'TCGA':
            graph_name = info.split('\t')[0]
            _, graph_name = graph_name.split('/')
            graph_path = os.path.join(self.root, 'TCGA_LUNG_features')
        else:
            graph_path = os.path.join(self.root, f'{site}_features')
        graph_path = os.path.join(graph_path, 'simclr_files')

        sample: dict[str, Any] = {}
        sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample['id'] = graph_name

        feature_path = os.path.join(graph_path, graph_name, 'features.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'features.pt for {graph_name} doesn\'t exist')

        adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_name} doesn\'t exist')
        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        sample['image'] = features
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)
