import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, label_path, opts, target_transform=None, source_transform=None):
		super().__init__()
		self.label_path = label_path
		self.source_root = source_root
		self.target_root = target_root
		
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		
		self._load_raw_labels()
	
	def _load_raw_labels(self):
		with open(self.label_path, "r") as f:
			data = json.load(f)
			cams = data['labels']
			shs = data['sh']
		self.labels = {
			'cam': cams, 
			'shs': shs, 
		}

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		fname = os.path.relpath(from_path, start=self.source_root)
		label = {
			'cam': torch.from_numpy(np.array(self.labels['cam'][fname])).reshape(-1),
			'sh': torch.from_numpy(np.array(self.labels['shs'][fname])).reshape(-1), 
		}
		
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im, label
