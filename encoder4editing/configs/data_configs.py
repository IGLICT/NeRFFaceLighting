import os
from configs import transforms_config
from configs.paths_config import dataset_paths

DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'label_path': os.path.join(dataset_paths['ffhq'], 'dataset.json'), 
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['ffhq'],
		'test_target_root': dataset_paths['ffhq'],
	},
}