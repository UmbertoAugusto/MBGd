import os
from itertools import product
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import numpy as np
import yaml

# Load the general YAML configuration
with open('./configs/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
FOLDS = config['REGISTER_DATASETS']['FOLDS']
JSON_PATH = config['REGISTER_DATASETS']['JSON_PATH']
FRAMES_PATH = config['REGISTER_DATASETS']['FRAMES_PATH']
OBJECTS = config['REGISTER_DATASETS']['OBJECTS']

def register_datasets(dataset_names, json_dir, img_root):

    for d in dataset_names:
        register_coco_instances(f"mbg_{d.lower()}",{},os.path.join(json_dir, f"coco_format_{d}.json"),img_root,) #coco_format_train123_tire.json eh o dataset mbg_train123_tire
        cdc_metadata = MetadataCatalog.get(f"mbg_{d.lower()}")
        
def register_mosquitoes(fold_val,fold_test):

    folds_train = ''
    for i in range(1,6):
        if i == fold_val or i == fold_test:
            continue
        folds_train += str(i)
    sets = [f"train{folds_train}"]
    sets += [f"val{fold_val}"]
    sets += [f"test{fold_test}"]

    comb = list(product(sets, OBJECTS))

    sets = ["_".join(c) for c in comb]

    register_datasets(sets, JSON_PATH, FRAMES_PATH)

if __name__ == "__main__":
    register_mosquitoes(fold_val=1,fold_test=5)