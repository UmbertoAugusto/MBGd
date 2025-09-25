import argparse
import os
import pandas as pd
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.logger import setup_logger
from evaluation import CfnMat
from register_dataset import register_mosquitoes
import numpy as np
import yaml

# Argument parser
parser = argparse.ArgumentParser(description="MBG Training")
parser.add_argument("--config-file", default="config.yaml", metavar="FILE", help="path to the general config file")
parser.add_argument("--test-fold", default=None, type=int, help="path to data")
parser.add_argument("--val-fold", default=None, type=int, help="path to data")
parser.add_argument("--object", default=None, type=str, help="object to be used")
args = parser.parse_args()

# Naming datasets
folds_train = ''
for i in range(1,6):
    if i == args.val_fold or i == args.test_fold:
        continue
    folds_train += str(i)
train_data = f"train{folds_train}_{args.object}"
test_data = f"test{args.test_fold}_{args.object}"

# Initialize Yaml general configuration
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
# Initialize Detectron2 configuration
cfg = get_cfg()

# Register datasets using the parameters from JSON
register_mosquitoes(fold_val=args.val_fold,fold_test=args.test_fold)

# Access the specific model configuration
model_name = (config['TRAINING']['MODEL_NAME']).upper()
model_config = config['MODELS'][model_name]

# Set model parameters
cfg.merge_from_file(model_config['_BASE_'])
cfg.DATASETS.TRAIN = (f"mbg_{train_data.lower()}",)
cfg.DATASETS.VAL = (f"mbg_{test_data.lower()}",)
cfg.OUTPUT_DIR = config['TEST']['OUTPUT_DIR']
save_results_dir = cfg.OUTPUT_DIR
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'train{folds_train}_val{args.val_fold}_test{args.test_fold}_{args.object}')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
setup_logger(cfg.OUTPUT_DIR)
cfg.MODEL.MASK_ON = model_config['MODEL']['MASK_ON']
cfg.MODEL.RESNETS.DEPTH = model_config['MODEL']['RESNETS']['DEPTH']
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = model_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN']
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = model_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST']
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = model_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN']
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = model_config['MODEL']['RPN']['POST_NMS_TOPK_TEST']
cfg.MODEL.RPN.NMS_THRESH = model_config['MODEL']['RPN']['NMS_THRESH']
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = model_config['MODEL']['ROI_BOX_HEAD']['POOLER_SAMPLING_RATIO']
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = model_config['MODEL']['ROI_BOX_HEAD']['POOLER_RESOLUTION']
cfg.MODEL.BACKBONE.FREEZE_AT = model_config['MODEL']['BACKBONE']['FREEZE_AT']
cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_config['MODEL']['ROI_HEADS']['NUM_CLASSES']
cfg.DATALOADER.NUM_WORKERS = config['TRAINING']['TRAINING_NUM_WORKERS']
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['TRAINING']['BATCH_SIZE_PER_IMAGE']
cfg.SOLVER.IMS_PER_BATCH = config['TRAINING']['IMAGES_PER_BATCH']
cfg.SOLVER.BASE_LR = config['TRAINING']['LEARNING_RATE']
cfg.SOLVER.WEIGHT_DECAY = config['TRAINING']['WEIGHT_DECAY']
cfg.SOLVER.STEPS = config['TRAINING']['STEPS']
cfg.SOLVER.MAX_ITER = config['TRAINING']['MAX_ITER']
cfg.MIN_DELTA = config['TRAINING'].get('MIN_DELTA')
cfg.PATIENCE = config['TRAINING'].get('PATIENCE') 
cfg.MODE = config['TRAINING'].get('MODE', 'pocket_test')  
cfg.VAL_PERIOD = config['TRAINING'].get('VAL_PERIOD')  
cfg.AUGMENTATION = config['AUGMENTATION'].get('ENABLE')

# Update model weights to the best model found
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['TEST']['TEST_WEIGHTS'])

def init_res_dict():
    '''iniciates a dictionary to store results metrics'''
    return {
        'fold_test': [],
        'fold_val': [],
        'conf_score': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'AP50': [],
    }

res = init_res_dict()

#getting best conf_score from validation run
val_results_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
name_base = f'val_results_{args.object}_train{folds_train}_val{args.val_fold}_test{args.test_fold}'
val_results_path = os.path.join(val_results_dir, name_base + '.csv')
df = pd.read_csv(val_results_path)
score = float(df['score'].iloc[0])

print(f'EVALUATION USING THE BEST SCORE FROM VALIDATION = ({score})')

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score

trainer = DefaultTrainer(cfg)
#trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

val_loader = build_detection_test_loader(cfg, f"mbg_{test_data.lower()}")
evaluator = COCOEvaluator(f"mbg_{test_data.lower()}",
                          cfg,
                          False,
                          output_dir=os.path.join(cfg.OUTPUT_DIR, f"mbg_{test_data.lower()}"))
cfn_mat = CfnMat(f"mbg_{test_data.lower()}", output_dir=cfg.OUTPUT_DIR)

results = inference_on_dataset(trainer.model,
                               val_loader,
                               DatasetEvaluators([evaluator, cfn_mat]),
                               )

pr = results['tp'] / (results['tp'] + results['fp'] + 1e-16)
rc = results['tp'] / (results['tp'] + results['fn'] + 1e-16)
f1 = (2 * pr * rc) / (pr + rc + 1e-16)

res['fold_test'].append(args.test_fold)
res['fold_val'].append(args.val_fold)
res['conf_score'].append(score)
res['TP'].append(results['tp'])
res['FP'].append(results['fp'])
res['FN'].append(results['fn'])
res['AP50'].append(results['bbox']['AP50'])
res['Precision'].append(pr)
res['Recall'].append(rc)
res['F1'].append(f1)


# Create DataFrame from results
df_results = pd.DataFrame(res)

# Save filtered results to CSV
name_base = 'test_results'
output_file = os.path.join(save_results_dir, name_base + '.csv')

if not os.path.exists(output_file):
    # Add header if file does not exist yet
    df_results.to_csv(output_file, mode='w', header=True, index=False)
else:
    # Add just results
    df_results.to_csv(output_file, mode='a', header=False, index=False)

print(f"Results saved for the best score (F1 = {f1}, score = {score})")
