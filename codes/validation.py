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
from post_processing import PostProcessingTiledImages

# Argument parser
parser = argparse.ArgumentParser(description="MBG Training")
parser.add_argument("--config-file", default="config.yaml", metavar="FILE", help="path to the general config file")
parser.add_argument("--test-fold", default=None, type=int, help="path to data")
parser.add_argument("--val-fold", default=None, type=int, help="path to data")
parser.add_argument("--object", default=None, type=str, help="object to be used")
parser.add_argument("--datatype", default=None, type=str, help="tiled or integer")
args = parser.parse_args()

# Naming datasets
folds_train = ''
for i in range(1,6):
    if i == args.val_fold or i == args.test_fold:
        continue
    folds_train += str(i)
train_data = f"train{folds_train}_{args.object}"
val_data = f"val{args.val_fold}_{args.object}"

# Initialize Yaml general configuration
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
# Initialize Detectron2 configuration
cfg = get_cfg()

# Register datasets using the parameters from JSON
register_mosquitoes(fold_val=args.val_fold,fold_test=args.test_fold)

#get original anotatios directory path (to be used in post processing for tiled images)
original_json_path = config['REGISTER_DATASETS']['JSON_PATH']

# Access the specific model configuration
model_name = (config['TRAINING']['MODEL_NAME']).upper()
model_config = config['MODELS'][model_name]

# Set model parameters
cfg.merge_from_file(model_config['_BASE_'])
cfg.DATASETS.TRAIN = (f"mbg_{train_data.lower()}",)
cfg.DATASETS.VAL = (f"mbg_{val_data.lower()}",)
cfg.OUTPUT_DIR = config['TEST']['OUTPUT_DIR']
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
        'score': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Pr': [],
        'Rc': [],
        'F1': [],
        'AP50': [],
    }

res = init_res_dict()

scores = np.arange(0.1, 1, 0.02).tolist()

best_score = None
best_f1 = -1

for score in scores:
    print(f'EVALUATION USING SCORE = {score}')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score

    trainer = DefaultTrainer(cfg)
    #trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    val_loader = build_detection_test_loader(cfg, f"mbg_{val_data.lower()}")
    evaluator = COCOEvaluator(f"mbg_{val_data.lower()}",
                              cfg,
                              False,
                              output_dir=os.path.join(cfg.OUTPUT_DIR, f"mbg_{val_data.lower()}"))
    cfn_mat = CfnMat(f"mbg_{val_data.lower()}", output_dir=cfg.OUTPUT_DIR)

    results = inference_on_dataset(
        trainer.model,
        val_loader,
        DatasetEvaluators([evaluator, cfn_mat]),
    )

    if args.datatype.lower() == "integer":

        pr = results['tp'] / (results['tp'] + results['fp'] + 1e-16)
        rc = results['tp'] / (results['tp'] + results['fn'] + 1e-16)
        f1 = (2 * pr * rc) / (pr + rc + 1e-16)

        res['score'].append(score)
        res['TP'].append(results['tp'])
        res['FP'].append(results['fp'])
        res['FN'].append(results['fn'])
        res['AP50'].append(results['bbox']['AP50'])
        res['Pr'].append(pr)
        res['Rc'].append(rc)
        res['F1'].append(f1)
    
    elif args.datatype.lower() == "tiled":
        evaluator_output_dir = os.path.join(cfg.OUTPUT_DIR, f"mbg_{val_data.lower()}")
        json_path_predicoes = os.path.join(evaluator_output_dir, "coco_instances_results.json")
        json_path_original = os.path.join(original_json_path, f'coco_format_val{args.val_fold}_{args.object}.json')
        results = PostProcessingTiledImages(
                            pred_json_path = json_path_predicoes,
                            original_annotations_json_path = json_path_original,
                            confidence_threshold = score)
        
        f1 = results['F1']
        res['score'].append(score)
        res['TP'].append(results['TP'])
        res['FP'].append(results['FP'])
        res['FN'].append(results['FN'])
        res['AP50'].append('-')
        res['Pr'].append(results['Precision'])
        res['Rc'].append(results['Recall'])
        res['F1'].append(results['F1'])

    # Update best score if current score has a higher F1 or if it ties F1 but has a higher score
    if f1 > best_f1 or (f1 == best_f1 and score > best_score):
        best_f1 = f1
        best_score = score

# Create DataFrame from results
df = pd.DataFrame(res)

# Filter results for the best score
best_results = df[df['score'] == best_score]

# Save filtered results to CSV
save_results_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
name_base = f'val_results_{args.object}_train{folds_train}_val{args.val_fold}_test{args.test_fold}'
best_results.to_csv(os.path.join(save_results_dir, name_base + '.csv'), index=False)

print(f"Results saved for the best score (F1 = {best_f1}, score = {best_score})")
