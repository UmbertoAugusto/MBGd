import json
import numpy as np
import torch
from collections import defaultdict
import os
from sahi.prediction import ObjectPrediction
from torchvision.ops import nms

# --- FUNÇÕES AUXILIARES ---

def parse_custom_filename_and_calculate_shift(filename_stem):
    """
    Extrai o nome da imagem original e calcula as coordenadas de deslocamento
    a partir de um nome de arquivo no formato 'video_XX_frameYYYY__tileZZZZ'.
    """
    # --- Parâmetros ---
    TILE_SIZE = 640
    OVERLAP_RATIO = 0.067 #para pneus eh 0.067
    NUM_COLUMNS = 7
    # ------------------------------------

    # 1. Separa o nome original do ID do tile
    # Ex: 'video_10_frame0000__tile0009' -> ('video_10_frame0000', '0009')
    try:
        original_name, tile_id_str = filename_stem.split('__tile')
        tile_id = int(tile_id_str)
    except ValueError:
        # Fallback para caso o nome do arquivo não siga o padrão esperado
        print(f"Aviso: Não foi possível parsear o nome do arquivo: {filename_stem}")
        return None, None

    # 2. Calcula o passo (stride)
    overlap_pixels = int(TILE_SIZE * OVERLAP_RATIO)
    step = TILE_SIZE - overlap_pixels  # Ex: 640 - 42 = 598

    # 3. Calcula a posição na grade (linha e coluna)
    col = tile_id % NUM_COLUMNS
    row = tile_id // NUM_COLUMNS

    # 4. Calcula as coordenadas de deslocamento (shift_amount)
    shift_x = col * step
    shift_y = row * step
    shift_amount = [shift_x, shift_y]
    
    return original_name, shift_amount

def calculate_iou(boxA, boxB):
    """
    Calcula o Intersection over Union (IoU) entre duas caixas delimitadoras.
    Formato da caixa: [x, y, w, h]
    """
    # Determina as coordenadas (x, y) da caixa de interseção
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Calcula a área da interseção
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calcula a área de ambas as caixas
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Calcula a IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Parâmetros
NMS_MATCH_THRESHOLD = 0.5 # acima disso eh considerado previsao duplicada e so fica uma
IOU_THRESHOLD_FOR_MATCH = 0.5 # acima disso previsao eh considerada certa

def PostProcessingTiledImages(pred_json_path, original_annotations_json_path, confidence_threshold):
    # Load json files
    with open(pred_json_path, 'r') as f:
        yolo_preds = json.load(f)

    with open(original_annotations_json_path, 'r') as f:
        original_gt = json.load(f)

    # Criar um mapa de ID para nome de imagem original para facilitar a busca
    gt_images_map = {img['id']: img for img in original_gt['images']}
    gt_annotations_map = defaultdict(list)
    for ann in original_gt['annotations']:
        gt_annotations_map[ann['image_id']].append(ann)

    # Read predictions
    raw_preds_per_original_image = defaultdict(list)

    for pred in yolo_preds:
        if pred['score'] < confidence_threshold:
            continue
            
        image_id_stem = pred['image_id'] # No YOLO JSON, 'image_id' é o stem do nome do arquivo

        original_name, shift_amount = parse_custom_filename_and_calculate_shift(image_id_stem)
        
        # Se o parse falhar, pula esta predição
        if original_name is None:
            continue

        raw_preds_per_original_image[original_name].append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id'],
            'shift_amount': shift_amount
        })


    # NMS
    final_predictions = {}

    for original_name, raw_preds in raw_preds_per_original_image.items():
        # Lista para guardar as predições limpas para esta imagem
        final_preds_for_image = []

        # 1. Converte predições brutas para objetos SAHI para facilitar o manuseio
        sahi_predictions = []
        for pred in raw_preds:
            #correcao no formato da bbox
            x_min, y_min, w, h = pred['bbox']
            x_max = x_min + w
            y_max = y_min + h
            #aplicando deslocamento para correcao nas coordenadas
            shift_x, shift_y = pred['shift_amount']
            bbox_xyxy = [x_min + shift_x, y_min + shift_y, x_max + shift_x, y_max + shift_y]
            sahi_pred = ObjectPrediction(
                bbox=bbox_xyxy,
                score=pred['score'],
                category_id=pred['category_id'],
                shift_amount=[0, 0],
                full_shape=None
            )
            sahi_predictions.append(sahi_pred)
            
        # 2. Encontra todas as classes únicas presentes nesta imagem
        unique_classes = {p.category.id for p in sahi_predictions}

        # 3. Aplica o NMS para cada classe separadamente
        for class_id in unique_classes:
            # Filtra as predições para a classe atual
            class_preds = [p for p in sahi_predictions if p.category.id == class_id]

            if not class_preds:
                continue

            # Prepara os dados para a função nms da torchvision
            # Bounding boxes no formato (x_min, y_min, x_max, y_max)
            boxes_xyxy = [p.bbox.to_xyxy() for p in class_preds]
            scores = [p.score.value for p in class_preds]
            
            # Converte para tensores do PyTorch
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # 4. Executa o NMS da torchvision
            kept_indices = nms(boxes_tensor, scores_tensor, iou_threshold=NMS_MATCH_THRESHOLD)

            # 5. Guarda as predições que sobreviveram ao NMS
            for index in kept_indices:
                final_preds_for_image.append(class_preds[index])
        
        # 6. Converte de volta para o formato simples para a etapa de avaliação
        final_predictions[original_name] = [
            {'bbox': p.bbox.to_xywh(), 'score': p.score.value, 'category_id': p.category.id}
            for p in final_preds_for_image
        ]

    # Metrics
    total_tp, total_fp, total_fn = 0, 0, 0

    # Checks all original images
    for image_id, image_info in gt_images_map.items():
        base_name = os.path.basename(image_info['file_name'])
        original_name = os.path.splitext(base_name)[0]
        
        gt_boxes = [ann['bbox'] for ann in gt_annotations_map.get(image_id, [])]
        pred_boxes = [p['bbox'] for p in final_predictions.get(original_name, [])]
        
        if not gt_boxes and not pred_boxes:
            continue

        # Arrays para marcar quais caixas já foram "match"
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        tp = 0
        fp = 0
        
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= IOU_THRESHOLD_FOR_MATCH and not gt_matched[best_gt_idx]:
                tp += 1
                gt_matched[best_gt_idx] = True # Marca como "usado"
            else:
                fp += 1
                
        fn = len(gt_boxes) - np.sum(gt_matched)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'TP':total_tp,'FP':total_fp,'FN':total_fn,'Precision':precision,'Recall':recall,'F1':f1_score}

if __name__ == '__main__':
    print(PostProcessingTiledImages(pred_json_path='/home/umberto.pereira/data/Umberto/YOLO/outputs/yolo_tiled_tires_1folds/test_fold1/val_fold4/test/predictions.json',
                                    original_annotations_json_path='/nfs/proc/projeto.dpcm/mbgv2a/tire/coco_json_tire/coco_format_test1_tire.json',
                                    confidence_threshold=0.5))