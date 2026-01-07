import json
import numpy as np
import torch
from collections import defaultdict
import os
from sahi.prediction import ObjectPrediction
from torchvision.ops import nms

# --- FUNÇÕES AUXILIARES ---

# --- Parâmetros ---
#valores usados para pneus:
TILE_SIZE = 640
OVERLAP_RATIO = 0.067 #para pneus eh 0.067
NUM_COLUMNS = 7
NUM_ROWS = 4
OVERLAP_PIXELS = int(TILE_SIZE * OVERLAP_RATIO)

NMS_MATCH_THRESHOLD = 0.5 # acima disso eh considerado previsao duplicada e so fica uma
# ------------------------------------

def parse_custom_filename_and_calculate_shift(filename_stem):
    """
    Extrai o nome da imagem original e calcula as coordenadas de deslocamento
    a partir de um nome de arquivo no formato 'video_XX_frameYYYY__tileZZZZ'.
    """
    # 1. Separa o nome original do ID do tile
    # Ex: 'video_10_frame0000__tile0009' -> ('video_10_frame0000', '0009')
    try:
        original_name, tile_id_str = filename_stem.split('__tile')
        tile_id = int(tile_id_str)
    except ValueError:
        # Fallback para caso o nome do arquivo não siga o padrão esperado
        print(f"Aviso: Não foi possível parsear o nome do arquivo: {filename_stem}")
        return None, None, None, None

    # 2. Calcula o passo (stride)
    step = TILE_SIZE - OVERLAP_PIXELS  # Ex: 640 - 42 = 598

    # 3. Calcula a posição na grade (linha e coluna)
    col = tile_id % NUM_COLUMNS
    row = tile_id // NUM_COLUMNS

    # 4. Calcula as coordenadas de deslocamento (shift_amount)
    shift_x = col * step
    shift_y = row * step
    shift_amount = [shift_x, shift_y]
    
    return original_name, shift_amount, row, col

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



def PostProcessingTiledImages(pred_json_path, original_annotations_json_path, tiled_annotations_json_path, confidence_threshold, iou_thresh_tp_fp=0.5):
    # Load json files
    with open(pred_json_path, 'r') as f: # predicoes
        faster_preds = json.load(f)

    with open(original_annotations_json_path, 'r') as f: # anotacoes nos quadros originais (inteiros)
        original_gt = json.load(f)

    with open(tiled_annotations_json_path, 'r') as f: # anotacoes nos quadros cortados, para relacionar image_id
        tiled_gt = json.load(f)

    # Mapa: ID Tiled -> Nome do Arquivo Tiled
    tiled_images_map = {img['id']: img['file_name'] for img in tiled_gt['images']}

    # Mapa: ID Original -> Info da Imagem Original (para iterar depois)
    gt_images_map = {img['id']: img for img in original_gt['images']}

    # Mapa: ID Original -> Anotações (Bbox)
    gt_annotations_map = defaultdict(list)
    for ann in original_gt['annotations']:
        gt_annotations_map[ann['image_id']].append(ann)

    # Read predictions
    raw_preds_per_original_image = defaultdict(list)

    for pred in faster_preds:
        if pred['score'] < confidence_threshold:
            continue
            
        image_id_int = pred['image_id'] # Na Faster/Detectron, 'image_id' é um int

        # Pega nome da imagem com aquele id usando o mapa de anotacoes em tiled frames
        if image_id_int not in tiled_images_map:
            print(f"Aviso: ID {image_id_int} não encontrado no Ground Truth.")
            continue

        full_filename = tiled_images_map[image_id_int]
        image_id_stem = os.path.splitext(os.path.basename(full_filename))[0]

        original_name, shift_amount, row, col = parse_custom_filename_and_calculate_shift(image_id_stem)
        
        # Se o parse falhar, pula esta predição
        if original_name is None:
            continue

        raw_preds_per_original_image[original_name].append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id'],
            'shift_amount': shift_amount,
            'row' : row,
            'col' : col
        })


    # NMS
    final_predictions = {}

    for original_name, raw_preds in raw_preds_per_original_image.items():
        # Lista para guardar as predições limpas para esta imagem
        final_preds_for_image = []

        # 1. Converte predições brutas para objetos SAHI para facilitar o manuseio

        # separa area que precisa do NMS (sobreposicao) da que nao precisa
        sahi_predictions_safe = []
        sahi_predictions_overlap = []

        for pred in raw_preds:
            #correcao no formato da bbox
            x_min, y_min, w, h = pred['bbox']
            x_max = x_min + w
            y_max = y_min + h
            row = pred['row']
            col = pred['col']

            # --- LÓGICA DE DECISÃO (SAFE vs OVERLAP) ---
            is_in_overlap = False

            # Checa borda ESQUERDA (só se não for coluna 0)
            if col > 0 and x_min < OVERLAP_PIXELS:
                is_in_overlap = True
            
            # Checa borda SUPERIOR (só se não for linha 0)
            if row > 0 and y_min < OVERLAP_PIXELS:
                is_in_overlap = True
                
            # Checa borda DIREITA (só se não for a última coluna)
            if col < (NUM_COLUMNS - 1) and x_max > (TILE_SIZE - OVERLAP_PIXELS):
                is_in_overlap = True
                
            # Checa borda INFERIOR (só se não for a última linha)
            if row < (NUM_ROWS - 1) and y_max > (TILE_SIZE - OVERLAP_PIXELS):
                is_in_overlap = True
            # --- FIM DA LÓGICA ---

            # aplicando deslocamento para correcao nas coordenadas e criando objeto sahi
            shift_x, shift_y = pred['shift_amount']
            bbox_xyxy = [x_min + shift_x, y_min + shift_y, x_max + shift_x, y_max + shift_y]

            sahi_pred = ObjectPrediction(
                bbox=bbox_xyxy,
                score=pred['score'],
                category_id=pred['category_id'],
                shift_amount=[0, 0],
                full_shape=None
            )

            # Adiciona na lista correta
            if is_in_overlap:
                sahi_predictions_overlap.append(sahi_pred)
            else:
                sahi_predictions_safe.append(sahi_pred)
            
        # 2. Encontra todas as classes únicas presentes nesta imagem
        all_preds = sahi_predictions_safe + sahi_predictions_overlap
        unique_classes = {p.category.id for p in all_preds}

        # 3. Aplica o NMS para cada classe separadamente *APENAS* nas previsões da zona de overlap
        for class_id in unique_classes:
            # Filtra as predições para a classe atual

            # Pega as previsões SEGURAS para esta classe (elas passam direto)
            safe_class_preds = [p for p in sahi_predictions_safe if p.category.id == class_id]
            final_preds_for_image.extend(safe_class_preds)

            # Pega as previsões de OVERLAP para esta classe (estas vão para o NMS)
            overlap_class_preds = [p for p in sahi_predictions_overlap if p.category.id == class_id]

            if not overlap_class_preds:
                continue

            # Prepara os dados para a função nms da torchvision (só da lista de overlap)
            # Bounding boxes no formato (x_min, y_min, x_max, y_max)
            boxes_xyxy = [p.bbox.to_xyxy() for p in overlap_class_preds]
            scores = [p.score.value for p in overlap_class_preds]
            
            # Converte para tensores do PyTorch
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # 4. Executa o NMS da torchvision
            kept_indices = nms(boxes_tensor, scores_tensor, iou_threshold=NMS_MATCH_THRESHOLD)

            # 5. Guarda as predições que sobreviveram ao NMS
            for index in kept_indices:
                final_preds_for_image.append(overlap_class_preds[index])
        
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
            
            if best_iou >= iou_thresh_tp_fp and not gt_matched[best_gt_idx]:
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
    print(PostProcessingTiledImages(pred_json_path='/home/umberto.pereira/data/Umberto/YOLO/outputs/yolo_tiled_tires_5folds/test_fold1/val_fold2/test/predictions.json',
                                    original_annotations_json_path='/nfs/proc/projeto.dpcm/mbgv2a/tire/coco_json_tire/coco_format_test1_tire.json',
                                    confidence_threshold=0.78))