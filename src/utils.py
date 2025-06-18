import seaborn as sns
tab20b_hex_colors = sns.color_palette('tab20b', n_colors=20).as_hex()
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

PALETA_ATARDECER_GRIETA = {
    # Colores principales / de base 
    'azul_suave': tab20b_hex_colors[0],         
    'ocre_claro': tab20b_hex_colors[1],         
    'verde_oliva_suave': tab20b_hex_colors[2],  
    
    # Colores de acento 
    'naranja_ambar': tab20b_hex_colors[3],     
    'terracota': tab20b_hex_colors[4],        

    # Colores de advertencia / grave 
    'rojo_profundo': tab20b_hex_colors[13],
    'purpura_oscuro': tab20b_hex_colors[6],  

    # Colores adicionales para expandir 
    'crema_brillante': tab20b_hex_colors[7], 
    'azul_verdoso': tab20b_hex_colors[8],     
    'marrón_suave': tab20b_hex_colors[9],       

    # Colores de texto y fondo 
    'fondo_claro': '#F7F7F7',                   
    'texto_oscuro': '#333333'                    
}

def compute_metrics(p, id_to_label=None):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=-1)

    metrics = {}
    if id_to_label is None:
        inferred_id_to_label = {0: 'class_0', 1: 'class_1'}
        if labels.max() >= len(inferred_id_to_label):
             # Extender si hay más de dos clases y id_to_label no se proporcionó
             max_label_id = labels.max()
             for i in range(len(inferred_id_to_label), max_label_id + 1):
                 inferred_id_to_label[i] = f'class_{i}'

        num_classes = len(inferred_id_to_label)
        current_id_to_label = inferred_id_to_label
    else:
        num_classes = len(id_to_label)
        current_id_to_label = id_to_label
    

    # 1. Métricas generales (accuracy, F1, recall, precision ponderados)
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['f1_overall'] = f1_score(labels, preds, average='weighted', zero_division=0)
    metrics['recall_overall'] = recall_score(labels, preds, average='weighted', zero_division=0)
    metrics['precision_overall'] = precision_score(labels, preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        labels, preds, output_dict=True,
        target_names=[current_id_to_label[i] for i in sorted(current_id_to_label.keys())],
        zero_division=0
    )

    # 2. Añadir métricas por clase dinámicamente desde el report_dict
    for i in sorted(current_id_to_label.keys()):
        class_name = current_id_to_label[i]
        if str(i) in report_dict: # El output_dict usa los IDs como strings '0', '1', etc.
            metrics[f'f1_class_{i}_{class_name.replace(" ", "_")}'] = report_dict[str(i)]['f1-score']
            metrics[f'recall_class_{i}_{class_name.replace(" ", "_")}'] = report_dict[str(i)]['recall']
            metrics[f'precision_class_{i}_{class_name.replace(" ", "_")}'] = report_dict[str(i)]['precision']
        else: # Si una clase no tiene soporte en los datos, sus métricas serán 0
            metrics[f'f1_class_{i}_{class_name.replace(" ", "_")}'] = 0.0
            metrics[f'recall_class_{i}_{class_name.replace(" ", "_")}'] = 0.0
            metrics[f'precision_class_{i}_{class_name.replace(" ", "_")}'] = 0.0


    # 3. Informe de clasificación completo
    metrics['classification_report_dict'] = report_dict 

    return metrics