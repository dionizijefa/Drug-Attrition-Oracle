import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, average_precision_score, roc_auc_score, \
    f1_score, auc

def table_metrics(predictions, withdrawn_col):
    ap_wd = average_precision_score(predictions[withdrawn_col], predictions['probabilities'])
    ap_ad = average_precision_score(predictions[withdrawn_col], predictions['probabilities'], pos_label=0)
    auroc_wd = roc_auc_score(predictions[withdrawn_col], predictions['probabilities'])

    optimal_f1_score = []
    optimal_threshold = []
    for threshold in np.arange(1.0, 0, -0.01):
        predictions_df = predictions.copy()
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
        optimal_f1_score.append(
            f1_score(
                predictions_df[withdrawn_col], predictions_df['predicted_class'], average='binary'
            )
        )
        optimal_threshold.append(threshold)

    optimal_f1_index = np.argmax(np.array(optimal_f1_score))
    optimal_threshold = optimal_threshold[optimal_f1_index]

    # calculate threshold dependent metrics @ optimal weighted f1-score
    predictions_df = predictions.copy()
    predictions_df['predicted_class'] = 0
    predictions_df.loc[predictions_df['probabilities'] >= optimal_threshold, 'predicted_class'] = 1

    precision_wd = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'])
    precision_ad = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                   pos_label=0)
    recall_wd = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'])
    recall_ad = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                   pos_label=0)

    tn, fp, fn, tp = confusion_matrix(predictions_df[withdrawn_col], predictions_df['predicted_class']).ravel()
    results_df = pd.DataFrame(
        {
            'Threshold at opt. F1-score (withdrawn)': optimal_threshold,
            'Opt. F1-score (withdrawn only)': max(optimal_f1_score),
            'AP withdrawn': ap_wd,
            'AP approved': ap_ad,
            'AUROC withdrawn': auroc_wd,
            'Precision withdrawn': precision_wd,
            'Recall withdrawn': recall_wd,
            'Precision approved': precision_ad,
            'Recall approved': recall_ad,
            'True positives': tp,
            'True negatives': tn,
            'False positives': fp,
            'False negatives':  fn,
         },
        index=[0]
    )

    return results_df


