import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, average_precision_score, roc_auc_score, \
    f1_score
from torch import cat


def optimal_threshold_f1(model, loader, descriptors=False):
    model.eval()
    optimal_f1_score = []
    optimal_threshold = []

    probabilities = []
    targets = []

    if descriptors==True:
        for i in loader:
            probabilities.append(model.forward(i.x, i.edge_index, i.batch, i.descriptors))
            targets.append(i.y)
    else:
        for i in loader:
            probabilities.append(model.forward(i.x, i.edge_index, i.batch))
            targets.append(i.y)
    probabilities = np.array(cat(probabilities).detach().cpu().numpy().flatten())
    targets = np.array(cat(targets).detach().cpu().numpy().flatten())
    probabilities = 1 / (1 + np.exp(-probabilities))

    predictions = pd.DataFrame({'class': targets, 'probabilities': probabilities})

    for threshold in np.arange(1.0, 0, -0.01):
        predictions_df = predictions.copy()
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
        optimal_f1_score.append(
            f1_score(
                predictions_df['class'], predictions_df['predicted_class'], average='binary'
            )
        )
        optimal_threshold.append(threshold)

    optimal_f1_index = np.argmax(np.array(optimal_f1_score))
    optimal_threshold = optimal_threshold[optimal_f1_index]
    return optimal_threshold


def table_metrics(predictions, withdrawn_col, optimal_threshold):
    ap_wd = average_precision_score(predictions[withdrawn_col], predictions['probabilities'])
    ap_ad = average_precision_score(predictions[withdrawn_col], predictions['probabilities'], pos_label=0)
    auroc_wd = roc_auc_score(predictions[withdrawn_col], predictions['probabilities'])

    """
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
    """

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

    optimal_f1_score = f1_score(
                predictions_df[withdrawn_col], predictions_df['predicted_class'], average='binary'
            )

    tn, fp, fn, tp = confusion_matrix(predictions_df[withdrawn_col], predictions_df['predicted_class']).ravel()
    specificity = tn / (tn+fp)
    balanced_accuracy = 0.5*(
        recall_wd + specificity
    )
    results_df = pd.DataFrame(
        {
            'Threshold at opt. F1-score (withdrawn)': optimal_threshold,
            'Opt. F1-score (withdrawn only)': optimal_f1_score,
            'AP withdrawn': ap_wd,
            'AP approved': ap_ad,
            'AUROC withdrawn': auroc_wd,
            'Balanced accuracy': balanced_accuracy,
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

def metrics_at_significance(predictions, withdrawn_col, optimal_threshold):

    n_examples_at_sig = []
    ap_wd_at_sig = []
    ap_ad_at_sig = []
    auroc_wd_at_sig = []
    precision_wd_at_sig = []
    precision_ad_at_sig = []
    recall_wd_at_sig = []
    recall_ad_at_sig = []
    f1_at_sig = []
    tn_at_sig = []
    fp_at_sig = []
    fn_at_sig = []
    tp_at_sig = []
    balanced_accuracy_at_sig = []
    for significance in np.arange(0, 0.85, 0.05):
        """ We look at predicitions for which it is possible to predict the withdrawn class"""
        predictions_df = predictions.copy()
        predictions_df = predictions_df.loc[predictions_df['p_withdrawn'] > significance]
        n_examples_at_sig.append(len(predictions_df))
        ap_wd_at_sig.append(average_precision_score(predictions_df[withdrawn_col], predictions_df['probabilities']))
        ap_ad_at_sig.append(average_precision_score(predictions_df[withdrawn_col], predictions_df['probabilities'],
                                                    pos_label=0))
        auroc_wd_at_sig.append(roc_auc_score(predictions_df[withdrawn_col], predictions_df['probabilities']))


        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] >= optimal_threshold, 'predicted_class'] = 1

        precision_wd_at_sig.append(precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class']))
        precision_ad_at_sig.append(precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                       pos_label=0))
        recall_wd_at_sig.append(recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class']))
        recall_ad_at_sig.append(recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                       pos_label=0))

        f1_at_sig.append(f1_score(
                    predictions_df[withdrawn_col], predictions_df['predicted_class'], average='binary'
                ))

        tn, fp, fn, tp = confusion_matrix(predictions_df[withdrawn_col], predictions_df['predicted_class']).ravel()
        tn_at_sig.append(tn)
        fp_at_sig.append(fp)
        fn_at_sig.append(fn)
        tp_at_sig.append(tp)
        recall_wd = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'])
        specificity = tn / (tn + fp)
        balanced_accuracy_at_sig.append(0.5 * (
                recall_wd + specificity
        ))

    results_df = pd.DataFrame(
        {
            'Significance': 1-(np.arange(0, 0.85, 0.05)),
            'Num. samples @ signif': n_examples_at_sig,
            'F1 (withdrawn) @ signif': f1_at_sig,
            'AP withdrawn @ signif': ap_wd_at_sig,
            'AP approved @ signif': ap_ad_at_sig,
            'AUROC withdrawn @ signif': auroc_wd_at_sig,
            'Balanced accuracy @ signif': balanced_accuracy_at_sig,
            'Precision withdrawn @ signif': precision_wd_at_sig,
            'Recall withdrawn @ signif': recall_wd_at_sig,
            'Precision approved @ signif': precision_ad_at_sig,
            'Recall approved @ signif': recall_ad_at_sig,
            'True positives @ signif': tp_at_sig,
            'True negatives @ signif': tn_at_sig,
            'False positives @ signif': fp_at_sig,
            'False negatives @ signif':  fn_at_sig,
         }
    )

    return results_df



