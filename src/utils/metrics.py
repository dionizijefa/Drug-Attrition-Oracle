import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, average_precision_score, roc_auc_score, \
    f1_score, auc


def table_metrics_significance(predictions, withdrawn_col):
    ap_wd_at_significance = []
    ap_ad_at_significance = []
    auroc_wd_at_significance = []
    precision_wd_at_significance = []
    recall_wd_at_significance = []
    precision_ad_at_significance = []
    recall_ad_at_significance = []
    tp_at_significance = []
    tn_at_significance = []
    fp_at_significance = []
    fn_at_significance = []

    optimal_f1_score = []
    optimal_threshold = []
    for threshold in np.arange(1.0, 0, -0.01):
        predictions_df = predictions
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
        optimal_f1_score.append(
            f1_score(
                predictions_df[withdrawn_col], predictions_df['predicted_class'],
                average='weighted',
            )
        )
        optimal_threshold.append(threshold)

    optimal_f1_index = np.argmax(np.array(optimal_f1_score))
    optimal_threshold = optimal_threshold[optimal_f1_index]

    for significance in np.arange(0, 1.0, 0.05):
        # calculate threshold dependent metrics @ optimal weighted f1-score
        predictions_df = predictions
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] >= optimal_threshold, 'predicted_class'] = 1
        withdrawns = predictions_df.loc[(predictions_df['predicted_class'] == 1) &
                                        (predictions_df['p_withdrawn'] > significance)]
        approveds = predictions_df.loc[(predictions_df['predicted_class'] == 0) &
                                       (predictions_df['p_approved'] > significance)]
        predictions_df = pd.concat([withdrawns, approveds])

        tn, fp, fn, tp = confusion_matrix(predictions_df[withdrawn_col], predictions_df['predicted_class']).ravel()
        tp_at_significance.append(tp)
        tn_at_significance.append(tn)
        fp_at_significance.append(fp)
        fn_at_significance.append(fn)

        precision_wd_at_significance.append(
            precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=1)
        )
        precision_ad_at_significance.append(
            precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=0)
        )
        recall_wd_at_significance.append(
            recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=1)
        )
        recall_ad_at_significance.append(
            recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=0)
        )

    for significance in np.arange(0, 1, 0.05):
        recall_wd = []
        recall_ad = []
        precision_wd = []
        precision_ad = []
        for threshold in np.arange(1, 0, -0.01):
            predictions_df = predictions
            predictions_df['predicted_class'] = 0
            predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
            withdrawns = predictions_df.loc[(predictions_df['predicted_class'] == 1) &
                                            (predictions_df['p_withdrawn'] > significance)]
            approveds = predictions_df.loc[(predictions_df['predicted_class'] == 0) &
                                           (predictions_df['p_approved'] > significance)]
            predictions_df = pd.concat([withdrawns, approveds])
            rec_wd = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=1,
                                  zero_division=0)
            rec_ad = recall_score (predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=0,
                                   zero_division=0)
            prec_wd = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=1,
                                      zero_division=0)
            prec_ad = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'], pos_label=0,
                                      zero_division=0)

            #if rec_wd == 0 or rec_ad == 0 or prec_wd == 0 or prec_ad == 0:
                #continue
            #else:
            recall_wd.append(rec_wd)
            recall_ad.append(rec_ad)
            precision_wd.append(prec_wd)
            precision_ad.append(prec_ad)

        # 1 - specificity (specificty = recall of the negative class)
        auroc_wd_at_significance.append(auc((1-np.array(recall_ad)), np.array(recall_wd)))

        ap_wd = []
        ap_ad = []
        for i in range(len(recall_wd)):
            if i == 0:
                ap_wd.append((1 - recall_wd[i]) * precision_wd[i])
                ap_ad.append((1 - recall_ad[i]) * precision_ad[i])
            else:
                ap_wd.append((recall_wd[i] - recall_wd[i-1]) * precision_wd[i])
                ap_ad.append((recall_ad[i] - recall_ad[i-1]) * precision_ad[i])

        ap_wd_at_significance.append(sum(ap_wd))
        ap_ad_at_significance.append(sum(ap_ad))

    results_df = pd.DataFrame(
        {
            'Significance': np.arange(0, 1.0, 0.05),
            'AP withdrawn': ap_wd_at_significance,
            'AP approved': ap_ad_at_significance,
            'AUROC withdrawn': auroc_wd_at_significance,
            'Precision_withdrawn': precision_wd_at_significance,
            'Recall withdrawn': recall_wd_at_significance,
            'Precision approved': precision_ad_at_significance,
            'Recall approved': recall_ad_at_significance,
            'True positives': tp_at_significance,
            'True negatives': tn_at_significance,
            'False positives': fp_at_significance,
            'False negatives':  fn_at_significance,
         }
    )

    return results_df


def table_metrics(predictions, withdrawn_col):


    ap_wd = average_precision_score(predictions[withdrawn_col], predictions['probabilities'])
    ap_ad = average_precision_score(predictions[withdrawn_col], predictions['probabilities'], pos_label=0)
    auroc_wd = roc_auc_score(predictions[withdrawn_col], predictions['probabilities'])

    optimal_f1_score = []
    optimal_threshold = []
    for threshold in np.arange(1.0, 0, -0.01):
        predictions_df = predictions
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
        optimal_f1_score.append(
            f1_score(
                predictions_df[withdrawn_col], predictions_df['predicted_class'],
                average='weighted',
            )
        )
        optimal_threshold.append(threshold)

    optimal_f1_index = np.argmax(np.array(optimal_f1_score))
    optimal_threshold = optimal_threshold[optimal_f1_index]
    print(optimal_threshold)

    # calculate threshold dependent metrics @ optimal weighted f1-score
    predictions_df = predictions
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
            'AP withdrawn': ap_wd,
            'AP approved': ap_ad,
            'AUROC withdrawn': auroc_wd,
            'Precision_withdrawn': precision_wd,
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

def table_metrics_threshold(predictions, withdrawn_col, threshold):

    ap_wd = average_precision_score(predictions[withdrawn_col], predictions['probabilities'])
    ap_ad = average_precision_score(predictions[withdrawn_col], predictions['probabilities'], pos_label=0)
    auroc_wd = roc_auc_score(predictions[withdrawn_col], predictions['probabilities'])


    # calculate threshold dependent metrics @ optimal weighted f1-score
    predictions_df = predictions
    predictions_df['predicted_class'] = 0
    predictions_df.loc[predictions_df['probabilities'] >= threshold, 'predicted_class'] = 1

    precision_wd = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'])
    precision_ad = precision_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                   pos_label=0)
    recall_wd = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'])
    recall_ad = recall_score(predictions_df[withdrawn_col], predictions_df['predicted_class'],
                                   pos_label=0)

    tn, fp, fn, tp = confusion_matrix(predictions_df[withdrawn_col], predictions_df['predicted_class']).ravel()
    results_df = pd.DataFrame(
        {
            'AP withdrawn': ap_wd,
            'AP approved': ap_ad,
            'AUROC withdrawn': auroc_wd,
            'Precision_withdrawn': precision_wd,
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

