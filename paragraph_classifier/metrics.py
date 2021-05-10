from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss


def y_pred(y_score):
    return [[1 if score >= 0.5 else 0 for score in ys] for ys in y_score]

def get_metrics_dict(labels_list):
    metrics_dict = {}
    # Add recall for each class
    metrics_dict['recall_dialog'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[0], average = None, zero_division='warn')
    metrics_dict['recall_story'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[1], average = None, zero_division='warn')
    metrics_dict['recall_descriptionOfPlace'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[2], average = None, zero_division='warn')
    metrics_dict['recall_descriptionOfAppearance'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[3], average = None, zero_division='warn')
    metrics_dict['recall_descriptionOfAction'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[4], average = None, zero_division='warn')
    metrics_dict['recall_descriptionOfObject'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[5], average = None, zero_division='warn')
    metrics_dict['recall_descriptionOfSound'] = lambda yt, ys: recall_score(yt, y_pred(ys), labels=[6], average = None, zero_division='warn')
    
    # Add macro, micro and weighted average recall
    averages = ['macro', 'micro', 'weighted']
    for avg in averages:
        metrics_dict['recall_{}'.format(avg)] = lambda yt, ys: recall_score(yt, y_pred(ys), average = avg, zero_division='warn')

    # Add precision for each class
    metrics_dict['precision_dialog'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[0], average = None, zero_division='warn')
    metrics_dict['precision_story'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[1], average = None, zero_division='warn')
    metrics_dict['precision_descriptionOfPlace'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[2], average = None, zero_division='warn')
    metrics_dict['precision_descriptionOfAppearance'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[3], average = None, zero_division='warn')
    metrics_dict['precision_descriptionOfAction'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[4], average = None, zero_division='warn')
    metrics_dict['precision_descriptionOfObject'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[5], average = None, zero_division='warn')
    metrics_dict['precision_descriptionOfSound'] = lambda yt, ys: precision_score(yt, y_pred(ys), labels=[6], average = None, zero_division='warn')

    # Add macro, micro and weighted average precision
    averages = ['macro', 'micro', 'weighted']
    for avg in averages:
        metrics_dict['precision_{}'.format(avg)] = lambda yt, ys: precision_score(yt, y_pred(ys), average = avg, zero_division='warn')

    # Add average precision score (area under the precision-recall curve)
    metrics_dict['avg_precision_score'] = lambda yt, ys: average_precision_score(yt, ys)

    # Add LRAP
    metrics_dict['LRAP'] = label_ranking_average_precision_score

    # Add label_ranking_loss
    metrics_dict['label_ranking_loss'] = label_ranking_loss
    
    return metrics_dict
