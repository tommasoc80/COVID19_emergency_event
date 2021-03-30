from covid19_exceptius.types import * 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


def _round(x: float) -> float:
    return round(x, 3)


def get_metrics(preds: List[List[int]], labels: List[List[int]]) -> Dict[str, Any]:
    per_column_preds = list(zip(*preds))
    per_column_labels = list(zip(*labels))
    per_column_metrics = [(f1_score(pcp, pcl, labels=[1, 0], average='weighted'),
                           recall_score(pcp, pcl, labels=[1, 0], average='weighted'),
                           precision_score(pcp, pcl, labels=[1, 0], average='weighted'))
                          for pcp, pcl in zip(per_column_preds, per_column_labels)]
    f1s, ps, rs = list(zip(*per_column_metrics))
    return {'accuracy': _round(accuracy_score(array(preds), array(labels))),
            'hamming': _round(hamming_loss(array(preds), array(labels))),
            'mean_f1': _round(sum(f1s)/len(f1s)),
            'mean_precision': _round(sum(ps)/len(ps)),
            'mean_recall': _round(sum(rs)/len(rs)),
            'column_wise':
                [{'f1': _round(f1), 'precision': _round(p), 'recall': _round(r)} for f1, p, r in per_column_metrics]
            }
