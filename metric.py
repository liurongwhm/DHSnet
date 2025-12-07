import numpy as np

def calculate_metrics(prediction_matrix, gt):
    from sklearn.metrics import cohen_kappa_score
    if prediction_matrix.shape != gt.shape:
        raise ValueError('The shapes of prediction_matrix and gt are not the same.')
    N = int(np.max([prediction_matrix.max(), gt.max()]))
    Acc_list = []
    for i in range(1, N + 1):
        Acc_i = ((prediction_matrix == i) & (gt == i)).sum() / (gt == i).sum()
        Acc_list.append(Acc_i)

    OA = ((prediction_matrix == gt) & (gt != 0)).sum() / (gt != 0).sum()
    AA = np.mean(Acc_list)
    KAPPA = cohen_kappa_score((gt[gt != 0]).flatten(), prediction_matrix[gt != 0].flatten())

    result = np.array(Acc_list + [OA, AA, KAPPA])
    return result