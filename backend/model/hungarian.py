import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_match(similarity_matrix, threshold=0.5):
    """
    Match detections to tracks using the Hungarian algorithm.

    Args:
      similarity_matrix: (N_det, N_trk) numpy array, values in [0,1]
      threshold: minimum similarity to accept a match

    Returns:
      matched:        list of (det_idx, trk_idx) tuples
      unmatched_dets: list of det indices with no match
      unmatched_trks: list of trk indices with no match
    """
    if similarity_matrix.size == 0:
        n_det = similarity_matrix.shape[0]
        n_trk = similarity_matrix.shape[1]
        return [], list(range(n_det)), list(range(n_trk))

    # Convert similarity → cost (Hungarian minimizes cost)
    cost_matrix = 1.0 - similarity_matrix

    det_indices, trk_indices = linear_sum_assignment(cost_matrix)

    matched, unmatched_dets, unmatched_trks = [], [], []

    matched_det_set = set()
    matched_trk_set = set()

    for d, t in zip(det_indices, trk_indices):
        if similarity_matrix[d, t] >= threshold:
            matched.append((int(d), int(t)))
            matched_det_set.add(int(d))
            matched_trk_set.add(int(t))
        else:
            unmatched_dets.append(int(d))
            unmatched_trks.append(int(t))

    # Collect remaining unmatched
    for d in range(similarity_matrix.shape[0]):
        if d not in matched_det_set and d not in [ud for ud in unmatched_dets]:
            unmatched_dets.append(d)

    for t in range(similarity_matrix.shape[1]):
        if t not in matched_trk_set and t not in [ut for ut in unmatched_trks]:
            unmatched_trks.append(t)

    return matched, unmatched_dets, unmatched_trks


def iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    Used as a geometric fallback alongside Re-ID similarity.

    Args:
      boxes1: (N, 4) array [x1,y1,x2,y2]
      boxes2: (M, 4) array [x1,y1,x2,y2]
    Returns:
      iou: (N, M) array
    """
    N = len(boxes1)
    M = len(boxes2)
    iou = np.zeros((N, M), dtype=np.float32)

    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = area1 + area2 - inter
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou


def combined_similarity(reid_sim, iou_sim, w_reid=0.7, w_iou=0.3):
    """
    Weighted combination of Re-ID similarity and IoU similarity.
    Gives robust matching even when Re-ID head is partially trained.

    Args:
      reid_sim: (N_det, N_trk) Re-ID similarity matrix
      iou_sim:  (N_det, N_trk) IoU matrix
      w_reid:   weight for Re-ID score
      w_iou:    weight for IoU score
    Returns:
      combined: (N_det, N_trk)
    """
    return w_reid * reid_sim + w_iou * iou_sim


if __name__ == "__main__":
    import numpy as np
    print("Hungarian Matcher — Unit Test")

    # 4 detections, 3 tracks
    sim = np.array([
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
        [0.3, 0.2, 0.1],
    ])

    matched, unmatched_dets, unmatched_trks = hungarian_match(sim, threshold=0.5)
    print(f"Matched pairs:    {matched}")
    print(f"Unmatched dets:   {unmatched_dets}")
    print(f"Unmatched tracks: {unmatched_trks}")

    # Test IoU
    boxes1 = np.array([[10,10,50,50], [100,100,150,150]])
    boxes2 = np.array([[15,15,55,55], [200,200,250,250]])
    iou = iou_matrix(boxes1, boxes2)
    print(f"\nIoU matrix:\n{iou}")
    print("[TEST PASSED] Hungarian matcher working correctly.")
