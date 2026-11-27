import torch

from yolo_kalman.detector import postprocess_e2e


def test_e2e_drops_duplicate_boxes_of_one_object():
    """The NMS-free head emits one object twice (second box or second class): keep the better one."""
    raw = torch.tensor([[
        [100, 100, 130, 170, 0.90, 0],   # person
        [300, 100, 400, 200, 0.85, 2],   # car
        [101, 101, 130, 171, 0.74, 0],   # same person again (IoU 0.94)
        [300, 100, 400, 200, 0.60, 7],   # same car labelled truck
        [120, 100, 150, 170, 0.50, 0],   # second person, half hidden by the first (IoU 0.2): keep
        [0, 0, 0, 0, 0.00, 0],           # padding
    ]])
    d = postprocess_e2e(raw, conf=0.25, max_det=6, classes=None)
    assert d.valid[0].tolist() == [True, True, False, False, True, False]
    assert d.scores[0, 2] == 0 and d.scores[0, 3] == 0

    # a duplicate of a box that the class filter removed is not a duplicate of a kept box
    d = postprocess_e2e(raw, conf=0.25, max_det=6, classes=torch.tensor([0, 7]))
    assert d.valid[0].tolist() == [True, False, False, True, True, False]
