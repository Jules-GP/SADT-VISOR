"""`segment_volume` with the network replaced by a stand-in that returns
exactly the detections a test asks for.

Everything around the network runs for real: the normalisation, the tensor
handed to it, the score threshold, the mask binarisation and the painting. No
GPU, no checkpoint.
"""

import numpy as np
import pytest
import torch

from sadt_clic import pipeline


class RecordingNetwork:
    """A stand-in for the Mask R-CNN.

    Records every image it is handed, and returns the same detections for each
    slice. A detection is `(score, label, mask)`, the mask a float array over
    the slice, as torchvision's soft masks are.
    """

    def __init__(self, detections=()):
        self.detections = list(detections)
        self.calls = []

    def __call__(self, images):
        assert len(images) == 1, "the volume is fed one slice at a time"
        tensor = images[0]
        self.calls.append(tensor.clone())
        scores = torch.tensor([d[0] for d in self.detections], dtype=torch.float32)
        labels = torch.tensor([d[1] for d in self.detections], dtype=torch.int64)
        if self.detections:
            masks = torch.stack(
                [torch.from_numpy(np.asarray(d[2], dtype=np.float32))
                 for d in self.detections]
            ).unsqueeze(1)
        else:
            masks = torch.zeros((0, 1) + tuple(tensor.shape[1:]), dtype=torch.float32)
        return [{"scores": scores, "labels": labels, "masks": masks}]


def _volume(shape=(4, 5, 3)):
    """A volume with a real spread on every slice, so nothing normalises flat."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def _mask(shape, region):
    """A hard mask over `region`, as a slice-shaped float array."""
    mask = np.zeros(shape, dtype=np.float32)
    mask[region] = 1.0
    return mask


def test_a_detection_exactly_at_the_threshold_is_kept():
    """The comparison is `>=`. A caller who set the threshold to a score it
    read off a previous run expects that detection back."""
    volume = _volume()
    network = RecordingNetwork([(0.7, 2, _mask((4, 5), (slice(0, 2), slice(0, 2))))])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 3
    assert labels[0, 0, 0] == 2


def test_a_detection_just_below_the_threshold_is_dropped():
    volume = _volume()
    network = RecordingNetwork([(0.6999, 2, _mask((4, 5), (slice(0, 2), slice(0, 2))))])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 0
    assert not labels.any()


def test_a_detection_just_above_the_threshold_is_kept():
    volume = _volume()
    network = RecordingNetwork([(0.7001, 2, _mask((4, 5), (slice(0, 2), slice(0, 2))))])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 3
    assert labels[0, 0, 0] == 2


def test_the_threshold_is_the_one_knob_that_moves_the_segmentation():
    """Same network, two thresholds, two different segmentations -- which is
    exactly why the value is an argument and is recorded in the report."""
    volume = _volume()
    network = RecordingNetwork([
        (0.9, 1, _mask((4, 5), (slice(0, 1), slice(0, 1)))),
        (0.5, 2, _mask((4, 5), (slice(2, 3), slice(2, 3)))),
    ])

    strict, strict_count = pipeline.segment_volume(network, volume, "cpu", 0.8)
    loose, loose_count = pipeline.segment_volume(network, volume, "cpu", 0.4)

    assert strict_count == 3 and loose_count == 6
    assert sorted(np.unique(strict)) == [0, 1]
    assert sorted(np.unique(loose)) == [0, 1, 2]


def test_zero_detections_give_a_real_empty_mask():
    """An empty result is an answer, not a crash: the network returns
    zero-length tensors and the indexing and the squeeze have to survive them."""
    volume = _volume()
    network = RecordingNetwork([])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 0
    assert labels.shape == volume.shape
    assert labels.dtype == np.int16
    assert not labels.any()


def test_every_detection_filtered_out_gives_a_real_empty_mask():
    """The other route to an empty result: detections exist but none clears."""
    volume = _volume()
    network = RecordingNetwork([(0.1, 3, _mask((4, 5), (slice(None), slice(None))))])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 0
    assert not labels.any()


def test_the_label_value_written_is_the_predicted_class():
    """Not 1, and not the detection's index: the class the network named."""
    volume = _volume()
    network = RecordingNetwork([(0.9, 3, _mask((4, 5), (slice(0, 2), slice(0, 2))))])

    labels, _ = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert set(np.unique(labels)) == {0, 3}


def test_overlapping_detections_are_painted_in_order():
    """Two detections covering the same voxels: the last one wins, and the
    voxels only it covers keep its label."""
    volume = _volume()
    shape = (4, 5)
    network = RecordingNetwork([
        (0.9, 1, _mask(shape, (slice(0, 3), slice(0, 3)))),
        (0.8, 2, _mask(shape, (slice(1, 4), slice(1, 4)))),
    ])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 6
    assert labels[0, 0, 0] == 1, "covered only by the first detection"
    assert labels[2, 2, 0] == 2, "covered by both; the last one painted wins"
    assert labels[3, 3, 0] == 2, "covered only by the second detection"


def test_a_detection_paints_only_its_own_slice():
    """The mask is 2D and belongs to one axial slice. Painting it into the
    volume rather than the slice would smear the canine through the scan."""
    volume = _volume(shape=(4, 5, 3))
    shape = (4, 5)

    class OneSliceOnly(RecordingNetwork):
        def __call__(self, images):
            index = len(self.calls)
            self.detections = (
                [(0.9, 1, _mask(shape, (slice(None), slice(None))))] if index == 1 else []
            )
            return super().__call__(images)

    labels, detections = pipeline.segment_volume(OneSliceOnly(), volume, "cpu", 0.7)

    assert detections == 1
    assert not labels[..., 0].any()
    assert (labels[..., 1] == 1).all()
    assert not labels[..., 2].any()


def test_the_output_dtype_is_int16():
    """A label volume, not a probability map: it is written next to the scan
    and read as a segmentation."""
    labels, _ = pipeline.segment_volume(RecordingNetwork([]), _volume(), "cpu", 0.7)

    assert labels.dtype == np.int16


def test_the_output_has_the_shape_of_the_input():
    volume = _volume(shape=(7, 6, 5))

    labels, _ = pipeline.segment_volume(RecordingNetwork([]), volume, "cpu", 0.7)

    assert labels.shape == (7, 6, 5)


def test_detections_are_counted_across_every_slice():
    """The count is what tells a caller the scan was seen at all, so it is the
    total over the volume, not the last slice's."""
    volume = _volume(shape=(4, 5, 6))
    network = RecordingNetwork([
        (0.9, 1, _mask((4, 5), (slice(0, 1), slice(0, 1)))),
        (0.9, 2, _mask((4, 5), (slice(1, 2), slice(1, 2)))),
    ])

    _, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 12


def test_the_network_is_called_once_per_slice():
    """Axial slice by axial slice, over the third axis."""
    volume = _volume(shape=(4, 5, 9))
    network = RecordingNetwork([])

    pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert len(network.calls) == 9


def test_each_slice_reaches_the_network_as_a_three_channel_image():
    """`maskrcnn_resnet50_fpn` has an ImageNet backbone and wants 3 channels;
    the grey slice is repeated into all three."""
    volume = _volume(shape=(4, 5, 2))
    network = RecordingNetwork([])

    pipeline.segment_volume(network, volume, "cpu", 0.7)

    handed = network.calls[0]
    assert tuple(handed.shape) == (3, 4, 5)
    assert torch.equal(handed[0], handed[1]) and torch.equal(handed[1], handed[2])
    assert handed.dtype == torch.float32


def test_the_slice_handed_to_the_network_is_normalised():
    """Raw Hounsfield units in, [0, 1] out, before the tensor is built."""
    volume = (np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2) * 100) - 1000
    network = RecordingNetwork([])

    pipeline.segment_volume(network, volume, "cpu", 0.7)

    handed = network.calls[0]
    assert float(handed.min()) == pytest.approx(0.0)
    assert float(handed.max()) == pytest.approx(1.0)


def test_the_slice_handed_to_the_network_is_the_right_one():
    """Off-by-one over the third axis would segment the wrong plane and paint
    the answer into a neighbour."""
    volume = _volume(shape=(4, 5, 3))
    network = RecordingNetwork([])

    pipeline.segment_volume(network, volume, "cpu", 0.7)

    for index, handed in enumerate(network.calls):
        expected = pipeline.normalise(volume[..., index])
        assert np.allclose(handed[0].numpy(), expected)


def test_a_mask_pixel_exactly_at_the_binary_threshold_is_outside():
    """The soft mask is binarised with `> 0.5`, so a pixel the network is
    exactly undecided about is not painted."""
    volume = _volume()
    soft = np.full((4, 5), pipeline.MASK_BINARY_THRESHOLD, dtype=np.float32)
    soft[0, 0] = pipeline.MASK_BINARY_THRESHOLD + 0.01
    network = RecordingNetwork([(0.9, 1, soft)])

    labels, _ = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert labels[0, 0, 0] == 1
    assert labels[1, 1, 0] == 0
    assert int((labels[..., 0] == 1).sum()) == 1


def test_a_detection_whose_mask_is_all_background_is_counted_but_paints_nothing():
    """`detections` counts what cleared the SCORE threshold; `labels_present`
    in the report says what was actually painted. The two can legitimately
    disagree, and the report keeps both for that reason."""
    volume = _volume()
    network = RecordingNetwork([(0.99, 1, np.zeros((4, 5), dtype=np.float32))])

    labels, detections = pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert detections == 3
    assert not labels.any()


def test_the_volume_is_not_modified():
    """The scan is the caller's data and is written back out unchanged by the
    tools downstream of this one."""
    volume = _volume()
    original = volume.copy()
    network = RecordingNetwork([(0.9, 1, _mask((4, 5), (slice(None), slice(None))))])

    pipeline.segment_volume(network, volume, "cpu", 0.7)

    assert np.array_equal(volume, original)


def test_the_network_runs_with_gradients_off():
    """A Mask R-CNN over hundreds of slices builds a graph per slice; keeping
    them is how a long batch runs the machine out of memory."""
    volume = _volume(shape=(4, 5, 2))
    seen = []

    class GradientWatcher(RecordingNetwork):
        def __call__(self, images):
            seen.append(torch.is_grad_enabled())
            return super().__call__(images)

    pipeline.segment_volume(GradientWatcher([]), volume, "cpu", 0.7)

    assert seen == [False, False]
