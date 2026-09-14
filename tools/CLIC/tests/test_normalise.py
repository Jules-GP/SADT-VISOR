"""`normalise`: one slice to [0, 1], including the slices that have no spread.

A CBCT batch reliably contains slices of pure air above the head and pure
padding below it. Those are constant, and the naive rescaling divides by zero
and hands the network a plane of NaNs -- which torchvision does not refuse, it
just returns detections from noise.
"""

import numpy as np
import pytest

from sadt_clic import pipeline


def test_a_constant_slice_is_zeros_not_nans():
    """The air above the head, in every scan."""
    flat = np.full((6, 6), 0.0, dtype=np.float32)

    result = pipeline.normalise(flat)

    assert not np.isnan(result).any()
    assert (result == 0).all()


def test_a_constant_negative_slice_is_zeros_not_nans():
    """Air is about -1000 HU, so the constant slice is usually negative, not
    zero: a guard written as `if not slice.any()` would miss it."""
    flat = np.full((6, 6), -1000.0, dtype=np.float32)

    result = pipeline.normalise(flat)

    assert not np.isnan(result).any()
    assert (result == 0).all()


def test_a_spread_below_the_epsilon_is_zeros():
    """Two voxels a billionth apart are one flat slice, and dividing by that
    difference amplifies quantisation noise to full scale."""
    almost_flat = np.full((4, 4), 5.0, dtype=np.float32)
    almost_flat[0, 0] = 5.0 + 1e-9

    result = pipeline.normalise(almost_flat)

    assert (result == 0).all()


def test_a_single_voxel_slice_is_zeros():
    """A 1x1 slice has no spread by construction."""
    result = pipeline.normalise(np.array([[42.0]], dtype=np.float32))

    assert result.shape == (1, 1)
    assert not np.isnan(result).any()
    assert result[0, 0] == 0


def test_negative_values_are_mapped_into_the_unit_range():
    """Hounsfield units run from about -1000 to +3000; the network wants
    [0, 1]."""
    slice_2d = np.array([[-1000.0, 0.0], [1000.0, 3000.0]], dtype=np.float32)

    result = pipeline.normalise(slice_2d)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result[0, 1] == pytest.approx(0.25)


def test_a_slice_already_in_the_unit_range_is_still_rescaled():
    """Normalising is per slice and unconditional: the darkest voxel of the
    slice becomes 0 and the brightest 1, whatever the input range was."""
    slice_2d = np.array([[0.2, 0.5], [0.8, 0.8]], dtype=np.float32)

    result = pipeline.normalise(slice_2d)

    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.5)


def test_the_output_range_is_exactly_what_the_model_expects():
    """torchvision's detector normalises its input again with ImageNet
    statistics, which assume [0, 1]. Nothing may leave the range."""
    rng = np.random.default_rng(3)
    slice_2d = (rng.random((32, 32)) * 4000 - 1000).astype(np.float32)

    result = pipeline.normalise(slice_2d)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert ((result >= 0) & (result <= 1)).all()


def test_a_float32_slice_stays_float32():
    """The volume is read as float32 and there is one of these per slice; an
    upcast to float64 doubles the array and the tensor built from it."""
    slice_2d = np.arange(16, dtype=np.float32).reshape(4, 4)

    assert pipeline.normalise(slice_2d).dtype == np.float32


def test_a_constant_float32_slice_also_stays_float32():
    """The two branches must not disagree about dtype, or the memory cost of a
    volume depends on how flat its slices are."""
    flat = np.full((4, 4), 7.0, dtype=np.float32)

    assert pipeline.normalise(flat).dtype == np.float32


def test_an_integer_slice_is_not_integer_divided_to_zeros():
    """A scan stored as int16 reaches this unscaled if a caller passes the raw
    array; integer division would flatten every voxel but the brightest to 0."""
    slice_2d = np.array([[0, 100], [200, 400]], dtype=np.int16)

    result = pipeline.normalise(slice_2d)

    assert np.issubdtype(result.dtype, np.floating)
    assert result[0, 1] == pytest.approx(0.25)
    assert result.max() == pytest.approx(1.0)


def test_the_input_is_not_modified_in_place():
    """`segment_volume` passes a view into the caller's volume, so an in-place
    rescale would corrupt the scan one slice at a time."""
    slice_2d = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float32)
    original = slice_2d.copy()

    pipeline.normalise(slice_2d)

    assert np.array_equal(slice_2d, original)


def test_the_shape_is_preserved():
    """The mask painted back is indexed by the same coordinates."""
    slice_2d = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert pipeline.normalise(slice_2d).shape == (3, 4)
