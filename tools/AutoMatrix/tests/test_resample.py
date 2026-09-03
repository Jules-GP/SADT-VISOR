"""Resampling: the interpolation rule, and the geometry that must survive it.

Nearest neighbour for a segmentation, linear otherwise. Interpolating a label
map linearly produces labels that were never in it, and a label is an
anatomical structure -- 2 between 1 and 3 is a different bone, not a rounding
error.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_automatrix import pipeline


# A direction that is not the identity, so a test asserting geometry survives a
# resampling can tell "preserved" from "defaulted": ITK hands out the identity
# for an image that never had a direction set.
ROTATED_DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _shift(amount):
    return sitk.TranslationTransform(3, (amount, amount, amount))


# ---------------------------------------------------------------------------
# The interpolation rule
# ---------------------------------------------------------------------------

def test_nearest_neighbour_keeps_the_label_set_it_was_given(label_image):
    labels = pipeline.resample(label_image(), _shift(0.3), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}


def test_linear_interpolation_invents_the_label_between_them(label_image):
    """The other half of the same statement: if this stopped blending, the test
    above would pass against a broken tool."""
    scan = pipeline.resample(label_image(), _shift(0.3), is_segmentation=False)

    assert 2 in set(np.unique(sitk.GetArrayFromImage(scan)))


def test_a_half_voxel_shift_could_not_have_shown_that(label_image):
    """Why the offset is 0.3 and not 0.5, kept as an assertion so nobody
    "tidies" it: without a reference the output origin moves by the translation
    too, so a 0.5 shift samples exactly one voxel away, on the grid, and the
    two interpolators return the very same array."""
    image = label_image()

    as_labels = pipeline.resample(image, _shift(0.5), is_segmentation=True)
    as_scan = pipeline.resample(image, _shift(0.5), is_segmentation=False)

    assert np.array_equal(sitk.GetArrayFromImage(as_labels),
                          sitk.GetArrayFromImage(as_scan))


@pytest.mark.parametrize("offset", [0.1, 0.25, 0.3, 0.75, 1.4])
def test_no_off_grid_shift_makes_nearest_neighbour_invent_a_label(label_image, offset):
    labels = pipeline.resample(label_image(), _shift(offset), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}


def test_the_rule_holds_on_a_reference_grid_too(label_image):
    """A reference changes where the samples fall, not how they are taken."""
    reference = sitk.Image(10, 10, 10, sitk.sitkInt16)
    reference.SetSpacing((0.7, 0.7, 0.7))

    labels = pipeline.resample(label_image(), _shift(0.3), reference, True)
    scan = pipeline.resample(label_image(), _shift(0.3), reference, False)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}
    assert 2 in set(np.unique(sitk.GetArrayFromImage(scan)))


def test_the_identity_transform_returns_the_image_unchanged(label_image):
    image = label_image()

    resampled = pipeline.resample(image, sitk.Transform(3, sitk.sitkIdentity))

    assert np.array_equal(sitk.GetArrayFromImage(resampled),
                          sitk.GetArrayFromImage(image))


def test_what_falls_outside_the_input_is_zero(label_image):
    """The default pixel value, and it must be 0: a segmentation's background
    is 0 and a scan resampled onto a value of its own would gain a slab of
    fabricated tissue at its edge."""
    far = pipeline.resample(label_image(), _shift(50.0), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(far))) == {0}


# ---------------------------------------------------------------------------
# Geometry, with no reference: the input's own grid
# ---------------------------------------------------------------------------

def test_the_grid_is_the_input_s_own(label_image):
    image = label_image(spacing=(0.5, 0.75, 1.25))

    resampled = pipeline.resample(image, _shift(1.0))

    assert resampled.GetSize() == image.GetSize()
    assert resampled.GetSpacing() == (0.5, 0.75, 1.25)


def test_only_the_origin_moves_and_it_moves_by_the_transform(label_image):
    """Exactly: origin (10, 20, 30) plus the translation (1, 2, 3)."""
    image = label_image(origin=(10.0, 20.0, 30.0))

    resampled = pipeline.resample(image, sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))

    assert resampled.GetOrigin() == (11.0, 22.0, 33.0)


def test_a_direction_that_is_not_the_identity_survives(label_image):
    """Asserted against a rotated direction rather than the identity, because
    an image that never had one set reports the identity and a lost direction
    would look like a preserved one."""
    image = label_image()
    image.SetDirection(ROTATED_DIRECTION)

    resampled = pipeline.resample(image, _shift(1.0))

    assert resampled.GetDirection() == ROTATED_DIRECTION


@pytest.mark.parametrize("pixel_type", [
    sitk.sitkInt16, sitk.sitkUInt8, sitk.sitkFloat32, sitk.sitkFloat64,
    sitk.sitkUInt16, sitk.sitkInt32,
])
def test_the_output_keeps_the_input_s_pixel_type(label_image, pixel_type):
    """A segmentation that came in as uint8 must not leave as float: whoever
    reads it back is reading label values."""
    image = sitk.Cast(label_image(), pixel_type)

    assert pipeline.resample(image, _shift(0.3)).GetPixelID() == pixel_type


# ---------------------------------------------------------------------------
# Geometry, with a reference: the reference's grid
# ---------------------------------------------------------------------------

def test_a_reference_decides_size_spacing_origin_and_direction(label_image):
    reference = sitk.Image(3, 4, 5, sitk.sitkUInt8)
    reference.SetSpacing((2.0, 2.0, 2.0))
    reference.SetOrigin((-1.0, -2.0, -3.0))
    reference.SetDirection(ROTATED_DIRECTION)

    resampled = pipeline.resample(label_image(), _shift(1.0), reference)

    assert resampled.GetSize() == (3, 4, 5)
    assert resampled.GetSpacing() == (2.0, 2.0, 2.0)
    assert resampled.GetOrigin() == (-1.0, -2.0, -3.0)
    assert resampled.GetDirection() == ROTATED_DIRECTION


def test_a_reference_does_not_decide_the_pixel_type(label_image):
    """The reference gives the grid, not the storage: a uint8 reference must
    not truncate an int16 label map, nor a float32 scan."""
    reference = sitk.Image(4, 4, 4, sitk.sitkUInt8)

    resampled = pipeline.resample(label_image(), _shift(1.0), reference)

    assert resampled.GetPixelID() == sitk.sitkInt16


def test_the_reference_image_itself_is_not_modified(label_image):
    reference = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    reference.SetSpacing((2.0, 2.0, 2.0))

    pipeline.resample(label_image(), _shift(1.0), reference)

    assert reference.GetSpacing() == (2.0, 2.0, 2.0)
    assert set(np.unique(sitk.GetArrayFromImage(reference))) == {0}


def test_a_reference_on_the_same_grid_changes_nothing_but_the_sampling(label_image):
    """A reference matching the input pins the output where the input was --
    unlike the no-reference case, where the origin follows the transform. The
    same transform therefore lands the content in two different places, which
    is the whole reason the argument exists."""
    image = label_image()
    with_reference = pipeline.resample(image, _shift(1.0), image, True)
    without = pipeline.resample(image, _shift(1.0), None, True)

    assert with_reference.GetOrigin() == (0.0, 0.0, 0.0)
    assert without.GetOrigin() == (1.0, 1.0, 1.0)
    assert not np.array_equal(sitk.GetArrayFromImage(with_reference),
                              sitk.GetArrayFromImage(without))


def test_the_content_moves_by_a_whole_voxel_when_the_shift_is_one(label_image):
    """The arithmetic end to end, on a reference grid so the origin stays put:
    a one-voxel translation moves the labels one voxel, in the direction ITK's
    resampler defines -- it maps OUTPUT points through the transform to find
    where to read, so the content moves the other way."""
    image = label_image()
    before = sitk.GetArrayFromImage(image)

    along_x = sitk.GetArrayFromImage(
        pipeline.resample(image, sitk.TranslationTransform(3, (1.0, 0.0, 0.0)),
                          image, True))
    along_z = sitk.GetArrayFromImage(
        pipeline.resample(image, sitk.TranslationTransform(3, (0.0, 0.0, 1.0)),
                          image, True))

    # A numpy view of an ITK image is indexed [z, y, x], so a shift along x
    # moves the last axis and a shift along z the first.
    assert np.array_equal(along_x[:, :, :-1], before[:, :, 1:])
    assert np.array_equal(along_z[:-1], before[1:])


def test_the_input_image_is_not_modified(label_image):
    image = label_image()
    before = sitk.GetArrayFromImage(image).copy()

    pipeline.resample(image, _shift(0.3), is_segmentation=False)

    assert np.array_equal(sitk.GetArrayFromImage(image), before)
    assert image.GetOrigin() == (0.0, 0.0, 0.0)
