"""Float64 three-dimensional geometry, independent of the XY native extension."""

import numpy as np


def make_periodic(x, half_width, optimized=True):
    """Use the original strict comparisons and single subtract-then-add correction."""
    if optimized:
        np.subtract(x, 2 * half_width, out=x, where=x > half_width)
        np.add(x, 2 * half_width, out=x, where=x < -half_width)
    else:
        x[x > half_width] -= 2 * half_width
        x[x < -half_width] += 2 * half_width
    return x


def vector_norm(x, axis=0):
    """Keep the same XYZ addition order for reference and optimized paths."""
    xyz = np.moveaxis(x, axis, 0)
    return np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)


def get_dist_b2b_batch(positions, half_width, periodic, pair_sizes):
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, :, None] - positions[:, None, :]
    if periodic:
        delta = make_periodic(delta, half_width)
    center = vector_norm(delta)
    edge = np.abs(center - pair_sizes)
    return center, edge, center < pair_sizes


def get_dist_b2w(positions, sizes, half_width):
    """Penetration and contacts for -X,+X,-Y,+Y,-Z,+Z walls."""
    negative = sizes[None, :] - (positions + half_width)
    positive = sizes[None, :] - (half_width - positions)
    distances = np.stack((negative, positive), axis=1).reshape(6, -1)
    return np.abs(distances), distances > 0


def get_focused_batch(positions, headings, field_of_view, width,
                      remove_self, optimized=True):
    """XYZ neighbors with the original argsort tie and self-removal rules."""
    shape = (positions.shape[0], width, 3)
    target_positions = np.zeros(shape)
    target_headings = np.zeros(shape)
    norms = vector_norm(positions, axis=2)
    offset = int(remove_self)
    needed = min(norms.shape[1], width + offset)
    if optimized and norms.shape[1] >= 64 and 0 < needed < norms.shape[1]:
        pool = np.argpartition(norms, needed, axis=1)[:, :needed + 1]
        distances = np.take_along_axis(norms, pool, axis=1)
        local_order = np.argsort(distances, axis=1)
        ranked = np.take_along_axis(distances, local_order, axis=1)
        order = np.take_along_axis(pool, local_order, axis=1)[:, :needed].copy()
        ambiguous = (np.any(~np.isfinite(norms), axis=1)
                     | np.any(ranked[:, 1:] == ranked[:, :-1], axis=1))
        if np.any(ambiguous):
            order[ambiguous] = np.argsort(norms[ambiguous], axis=1)[:, :needed]
    else:
        order = np.argsort(norms, axis=1)
    selected = order[:, offset:offset + width]
    count = selected.shape[1]
    if count:
        visible = np.take_along_axis(norms, selected, axis=1) < field_of_view
        selected = selected[:, :, None]
        target_positions[:, :count] = np.where(
            visible[:, :, None], np.take_along_axis(positions, selected, axis=1), 0.0)
        target_headings[:, :count] = np.where(
            visible[:, :, None], np.take_along_axis(headings, selected, axis=1), 0.0)
    return target_positions, target_headings


def rotate_headings(headings, rotation):
    """Rodrigues rotation for column vectors; zero rotation preserves headings."""
    angles = vector_norm(rotation)
    axes = np.zeros_like(rotation)
    np.divide(rotation, angles[None, :], out=axes, where=angles[None, :] > 0)
    cosine, sine = np.cos(angles)[None, :], np.sin(angles)[None, :]
    cross = np.cross(axes.T, headings.T).T
    result = (headings * cosine + cross * sine
              + axes * np.sum(axes * headings, axis=0, keepdims=True) * (1 - cosine))
    result /= vector_norm(result)[None, :]
    return result
