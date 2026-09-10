"""
hausdorff_distance_95 with process-level timeout protection.

medpy's hd95 (which wraps scipy's distance_transform_edt) can hang
indefinitely on certain degenerate binary masks.  When this happens
during validation the whole training run stalls.

This module provides ``hd95_safe`` — a drop-in replacement that runs
hd95 in a *separate* process (using the ``spawn`` start method so the
child never inherits the parent's CUDA context).  If the call exceeds
the time limit the child is killed and NaN is returned, allowing
validation to continue.

Usage
-----
    from light_training.evaluation.hd95_safe import hd95_safe

    # instead of:  h = hausdorff_distance_95(pred, gt, voxel_spacing=vs)
    h = hd95_safe(pred, gt, voxel_spacing=vs)
"""

import multiprocessing as mp
import numpy as np

# ── config ────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT_SEC = 30   # per-sample time limit

# Use 'spawn' to avoid inheriting CUDA context into the child process
# (fork + CUDA = undefined behaviour; spawn gives a clean slate).
_mp_spawn = mp.get_context("spawn")


# ── worker (must be at module top-level so pickle can find it) ────────
def _hd95_worker(pred, gt, voxel_spacing, conn):
    """Run hausdorff_distance_95 in a child process; send result via pipe."""
    # Deferred import so the child only loads what it needs.
    from light_training.evaluation.metric import hausdorff_distance_95
    try:
        val = hausdorff_distance_95(pred, gt, voxel_spacing=voxel_spacing)
        conn.send(val)
    except Exception:
        conn.send(float("nan"))
    finally:
        conn.close()


# ── public API ────────────────────────────────────────────────────────
def hd95_safe(pred, gt, voxel_spacing=None, timeout=DEFAULT_TIMEOUT_SEC):
    """hausdorff_distance_95 with a process-level timeout.

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary masks (3-D for a single volume).
    voxel_spacing : list or None
        Voxel spacing passed through to medpy.
    timeout : int
        Seconds to wait before killing the child and returning NaN.

    Returns
    -------
    float
        HD95 value, or NaN on timeout / error.
    """
    parent_conn, child_conn = _mp_spawn.Pipe(duplex=False)
    proc = _mp_spawn.Process(
        target=_hd95_worker,
        args=(pred, gt, voxel_spacing, child_conn),
        daemon=True,
    )
    proc.start()
    if parent_conn.poll(timeout):
        result = parent_conn.recv()
    else:
        result = float("nan")
        print(f"[hd95_safe] ⚠ timeout ({timeout}s) – returning NaN")
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return result
