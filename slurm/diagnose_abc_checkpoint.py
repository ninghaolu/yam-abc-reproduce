"""Opt-in checkpoint timings and stall stacks for the short ABC diagnostic job."""

import faulthandler
import functools
import logging
import time

from orbax.checkpoint._src.serialization import replica_slices
from openpi.training import checkpoints


def timed(label, fn, *, dump_stacks=False):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        logging.warning("CKPT_DIAG START %s", label)
        if dump_stacks:
            faulthandler.dump_traceback_later(300, repeat=True)
        try:
            return fn(*args, **kwargs)
        finally:
            if dump_stacks:
                faulthandler.cancel_dump_traceback_later()
            logging.warning("CKPT_DIAG END %s elapsed_s=%.3f", label, time.monotonic() - start)
    return wrapped


original_transfer = replica_slices.transfer_arrays_to_host


@functools.wraps(original_transfer)
def transfer(arrays, *args, **kwargs):
    logging.warning(
        "CKPT_DIAG transfer arrays=%d logical_GiB=%.3f",
        len(arrays), sum(a.size * a.dtype.itemsize for a in arrays) / 2**30,
    )
    return timed("transfer_arrays_to_host", original_transfer, dump_stacks=True)(arrays, *args, **kwargs)


replica_slices.transfer_arrays_to_host = transfer
checkpoints.save_state = timed("save_state", checkpoints.save_state)
# This also covers stalls after the synchronous staging phase has returned.
manager_class = checkpoints.ocp.CheckpointManager
manager_class.wait_until_finished = timed(
    "wait_until_finished", manager_class.wait_until_finished, dump_stacks=True
)
logging.warning("CKPT_DIAG enabled; stalled checkpoint calls dump stacks every 300 seconds")
