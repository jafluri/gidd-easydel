import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import dask.dataframe as dd


def _child_init():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _load_partition(path, storage_options=None, seed=0):
    ddf = dd.read_parquet(
        path,
        columns=["tokens"],
        engine="pyarrow",
        split_row_groups=False,
        storage_options=storage_options,
    )
    vals = ddf["tokens"].persist().compute().values
    rng = np.random.default_rng(seed)
    rng.shuffle(vals)
    return vals


def _loop_forever(it):
    while True:
        for x in it:
            yield x


def generate_rows_from_buckets(bucket_files, storage_options=None, seed=0, max_workers="auto", preload_factor=1):
    partition_iters = {
        i: _loop_forever(ps)
        for i, ps in enumerate(bucket_files)
    }
    num_buckets = len(bucket_files)

    if max_workers == "auto":
        max_workers = min(preload_factor * num_buckets, (os.cpu_count() or 1))

    rng = np.random.default_rng(seed)

    def _submit_next_partition(i):
        futures[i].append(executor.submit(_load_partition, next(partition_iters[i]), storage_options, rng.integers(1e6)))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_child_init) as executor:
        futures = {i: [] for i in range(num_buckets)}
        buffers = {}
        for _ in range(preload_factor):
            for i in partition_iters.keys():
                _submit_next_partition(i)

        next_bucket = rng.choice(list(futures.keys()))
        while futures:
            buffer = buffers.get(next_bucket)
            if buffer is None:
                values = futures[next_bucket].pop(0).result()
                buffer = {
                    "values": values,
                    "idx": 0,
                }
                buffers[next_bucket] = buffer
                _submit_next_partition(next_bucket)
            if buffer["idx"] >= len(buffer["values"]):
                del buffers[next_bucket]
                continue
            yield {"tokens": buffer["values"][buffer["idx"]]}
            buffer["idx"] += 1
            next_bucket = rng.choice(list(futures.keys()))
