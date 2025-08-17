import numpy as np
import dask.dataframe as dd
from typing import List

class BufferedPartitionSampler:
    def __init__(self, ddf: dd.DataFrame, K: int, loop: bool = True, random_state: int | None = None):
        """
        Initialize a buffered random sampler over Dask partitions.

        Args:
            ddf (dask.dataframe.DataFrame):
                The input DataFrame; each Dask partition is treated as an independent shard.
            K (int):
                Target number of active partitions in the buffer. Sampling proceeds only when the
                buffer is full. If the dataset has fewer than K partitions, the target becomes the
                number of available partitions.
            loop (bool, default False):
                If True, once all partitions have been consumed the sampler reshuffles the partition
                order and continues; if False, iteration stops when all partitions are exhausted.
            random_state (int | None, default None):
                Seed for numpy.random.default_rng used for partition order shuffling and per-partition
                row shuffling.
        """

        self.ddf = ddf
        self.K = max(1, int(K))
        self.loop = bool(loop)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.nparts = ddf.npartitions
        self.cols = list(ddf.columns)
        self._order = None
        self._order_i = 0
        self._buffer = []
        self._reset_order()
        self._fill_to_target(self._target_len())

    def __reduce__(self):
        return (self.__class__, (self.ddf, self.K, self.loop, self.random_state))

    def __iter__(self):
        return self

    def __next__(self):
        target = self._target_len()
        if len(self._buffer) < target:
            self._fill_to_target(target)
        if len(self._buffer) < target:
            raise StopIteration
        tries = 0
        max_tries = self.nparts + target + 1
        while True:
            j = int(self.rng.integers(0, len(self._buffer)))
            try:
                row = next(self._buffer[j]["it"])
                return dict(zip(self.cols, row))
            except StopIteration:
                del self._buffer[j]
                self._fill_to_target(target)
                if len(self._buffer) < target:
                    raise StopIteration
                tries += 1
                if tries >= max_tries:
                    raise StopIteration

    def take(self, n: int):
        out = []
        for _ in range(int(n)):
            out.append(self.__next__())
        return out

    def _target_len(self):
        return min(self.K, self.nparts)

    def _reset_order(self):
        self._order = list(range(self.nparts))
        self.rng.shuffle(self._order)
        self._order_i = 0

    def _next_pid(self):
        if self._order_i >= len(self._order):
            if not self.loop:
                return None
            self._reset_order()
        pid = self._order[self._order_i]
        self._order_i += 1
        return pid

    def _randint(self):
        return int(self.rng.integers(0, np.iinfo(np.int32).max))

    def _make_entry(self, pid: int):
        dpart = self.ddf.partitions[pid].sample(frac=1.0, replace=False, random_state=self._randint())
        it = dpart.itertuples(index=False, name=None)
        return {"pid": pid, "it": it}

    def _fill_to_target(self, target_len: int):
        while len(self._buffer) < target_len:
            pid = self._next_pid()
            if pid is None:
                break
            self._buffer.append(self._make_entry(pid))


class ShuffledBucketSampler:
    def __init__(self, ddfs: List[dd.DataFrame], K: int | None = None, loop: bool = True, random_state: int | None = None):
        """
        Initialize a shuffled bucket sampler over pre-shuffled parquet buckets.

        Args:
            ShuffledBucketSampler (List[dd.DataFrame]):
                List of to pre-shuffled parquet datasets. Each bucket should contain
                approximately the same number of rows.
            K (int):
                Target number of active buckets in the buffer. Sampling proceeds only when the
                buffer is full. If the dataset has fewer than K buckets, the target becomes the
                number of available buckets.
            loop (bool, default True):
                If True, once all buckets have been consumed the sampler reshuffles the bucket
                order and continues; if False, iteration stops when all buckets are exhausted.
            random_state (int | None, default None):
                Seed for numpy.random.default_rng used for bucket order shuffling and random
                bucket selection.
        """
        
        self.ddfs = list(ddfs)
        self.K = len(self.ddfs) if K is None else max(1, int(K))
        self.loop = bool(loop)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.nbuckets = len(self.ddfs)
        
        # Load the first bucket to get column information
        first_ddf = self.ddfs[0]
        self.cols = list(first_ddf.columns)
        
        self._order = None
        self._order_i = 0
        self._buffer = []
        self._reset_order()
        self._fill_to_target(self._target_len())

    def __reduce__(self):
        return (self.__class__, (self.ddfs, self.K, self.loop, self.random_state))

    def __iter__(self):
        return self

    def __next__(self):
        target = self._target_len()
        if len(self._buffer) < target:
            self._fill_to_target(target)
        if len(self._buffer) < target:
            raise StopIteration
        tries = 0
        max_tries = self.nbuckets + target + 1
        while True:
            j = int(self.rng.integers(0, len(self._buffer)))
            try:
                row = next(self._buffer[j]["it"])
                return dict(zip(self.cols, row))
            except StopIteration:
                del self._buffer[j]
                self._fill_to_target(target)
                if len(self._buffer) < target:
                    raise StopIteration
                tries += 1
                if tries >= max_tries:
                    raise StopIteration

    def take(self, n: int):
        out = []
        for _ in range(int(n)):
            out.append(self.__next__())
        return out

    def _target_len(self):
        return min(self.K, self.nbuckets)

    def _reset_order(self):
        self._order = list(range(self.nbuckets))
        self.rng.shuffle(self._order)
        self._order_i = 0

    def _next_bucket_idx(self):
        if self._order_i >= len(self._order):
            if not self.loop:
                return None
            self._reset_order()
        bucket_idx = self._order[self._order_i]
        self._order_i += 1
        return bucket_idx

    def _make_entry(self, bucket_idx: int):
        ddf = self.ddfs[bucket_idx]
        # Since buckets are already pre-shuffled, we don't need to shuffle again
        it = ddf.itertuples(index=False, name=None)
        return {"bucket_idx": bucket_idx, "it": it}

    def _fill_to_target(self, target_len: int):
        while len(self._buffer) < target_len:
            bucket_idx = self._next_bucket_idx()
            if bucket_idx is None:
                break
            self._buffer.append(self._make_entry(bucket_idx))


