import numpy as np
import dask.dataframe as dd

class BufferedPartitionSampler:
    def __init__(self, ddf: dd.DataFrame, K: int, loop: bool = True, random_state: int | None = None):
        self.ddf = ddf
        self.K = max(1, int(K))
        self.loop = bool(loop)
        self.rng = np.random.default_rng(random_state)
        self.nparts = ddf.npartitions
        self.cols = list(ddf.columns)
        self._order = None
        self._order_i = 0
        self._buffer = []
        self._reset_order()
        self._fill_to_target(self._target_len())

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
