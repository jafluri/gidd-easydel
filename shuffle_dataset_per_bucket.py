#!/usr/bin/env python3

import glob
import os, argparse, math
import numpy as np, pandas as pd
import dask, dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm.auto as tqdm

pa.set_memory_pool(pa.system_memory_pool())

WORK_TMP = "/local/home/dvruette/tmp/dask-tmp"
os.makedirs(WORK_TMP, exist_ok=True)



def per_partition_bucket_map(nparts: int, *, seed: int, B: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bucket_id = np.arange(nparts, dtype=np.uint32) % B
    rng.shuffle(bucket_id)
    return bucket_id

def add_rand(pdf: pd.DataFrame, *, seed: int, partition_info=None) -> pd.DataFrame:
    pid = 0 if partition_info is None else int(partition_info["number"])
    rng = np.random.default_rng(seed ^ pid)
    r64 = rng.integers(0, np.iinfo(np.uint64).max, size=len(pdf), dtype=np.uint64)
    return pdf.assign(rand64=r64)


@dask.delayed
def _write_one(pdf, base, pid, compression, row_group_size_bytes):
    if pdf.empty:
        return None
    # Each partition should contain exactly one bucket value now
    # bvals = pdf["bucket"].unique()
    # assert len(bvals) == 1, f"partition has multiple buckets: {bvals[:5]}"
    # bucket = int(bvals[0])

    table = pa.Table.from_pandas(pdf, preserve_index=False)

    # path = f"{base}/bucket={bucket}/part-{uuid.uuid4().hex}.parquet"
    path = os.path.join(base, f"part-{pid:05d}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=compression,
        data_page_size=1 << 20,
        write_statistics=False,
        row_group_size=row_group_size_bytes,  # pyarrow >= 17
        # schema={"tokens": pa.list_(pa.int32())},
    )
    return path


def main():
    ap = argparse.ArgumentParser(description="Partition-bucket shuffle (p2p within buckets, ~500MB shards).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0x5F3759DF)
    ap.add_argument("--files-per-bucket", type=int, default=None)
    ap.add_argument("--buckets", type=int, default=None)
    ap.add_argument("--target-shard-mb", type=int, default=500)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--threads-per-worker", type=int, default=1)
    ap.add_argument("--total-mem-gb", type=float, default=512.0)
    ap.add_argument("--compression", default="snappy", choices=["zstd","snappy","gzip"])
    ap.add_argument("--partition-size", default="1024MB", help="Partition size for Dask DataFrame.")
    ap.add_argument("--row-group-size-bytes", type=int, default=64*(1<<20))
    args = ap.parse_args()

    if args.buckets is None and args.files_per_bucket is None:
        raise ValueError("Either --buckets or --files-per-bucket must be specified.")
    if args.buckets is not None and args.files_per_bucket is not None:
        raise ValueError("Only one of --buckets or --files-per-bucket may be specified.")

    mem_per_worker_gb = max(4.0, min(args.total_mem_gb / max(1, args.workers), 32.0))
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        local_directory=WORK_TMP,
        processes=True,
        memory_limit=f"{mem_per_worker_gb}GB",
        dashboard_address=":8787",
    ) 
    client = Client(cluster)
    print("[dask] dashboard:", client.dashboard_link)

    dask.config.set({
        "dataframe.shuffle.method": "p2p",
        "distributed.worker.memory.target": 0.7,
        "distributed.worker.memory.spill": 0.8,
        "distributed.worker.memory.pause": 0.9,
        "distributed.worker.memory.terminate": 1.0,
        "dataframe.shuffle.max_branch": 16,
        "distributed.worker.temp-directory": WORK_TMP,
        "temporary-directory": WORK_TMP,
    })

    ddf = dd.read_parquet(
        args.input,
        engine="pyarrow",
        gather_statistics=False,
        split_row_groups=True,
        aggregate_files=False,
    )
    N = ddf.npartitions


    if args.buckets is None:
        files_per_bucket = args.files_per_bucket
        B = N // files_per_bucket
        if B * files_per_bucket != N:
            print(f"[warning] Partitions ({N}) are not evenly divisible into {files_per_bucket} files per bucket.")
            B += 1
    else:
        B = args.buckets
        files_per_bucket = (N - 1) // B + 1  # ceil division

    assert B > 0, f"Invalid bucket count: {B}"
    print(f"[plan] Using B={B} buckets with up to {files_per_bucket} files per bucket")

    meta = ddf._meta.copy()
    meta["rand64"] = np.array([], dtype="uint64")

    part_buckets = per_partition_bucket_map(N, seed=args.seed, B=B)


    idxs_by_bucket = {k: [] for k in range(B)}
    for i, k in enumerate(part_buckets.tolist()):
        idxs_by_bucket[k].append(i)

    fs, _, _ = fsspec.get_fs_token_paths(args.output)
    fs.mkdirs(args.output, exist_ok=True)

    ddf = ddf.map_partitions(add_rand, seed=args.seed, meta=meta)

    # tasks = []
    for k in tqdm.trange(B):
        idxs = idxs_by_bucket.get(k, [])
        if not idxs:
            continue
        sub = ddf.partitions[idxs]

        sub = sub.set_index("rand64", shuffle="p2p", sorted=False, drop=True, partition_size=args.partition_size)

        out_path = os.path.join(args.output, f"bucket={k}")
        tasks = [
            _write_one(
                part,
                out_path,
                i,
                args.compression,
                args.row_group_size_bytes
            )
            for i, part in enumerate(sub.to_delayed())
        ]

        dask.compute(tasks)

    print("[done] Shuffled dataset is ready.")

if __name__ == "__main__":
    main()
