#!/usr/bin/env python3
# Bucketized external shuffle with Dask (target ~500MB shards), no path column required.

import random
import glob
import os
import argparse, math, hashlib
import numpy as np, pandas as pd
import dask, dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import fsspec
import uuid
import pyarrow.parquet as pq

import pyarrow as pa
pa.set_memory_pool(pa.system_memory_pool())

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

WORK_TMP = "/local/home/dvruette/tmp/dask-tmp"
os.makedirs(WORK_TMP, exist_ok=True)

# def splitmix64(x: np.ndarray) -> np.ndarray:
#     z = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
#     z ^= (z >> np.uint64(30)); z = (z * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
#     z ^= (z >> np.uint64(27)); z = (z * np.uint64(0x94D049BB133111EB)) & MASK64
#     z ^= (z >> np.uint64(31)); return z

# def _list_parquet_sizes(input_path: str):
#     fs, _, _ = fsspec.get_fs_token_paths(input_path)
#     if fs.isdir(input_path):
#         files = [p for p in fs.find(input_path) if p.endswith(".parquet")]
#     else:
#         files = [p for p in fs.glob(input_path) if p.endswith(".parquet")]
#     if not files:
#         raise SystemExit(f"No parquet files under {input_path}")
#     return [fs.info(p)["size"] for p in files]

# def next_power_of_two(n: int) -> int:
#     return 1 if n <= 1 else 1 << (n - 1).bit_length()

# def add_rand_and_bucket(pdf: pd.DataFrame, *, seed: int, b: int, partition_info=None) -> pd.DataFrame:
#     """
#     Adds:
#       - rand64 : uint64 (deterministic per row, seeded by partition id)
#       - bucket : uint32 (top-b bits of rand64)
#       - pid    : uint32 (partition id)   [tiebreaker]
#       - rowpos : uint32 (0..len-1)       [tiebreaker]
#     """
#     n = len(pdf)
#     if n == 0:
#         out = pdf.copy()
#         out["rand64"] = np.array([], dtype=np.uint64)
#         out["bucket"] = np.array([], dtype=np.uint32)
#         return out

#     pid = np.uint32(0 if partition_info is None else partition_info["number"])
#     rowpos = np.arange(n, dtype=np.uint32)

#     # Deterministic per-partition stream: base_seed ^ pid, then per-row counter
#     base = (np.uint64(seed) ^ np.uint64(pid)) + rowpos.astype(np.uint64)
#     r64 = splitmix64(base)
#     bucket = (r64 >> np.uint64(64 - b)).astype(np.uint32)

#     out = pdf.copy()
#     out["rand64"] = r64
#     out["bucket"] = bucket
#     return out

def add_rand(pdf, partition_info=None):
    pid = partition_info["number"]
    rng = np.random.default_rng(20250815 + pid)
    pdf["_shuf"] = rng.integers(0, 2**31 - 1, size=len(pdf), dtype=np.int64)
    return pdf

@dask.delayed
def _write_one(pdf, base, bucket, compression, row_group_size_bytes):
    if pdf.empty:
        return None
    # Each partition should contain exactly one bucket value now
    # bvals = pdf["bucket"].unique()
    # assert len(bvals) == 1, f"partition has multiple buckets: {bvals[:5]}"
    # bucket = int(bvals[0])

    table = pa.Table.from_pandas(pdf, preserve_index=False)

    # path = f"{base}/bucket={bucket}/part-{uuid.uuid4().hex}.parquet"
    path = os.path.join(base, f"bucket={bucket}", f"part-{uuid.uuid4().hex}.parquet")
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
    ap = argparse.ArgumentParser(description="Bucketized random shuffle writer (Dask, ~500MB shards).")
    ap.add_argument("--input", required=True, help="Parquet dir or glob (use **/*.parquet for recursion).")
    ap.add_argument("--output", required=True, help="Output directory for shuffled dataset.")
    ap.add_argument("--seed", type=int, default=0x5F3759DF, help="Deterministic seed for rand64.")
    ap.add_argument("--buckets", type=int, default=0, help="If 0, auto-compute from --target-shard-mb.")
    ap.add_argument("--target-shard-mb", type=int, default=500, help="Target compressed shard size.")
    ap.add_argument("--workers", type=int, default=64, help="Dask worker processes.")
    ap.add_argument("--threads-per-worker", type=int, default=1, help="Threads per worker.")
    ap.add_argument("--total-mem-gb", type=float, default=512.0, help="Total RAM budget.")
    # ap.add_argument("--row-groups", action="store_true", help="Split input by Parquet row groups.")
    ap.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip"], help="Output compression.")
    ap.add_argument("--row-group-size-bytes", type=int, default=64*(1<<20), help="Output row group size.")
    args = ap.parse_args()

    # Cluster (bounded memory per worker)
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
        "distributed.worker.memory.target": 0.5,
        "distributed.worker.memory.spill": 0.6,
        "distributed.worker.memory.pause": 0.85,
        "distributed.worker.memory.terminate": 0.95,
        "dataframe.shuffle.method": "p2p",
        "dataframe.shuffle.max_branch": 16,
        "distributed.worker.temp-directory": WORK_TMP,
        "temporary-directory": WORK_TMP,
    })


    print("scheduler tmp:", client.scheduler_info().get("services", {}))
    print("worker local dirs:", client.run(lambda dask_worker: dask_worker.local_directory))

    # # Decide bucket count (B) from target shard size unless overridden
    # if args.buckets and (args.buckets & (args.buckets - 1)) == 0:
    #     B = args.buckets
    #     print(f"[plan] Using user-specified B={B}")
    # else:
    #     total_bytes = int(sum(_list_parquet_sizes(args.input)))
    #     target_bytes = int(args.target_shard_mb * (1<<20))
    #     need = max(1, math.ceil((total_bytes * 1.25) / target_bytes))  # 25% slack
    #     B = next_power_of_two(need)
    #     print(f"[plan] total≈{total_bytes/1e12:.3f}TB, target≈{args.target_shard_mb}MB → need~{need} → B={B}")
    # b = int(math.log2(B))

    # 1) Load
    ddf = dd.read_parquet(
        args.input,
        engine="pyarrow",
        gather_statistics=False,
        split_row_groups=True,
        aggregate_files=False,
    )

    meta = ddf._meta.copy()
    meta["_shuf"] = np.array([], dtype="int64")

    print(meta)

    ddf = ddf.map_partitions(add_rand, meta=meta)

    tasks = []

    for i in range(0, ddf.npartitions, 4):
        sub = ddf.partitions[i:i+4]

        sub = sub.set_index("_shuf", shuffle="p2p", sorted=False, drop=True, partition_size="500MB")

        print(f"[write] → {args.output}")
        task = sub.to_parquet(
            args.output,
            engine="pyarrow",
            compression=args.compression,
            write_index=False,
            schema={"tokens": pa.list_(pa.int32())},
            compute=False,
        )
        tasks.append(task)
        # tasks += [_write_one(
        #             part,
        #             args.output,
        #             i,
        #             args.compression,
        #             args.row_group_size_bytes
        #         )
        #         for part in sub.to_delayed()]
    dask.compute(tasks)


    # # 2) Add rand64/bucket + deterministic tiebreakers (pid,rowpos)
    # meta = ddf._meta.assign(
    #     rand64=np.uint64(0), bucket=np.uint32(0)
    # )
    # ddf = ddf.map_partitions(add_rand_and_bucket, seed=args.seed, b=b, meta=meta)

    # # 3) Range-partition by bucket (exactly one partition per bucket)
    # divisions = list(range(B)) + [B]
    # ddf = ddf.set_index("bucket", divisions=divisions, shuffle="p2p", sorted=False, drop=False)
    # # ddf = ddf.set_index("bucket", shuffle="p2p", sorted=False, drop=False).repartition(divisions=divisions)

    # # 4) Sort within each bucket by (rand64)
    # ddf = ddf.map_partitions(lambda pdf: pdf.sort_values("rand64"))

    # # 5) Write (≈ one file per bucket, ~target shard size)
    # print(f"[write] → {args.output}")
    # ddf.to_parquet(
    #     args.output,
    #     engine="pyarrow",
    #     compression=args.compression,
    #     write_index=False,
    #     partition_on=["bucket"],
    #     write_metadata_file=True,
    #     row_group_size_bytes=args.row_group_size_bytes,
    # )

    # # materialize writes (no extra repartitions)
    # tasks = [_write_one(
    #             part,
    #             args.output,
    #             args.compression,
    #             args.row_group_size_bytes
    #         )
    #         for part in ddf.to_delayed()]
    # dask.compute(tasks)

    print("[done] Shuffled dataset is ready.")

if __name__ == "__main__":
    main()
