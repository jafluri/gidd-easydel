import time
import os
import json
import uuid
import logging

import ray
from pprint import pprint
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from ray.autoscaler.sdk import request_resources
from ray._private.accelerators import TPUAcceleratorManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ray.init(runtime_env={"py_modules": [os.path.join(os.getcwd(), "gidd_easydel")]})

VERBOSE = (os.getenv("VERBOSE", "0") == "1")

TPU_VERSION = os.getenv("TPU_VERSION", "v6e-8")
TPU_POD_COUNT = os.getenv("TPU_POD_COUNT", "1")
TPU_ZONE = os.getenv("TPU_ZONE", "")
PORT = int(os.environ.get("COORD_PORT", "9876"))

GIDD_RUN_ID = os.getenv("GIDD_RUN_ID", str(uuid.uuid4()))

WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", None)

num_slices = int(TPU_POD_COUNT)

base_env = {
    "EASYDEL_PROFILING": os.getenv("EASYDEL_PROFILING", "0"),  # Enable EasyDeL profiling.
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
    "HF_TOKEN": os.getenv("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
    "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
    "HF_DATASETS_OFFLINE": "1",  # Prevent crash if HF is down
    "TRANSFORMERS_OFFLINE": "0",  # Need for tokenizer
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY_FOR_EASYDEL", ""),  # W&B API key.
    "TPU_NAME": os.getenv("TPU_NAME", ""),
    "TPU_VERSION": TPU_VERSION,
    "TPU_ZONE": TPU_ZONE,
    "TPU_POD_COUNT": TPU_POD_COUNT,
}

if not TPU_ZONE:
    print("Warning: TPU_ZONE is not set. This is fine if you're not running on TPU or if the TPU has environment variables SAVE_DIRECTORY and DATA_FILES.")
    SAVE_DIRECTORY = os.getenv("SAVE_DIRECTORY", "outputs/diffusion_trainer")
    DATA_FILES = os.getenv("DATA_FILES", None)
else:
    SAVE_DIRECTORY = os.getenv("SAVE_DIRECTORY", f"gs://gidd-checkpoints_{TPU_ZONE[:-2]}")
    DATA_FILES = os.getenv("DATA_FILES", f"gs://nemotron-cc_{TPU_ZONE[:-2]}")


from args import parse_args
ARGS = parse_args(SAVE_DIRECTORY, WANDB_ENTITY, DATA_FILES)
assert ARGS.data_files is not None

ARGS.gidd_run_id = GIDD_RUN_ID
ARGS.ray_job_id = ray.runtime_context.get_runtime_context().get_job_id()

@ray.remote
def main():
    import easydel as ed
    from gidd_easydel.train import train

    try:
        pprint(ARGS)
        train(ARGS)
    except Exception as e:
        import traceback
        print("An error occurred during training:")
        traceback.print_exc()
        raise e


def submit_to_host(remote_fn, host_info, env):
    call = remote_fn.options(
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=host_info["node_id"], soft=False),
        resources={"TPU": host_info["num_tpus"]},
        runtime_env={"env_vars": env},
    ).remote()
    return call


def submit_to_slice(remote_fn, slice_info, slice_id, coord_ip, coord_port=PORT):
    if VERBOSE:
        print(json.dumps(slice_info, indent=2, sort_keys=True))
    hosts_per_slice = slice_info["num_hosts"]
    calls = []
    for host_id in range(hosts_per_slice):
        if num_slices > 1:
            env = dict(base_env,
                MEGASCALE_COORDINATOR_ADDRESS=coord_ip,
                MEGASCALE_NUM_SLICES=str(num_slices),
                MEGASCALE_PORT=str(coord_port),
                MEGASCALE_SLICE_ID=str(slice_id),
                JAX_PLATFORMS="tpu"
            )
        else:
            env = dict(base_env)
        calls.append(
            submit_to_host(remote_fn, slice_info["hosts"][host_id], env)
        )
    return calls


@ray.remote(num_cpus=0)
def discover_hosts_on_slice():
    import ray
    return {
        "ip": ray.util.get_node_ip_address(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "pod_name": ray.util.accelerators.tpu.get_current_pod_name(),
        "num_tpus": TPUAcceleratorManager.get_current_node_num_accelerators(),
    }


@ray.remote(num_cpus=0)
def discover_slice():
    import ray
    pod_name = ray.util.accelerators.tpu.get_current_pod_name()
    num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
    tpu_type = TPUAcceleratorManager._get_current_node_tpu_pod_type()

    bundles = [{"CPU": 0, pod_name: 1} for _ in range(num_hosts)]
    request_resources(bundles=bundles)
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    try:
        host_info = ray.get([
            discover_hosts_on_slice.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i, True)).remote()
            for i in range(num_hosts)
        ])

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "ip": ray.util.get_node_ip_address(),
            "pod_name": pod_name,
            "num_hosts": num_hosts,
            "num_tpus_per_host": num_tpus_per_host,
            "tpu_type": tpu_type,
            "hosts": host_info,
        }, pg
    except:
        remove_placement_group(pg)
        raise


def submit_to_multislice(remote_fn, tpu_type, num_slices=1):
    label = f"TPU-{tpu_type}-head"
    bundles = [{"CPU": 0, label: 1} for _ in range(num_slices)]
    request_resources(bundles=bundles)

    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    all_pgs = [pg]

    results = ray.get([
        discover_slice.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i, True)).remote()
        for i in range(num_slices)
    ])
    slice_infos, host_pgs = zip(*results)
    all_pgs.extend(host_pgs)

    coord_ip = slice_infos[0]['ip']

    # num_procs = sum(
    #     slice_infos[slice_id]["num_hosts"] * slice_infos[slice_id]["num_tpus_per_host"]
    #     for slice_id in range(num_slices)
    # )

    try:
        calls = []
        for slice_id in range(num_slices):
            calls.extend(submit_to_slice(
                remote_fn=remote_fn,
                slice_info=slice_infos[slice_id],
                slice_id=slice_id,
                coord_ip=coord_ip,
                coord_port=PORT,
            ))
        return calls, all_pgs
    except:
        for pg in all_pgs:
            remove_placement_group(pg)
        raise


def run_on_multislice_resumable(
    remote_fn,
    tpu_type,
    num_slices=1,
    max_errors=0,
    max_preemptions=128,
):
    assert WANDB_ENTITY is not None and WANDB_PROJECT is not None, "W&B entity and project must be set for resumable run"
    num_preemptions = 0
    num_errors = 0
    while True:
        pgs = []
        try:
            calls, pgs = submit_to_multislice(remote_fn, tpu_type, num_slices)
            ray.get(calls)
        except (
            ray.exceptions.RayError,
            ray.exceptions.RayTaskError,
            ray.exceptions.TaskUnschedulableError,
        ) as e:
            if any(x in str(e).lower() for x in ["preempted", "not schedulable"]):
                num_preemptions += 1
                logger.warning(f"TPU job preempted ({num_preemptions=}): {e}")
                if num_preemptions > max_preemptions:
                    logger.error("Maximum number of preemptions reached. Exiting.")
                    raise
            elif "couldn't connect to 'https://huggingface.co'" in str(e).lower():
                logger.warning(f"TPU job failed (couldn't connect to HF). Resubmitting without penalty: {e}", exc_info=e)
            else:
                num_errors += 1
                logger.warning(f"TPU job failed ({num_errors=}): {e}", exc_info=e)
                if num_errors > max_errors:
                    logger.error("Maximum number of errors reached. Exiting.")
                    raise
            
            # try to resubmit
            import wandb
            runs = wandb.Api().runs(
                path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"config.gidd_run_id": GIDD_RUN_ID},
                order="+created_at",
            )
            if len(runs) > 1:
                logger.warning(f"Found multiple runs with gidd_run_id '{GIDD_RUN_ID}'. Resuming from newest one.")
                runs = runs[:1]

            if len(runs) == 1:
                run = runs[0]
                logger.info(f"Resuming from W&B run: {run.id}")
                ARGS.resume_wandb_id = run.id
        finally:
            for pg in pgs:
                remove_placement_group(pg)
            # let's just chill for a bit
            time.sleep(5)


ray.get(
    run_on_multislice_resumable(
        remote_fn=main,
        tpu_type=TPU_VERSION,
        num_slices=num_slices,
    )
)
