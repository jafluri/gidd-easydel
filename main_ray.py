import os, json, socket, ray
from pprint import pprint
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from ray.autoscaler.sdk import request_resources
from ray._private.accelerators import TPUAcceleratorManager

ray.init(runtime_env={"py_modules": [os.path.join(os.getcwd(), "gidd_easydel")]})


HOSTS_PER_SLICE_BY_VERSION = {
    "v6e-4": 1,
    "v6e-8": 1,
    "v6e-16": 4,
    "v6e-32": 8,
    "v6e-64": 16,
    "v6e-128": 32,
    "v6e-256": 64,
}

CHIPS_PER_HOST_BY_VERSION = {
    "v6e-4": 4,
    "v6e-8": 8,
    "v6e-16": 4,
    "v6e-32": 4,
    "v6e-64": 4,
    "v6e-128": 4,
    "v6e-256": 4,
}

TPU_VERSION = os.getenv("TPU_VERSION", "v6e-8")
TPU_POD_COUNT = os.getenv("TPU_POD_COUNT", "1")
TPU_ZONE = os.getenv("TPU_ZONE", "")

label = f"TPU-{TPU_VERSION}-head"
num_slices = int(TPU_POD_COUNT)

hosts_per_slice = HOSTS_PER_SLICE_BY_VERSION.get(TPU_VERSION, 1)
chips_per_host = CHIPS_PER_HOST_BY_VERSION.get(TPU_VERSION, 4)
num_hosts = num_slices * hosts_per_slice

port = int(os.environ.get("COORD_PORT", "9876"))



base_env = {
    "EASYDEL_PROFILING": os.getenv("EASYDEL_PROFILING", "0"),  # Enable EasyDeL profiling.
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
    "HF_TOKEN": os.getenv("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
    "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
    "HF_DATASETS_OFFLINE": "0",  # Allow online dataset access.
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
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

from args import parse_args
ARGS = parse_args(SAVE_DIRECTORY, WANDB_ENTITY, DATA_FILES)
assert ARGS.data_files is not None




@ray.remote(
    num_cpus=0,
    resources={"TPU": chips_per_host},
    runtime_env={"env_vars": base_env},
)
def main(proc_id, num_procs):
    import ray
    ip = ray.util.get_node_ip_address()
    print(f"inside worker: {proc_id=}, {ip=}, {num_procs=}")

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

    # import jax
    # jax.distributed.initialize(
    #     # coordinator_address=f"{os.getenv('MEGASCALE_COORDINATOR_ADDRESS')}:{os.getenv('MEGASCALE_PORT')}",
    #     # process_id=proc_id,
    #     # num_processes=num_procs,
    #     initialization_timeout=30,
    # )
    # print("initialized jax distributed")


    import jax
    return {
        "ip": ip,
        "slice_id": os.getenv("MEGASCALE_SLICE_ID"),
        "proc_id": proc_id,
        "host": socket.gethostname(),
        "proc_index": jax.process_index(),
        "proc_count": jax.process_count(),
        "device_count": len(jax.devices()),
    }




def run_on_host(remote_fn, slice_info, proc_id, num_procs, env):
    call = remote_fn.options(
        num_cpus=0,
        resources={slice_info["pod_name"]: 1, "TPU": slice_info["num_tpus_per_host"]},
        runtime_env={"env_vars": env},
    ).remote(proc_id=proc_id, num_procs=num_procs)
    return call


def run_on_slice(remote_fn, slice_info, slice_id, coord_ip, num_procs):
    print(json.dumps(slice_info, indent=2, sort_keys=True))
    hosts_per_slice = slice_info["num_hosts"]
    calls = []
    for host_id in range(hosts_per_slice):
        proc_id = host_id + slice_id * hosts_per_slice
        env = dict(base_env,
            MEGASCALE_COORDINATOR_ADDRESS=coord_ip,
            MEGASCALE_NUM_SLICES=str(num_slices),
            MEGASCALE_PORT=str(port),
            MEGASCALE_SLICE_ID=str(slice_id),
            JAX_PLATFORMS="tpu"
        )
        calls.append(
            run_on_host(remote_fn, slice_info, proc_id, num_procs, env)
        )
    return calls

@ray.remote(num_cpus=0)
def discover_slice():
    import os, ray
    pod_name = ray.util.accelerators.tpu.get_current_pod_name()
    num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
    tpu_type = TPUAcceleratorManager._get_current_node_tpu_pod_type()
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "ip": ray.util.get_node_ip_address(),
        "pod_name": pod_name,
        "num_hosts": num_hosts,
        "num_tpus_per_host": num_tpus_per_host,
        "tpu_type": tpu_type,
    }

def run_on_multislice(remote_fn, tpu_type, num_slices=1):
    label = f"TPU-{tpu_type}-head"
    bundles = [{"CPU": 0, label: 1} for _ in range(num_slices)]
    request_resources(bundles=bundles)

    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    slice_info = ray.get([
        discover_slice.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i, True)).remote()
        for i in range(num_slices)
    ])

    coord_ip = slice_info[0]['ip']

    num_procs = sum(
        slice_info[slice_id]["num_hosts"] * slice_info[slice_id]["num_tpus_per_host"]
        for slice_id in range(num_slices)
    )

    calls = []
    for slice_id in range(num_slices):
        calls.extend(run_on_slice(
            remote_fn=remote_fn,
            slice_info=slice_info[slice_id],
            slice_id=slice_id,
            coord_ip=coord_ip,
            num_procs=num_procs
        ))
    return calls

results = ray.get(run_on_multislice(remote_fn=main, tpu_type=TPU_VERSION, num_slices=num_slices))
print(json.dumps({"results": results}, indent=2, sort_keys=True))
