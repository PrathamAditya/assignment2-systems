import modal

app = modal.App("cs336-assignment2")

memory_volume = modal.Volume.from_name(
    "cs336-memory",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        copy=True,
        ignore=[
            ".git", ".venv", "__pycache__", "**/__pycache__",
            "*.pyc", "*.sqlite", "*.nsys-rep", "*.qdstrm",
            "nsys-report-*", "report*.nsys-rep",
            "cs336-basics/.venv", "cs336-basics/__pycache__",
        ],
    )
    .workdir("/root/project")
    .run_commands(
        "pip install uv",
        "uv sync",
    )
    .env({"PATH": "/root/project/.venv/bin:$PATH"})
)

# @app.function(
#     image=image,
#     gpu="B200",
#     volumes={"/my_vol": modal.Volume.from_name("flash_benchmarking")},
#     timeout=60 * 60,
# )

# def benchmark():
#     import json
#     from cs336_systems.flash_benchmarking import main
#     result = main("3")

#     file_path = "/my_vol/results.txt"
#     with open(file_path, "w") as f:
#         f.write(json.dumps(result, indent=4))

#     print(result)
#     modal.Volume.from_name("flash_benchmarking").commit()
#     print(f"Saved output_dict to {file_path}")

##############################################################################
##############################################################################

# Problem (distributed_communication_single_node): Distributed Communication (Single Node)
# @app.function(image=image, gpu="A10G:6", timeout=600)
# @app.function(image=image, gpu="H100:6", timeout=6000)
# def d_c_s_n_benchmark():
#     from cs336_systems.distributed_communication_single_node import main
#     main()

######################################################################################
######################################################################################

# Problem (naive_ddp): Naïve DDP
# @app.function(image=image, gpu="A10:4", timeout=3600)
# def naive_ddp():
#     from cs336_systems.naive_ddp import main
#     main()

######################################################################################
######################################################################################

# Problem (naive_ddp): Naïve DDP Benchmarking
@app.function(image=image, gpu="A100-80GB:2", timeout=3600)
def naive_ddp():
    from cs336_systems.minimal_ddp_flat_benchmarking import main
    main()

@app.local_entrypoint()
def run():
    naive_ddp.remote()