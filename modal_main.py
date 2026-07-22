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

@app.function(
    image=image,
    gpu="B200",
    volumes={"/memory": memory_volume},
    timeout=60 * 10,
)
# Size d_model d_ff num_layers num_heads
# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 2560 10240 32 32
# 10B 4608 12288 50 36
def benchmark():
    from cs336_systems.benchmarking_script import method
    # method(
    #     d_model=2560,
    #     d_ff=10240,
    #     num_layers=32,
    #     num_heads=32,
    #     rope_theta=10000.0,
    #     warmup_steps=0,
    #     steps=1,
    #     which_type="f",
    #     which_data_type=None,
    #     context_length=2048,
    #     memory_profiling=True
    # )
    method(
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
        rope_theta=None,
        warmup_steps=5,
        steps=1,
        which_type="fb",
        which_data_type="b",
        context_length=2048,
        memory_profiling=True,
        group_size= 6
    )
    # method(d_model=768, d_ff = 3072, num_layers = 32, num_heads = 12, rope_theta=None, warmup_steps=5,
    # steps=1,which_type="fb", context_length=2048,which_data_type = "b", memory_profiling = True, group_size=6)

@app.local_entrypoint()
def run():
    benchmark.remote()