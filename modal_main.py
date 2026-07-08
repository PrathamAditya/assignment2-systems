import modal

app = modal.App("cs336-assignment2")

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
    timeout=60 * 20,
)
# Size d_model d_ff num_layers num_heads
# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 2560 10240 32 32
# 10B 4608 12288 50 36
def benchmark():
    from cs336_systems.benchmarking_script import method
    method(
        d_model=4608,
        d_ff=12288,
        num_layers=50,
        num_heads=36,
        rope_theta=10000.0,
        warmup_steps=5,
        steps=10,
        which_type="f",
        which_data_type="b",
        context_length=512,
    )

@app.local_entrypoint()
def run():
    benchmark.remote()