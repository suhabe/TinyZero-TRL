"""Run with: modal run --detach modal_train.py"""

import subprocess
import sys
from pathlib import Path

import modal


app = modal.App("tinyzero-pytorch")
volume = modal.Volume.from_name("tinyzero-pytorch", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt"))).env({"HF_HOME": "/data/huggingface", "TOKENIZERS_PARALLELISM": "false"}).add_local_file(Path(__file__).with_name("train.py"), "/root/train.py")


@app.function(image=image, gpu="H200", timeout=86400, volumes={"/data": volume})
def run(model: str = "Qwen/Qwen2.5-3B", epochs: int = 15, batch_size: int = 256, minibatch_size: int = 64, max_new_tokens: int = 1024, max_steps: int = 10) -> None:
    command = [sys.executable, "/root/train.py", "--model", model, "--output-dir", "/data/checkpoints", "--epochs", str(epochs), "--batch-size", str(batch_size), "--minibatch-size", str(minibatch_size), "--max-new-tokens", str(max_new_tokens), "--max-steps", str(max_steps)]
    subprocess.run(command, check=True)
    volume.commit()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-3B", epochs: int = 15, batch_size: int = 256, minibatch_size: int = 64, max_new_tokens: int = 1024, max_steps: int = 10) -> None:
    run.remote(model, epochs, batch_size, minibatch_size, max_new_tokens, max_steps)
