"""Run the TRL/LoRA example on one Modal H100."""

from pathlib import Path

import modal


app = modal.App("tinyzero-trl-lora")
volume = modal.Volume.from_name("tinyzero-trl-lora", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt"))).env({"HF_HOME": "/data/huggingface", "TOKENIZERS_PARALLELISM": "false"}).add_local_python_source("train", "countdown")


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=86400, volumes={"/data": volume})
def run(max_steps: int = 1000, run_name: str = "countdown", max_completion_length: int = 1024, eval_size: int = 1024, resume_from_checkpoint: str | None = None) -> str:
    from train import train

    output_dir = f"/data/runs/{run_name}"
    train(output_dir=output_dir, max_steps=max_steps, max_completion_length=max_completion_length, eval_size=eval_size, resume_from_checkpoint=resume_from_checkpoint)
    volume.commit()
    return f"Adapter saved to {output_dir}/adapter in the tinyzero-trl-lora volume."


@app.local_entrypoint()
def main(max_steps: int = 1000, run_name: str = "countdown", max_completion_length: int = 1024, eval_size: int = 1024, resume_from_checkpoint: str | None = None) -> None:
    print(run.remote(max_steps=max_steps, run_name=run_name, max_completion_length=max_completion_length, eval_size=eval_size, resume_from_checkpoint=resume_from_checkpoint))
