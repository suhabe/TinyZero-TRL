# TinyZero in minimal PyTorch

A readable implementation of [TinyZero](https://github.com/Jiayi-Pan/TinyZero)'s **Countdown PPO recipe**. The learning algorithm is explicit PyTorch; Transformers loads and samples the language models. Despite the repository name, this implementation does not use TRL or veRL.

## Read the code

`train.py` contains the complete training loop:

1. Sample one completion per Countdown question from a base language model.
2. Give 0 for a missing answer, 0.1 for an incorrect answer, or 1 for a correct equation using every supplied number exactly once.
3. Subtract `0.001 * (policy_log_prob - reference_log_prob)` at every response token; add the equation reward at the final token.
4. Estimate advantages with a separate learned critic and GAE (`gamma = lambda = 1`), then whiten advantages over the rollout batch.
5. Update the actor with clipped PPO and an entropy bonus; update the critic with clipped value regression.

The reference model stays frozen. Actor and critic have separate trainable backbones. Only response tokens contribute to losses, including EOS when sampled. Each response is processed separately, so there is no padding mask machinery. The arithmetic scorer uses a small AST interpreter rather than executing generated code.

## Recipe defaults

These come from upstream's [training script](https://github.com/Jiayi-Pan/TinyZero/blob/main/scripts/train_tiny_zero.sh), [PPO config](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/trainer/config/ppo_trainer.yaml), [data preprocessing](https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py), and [reward function](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py).

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B` |
| Dataset | `Jiayi-Pan/Countdown-Tasks-3to4` |
| Training examples | First 327,680 rows |
| Validation examples | Next 1,024 rows |
| Prompt | Upstream base-model prefix ending with `<think>` |
| Rollout batch / PPO minibatch | 256 / 64 |
| Samples per question / PPO epochs per batch | 1 / 1 |
| Response limit / sampling temperature | 1,024 / 1.0 |
| Actor / critic learning rate | `1e-6` / `1e-5` |
| PPO / value clip | 0.2 / 0.5 |
| KL / entropy coefficient | 0.001 / 0.001 |
| Gradient norm limit | 1.0 |
| Training epochs | 15 |
| Validation / model export interval | Every 100 rollout batches |

This reproduces the learning recipe, not the distributed infrastructure or published results. Serial generation and one-response gradient accumulation are deliberately slower than veRL/vLLM. Unlike upstream, the microbatch is one response and losses are weighted by token count across each minibatch. Trainable parameters and Adam states stay FP32 so the small learning-rate updates are preserved; CUDA forwards use BF16 with gradient checkpointing. The default Modal GPU is an H200 to accommodate the actor, critic, reference, gradients, and optimizer states. GPU memory usage has not been measured here.

TinyZero reports that 0.5B models fail to learn reasoning; use them only for checking that the program runs. A short run is not a reproduction of the reported reasoning behavior.

## Run locally

Use Python 3.12 and a CUDA-capable PyTorch installation for training.

```bash
pip install -r requirements.txt
python train.py
```

A short check with a smaller model:

```bash
python train.py --model Qwen/Qwen2.5-0.5B --batch-size 2 --minibatch-size 2 --max-new-tokens 64 --max-steps 1
```

`--max-steps 0` runs all requested epochs. The console prints training reward and exact-answer accuracy. Exports contain the actor and tokenizer, loadable with `AutoModelForCausalLM.from_pretrained("checkpoints/final")`. They do not contain critic or optimizer state and cannot resume PPO training.

## Run on Modal

```bash
pip install 'modal==1.6.3'
modal setup
modal run --detach modal_train.py
```

The default Modal invocation runs **10 rollout batches**, then exports the actor. It uses an H200 GPU, downloads the public model and dataset, and persists the Hugging Face cache and exports in the `tinyzero-pytorch` volume.

For a short execution check:

```bash
modal run --detach modal_train.py --model Qwen/Qwen2.5-0.5B --batch-size 2 --minibatch-size 2 --max-new-tokens 64 --max-steps 1
```

To request the full recipe:

```bash
modal run --detach modal_train.py --max-steps 0 --epochs 15
```

Modal functions have a 24-hour maximum execution time. The full serial recipe is expected to exceed one invocation; this minimal runner provides no resume support. For long experiments, use the upstream distributed implementation. Re-running this script starts fresh and writes to the same export paths, so download an export before starting a run you want to keep separately.

Download the final model:

```bash
modal volume get tinyzero-pytorch /checkpoints/final ./tinyzero-model
```

## Check the implementation

```bash
pip install -r requirements.txt 'modal==1.6.3' mypy pytest
pytest -q
python -m mypy
```

The tests exercise Countdown scoring, GAE, PPO clipping, response-token alignment, and actual actor/critic optimization with a tiny randomly initialized Qwen model. They require no model or dataset downloads.
