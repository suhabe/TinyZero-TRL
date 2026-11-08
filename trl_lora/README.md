# TinyZero with TRL and PEFT

A minimal adaptation of [TinyZero's Countdown recipe](https://github.com/Jiayi-Pan/TinyZero) for `Qwen/Qwen2.5-3B`, Hugging Face TRL, and a PEFT LoRA adapter. The existing plain-PyTorch PPO implementation remains at the repository root.

**This uses GRPO instead of upstream's PPO with a learned critic.** It preserves the Countdown task, base-model prompt, rule reward, and KL coefficient; it is not an exact reproduction of TinyZero's optimizer or published results.

## Core idea

1. Sample eight answers to the same question from the current model.
2. Score each completion: 0 for missing answer tags, 0.1 for a tagged wrong answer, 1 for a correct expression using every supplied number exactly once.
3. TRL subtracts the group's mean reward and divides by its standard deviation to obtain advantages.
4. TRL updates only the LoRA weights and adds a KL penalty against the frozen base model.

There is no supervised warm-up, reward model, or learned critic. PEFT freezes the base weights; TRL temporarily disables the adapter to compute reference-policy log probabilities.

Read `countdown.py` for the prompt and reward, `train.py` for training, and `modal_train.py` for GPU execution. All Python functions are annotated, and code statements and declarations are kept on single lines.

## Recipe mapping

| Setting | This example |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B`, not Instruct |
| Dataset | `Jiayi-Pan/Countdown-Tasks-3to4` |
| Train / held-out rows | `[0:327680]` / `[327680:328704]` |
| Prompt | Upstream base-model prefix, ending in `<think>` |
| Reward | 0 / 0.1 / 1; no separate thinking-format bonus |
| Prompt / response limits | 256 / 1,024 tokens |
| Sampling temperature | 1.0 |
| Learning rate / KL coefficient | `1e-6` / `0.001` |
| Epochs / default update cap | 15 / 1,000 |
| LoRA | Rank 16, alpha 32, all linear layers except output head |
| Update batch | Eight completions of one question; microbatch one |
| Loss | TRL `loss_type="grpo"`, group-normalized advantages |
| Precision | BF16 base, PEFT trainable adapter weights |
| Evaluation / checkpoint interval | Every 100 optimizer updates, plus final evaluation |

Upstream uses full-parameter PPO, a separate critic, and a batch of 256 questions. This example uses critic-free GRPO and a much smaller batch for readability and memory use. The learning rate is retained from upstream; it has not been tuned for LoRA. Equal rewards within a group produce zero policy advantages, so learning depends on sampling varied answers.

The scorer uses an AST interpreter for integer literals and `+ - * /`, rather than executing model-generated code. It accepts the last answer tag across multiple lines, whereas upstream searches only the final line. It rejects non-Countdown operators such as exponentiation and floor division that upstream's permissive evaluator can accept.

## Run locally

Use Python 3.12 and a CUDA GPU with BF16 support.

```bash
cd trl_lora
pip install -r requirements.txt
python train.py
```

For a short execution check with the actual 3B base model:

```bash
python train.py --max-steps 1 --max-completion-length 64 --eval-size 8
```

To remove the 1,000-update cap and request all 15 epochs:

```bash
python train.py --max-steps -1
```

The final adapter and tokenizer go to `outputs/countdown/adapter`; evaluation metrics go to `outputs/countdown/eval_results.json`. Checkpoints include optimizer state for resuming:

```bash
python train.py --resume-from-checkpoint outputs/countdown/checkpoint-100
```

Use the same settings and output directory when resuming. `--max-steps` is the total target update count, including completed updates.

## Run on Modal

Only the Modal SDK is needed on your laptop. Run these commands from `trl_lora/`:

```bash
pip install modal==1.2.2
modal setup
modal run --detach modal_train.py --max-steps 1 --max-completion-length 64 --eval-size 8 --run-name smoke
```

Then start a longer run:

```bash
modal run --detach modal_train.py --run-name countdown
```

The function requests one H100, uses ordinary Transformers generation, and stores downloads, checkpoints, metrics, and adapters in the `tinyzero-trl-lora` volume. No Hugging Face token is needed for the public model and dataset. Use a different run name for each fresh experiment.

Download the final adapter:

```bash
modal volume get tinyzero-trl-lora /runs/countdown/adapter ./adapter
```

List checkpoints and resume a run:

```bash
modal volume ls tinyzero-trl-lora /runs/countdown
modal run --detach modal_train.py --run-name countdown --resume-from-checkpoint /data/runs/countdown/checkpoint-100
```

Modal invocations have a 24-hour timeout. Checkpoints are written every 100 updates and Modal periodically persists volume writes; resume a completed checkpoint if the invocation ends before training finishes. A full 15-epoch experiment can require many invocations. This example makes no cost or throughput claim.

## Load the adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./adapter")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter")
```

Use the same `make_prompt` function for inference; this is a base model, so do not add an instruction-model chat template.

## Check the implementation

```bash
pip install -r requirements-dev.txt
pytest -q
mypy countdown.py train.py modal_train.py
```

Tests need no model or dataset downloads. They cover the reward rules and perform an actual generation, GRPO update, evaluation, and adapter export with a tiny randomly initialized Qwen model. That integration test uses alternating synthetic rewards to guarantee a training signal; the Countdown reward is tested separately. These checks do not establish 3B GPU memory use, Modal execution, convergence, or reproduction of TinyZero's reasoning behavior.

## Upstream references

- [Training settings](https://github.com/Jiayi-Pan/TinyZero/blob/main/scripts/train_tiny_zero.sh)
- [Countdown dataset and base prompt](https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py)
- [Countdown reward](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py)
- [TRL GRPO 0.25.0 API](https://huggingface.co/docs/trl/v0.25.0/grpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
