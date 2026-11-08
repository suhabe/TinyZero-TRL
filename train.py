from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer


def create_trainer(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, train_dataset: Dataset, eval_dataset: Dataset, config: GRPOConfig) -> GRPOTrainer:
    raise NotImplementedError
