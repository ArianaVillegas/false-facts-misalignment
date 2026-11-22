"""Tinker LoRA SFT of Qwen3.8-27B on the paper's sycophancy-medicine data (arXiv:2606.09068).

Thinking-OFF renderer for train and eval alike (this project's EM behavior is thinking-gated).
Usage: python scripts/sft_syco_med.py --log_path=output/finetuning/qwen3.8-27b/syco_med_lr5e5 --learning_rate=5e-5
"""

import fire
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.supervised import (ChatDatasetBuilderCommonConfig,
                                        FromConversationFileBuilder, train)

from src.evaluation.inspect_models import no_thinking_renderer

MODEL = "Qwen/Qwen3.8-27B"
DATA = "data/finetuning/syco_med_messages.jsonl"


def build_config(log_path: str, learning_rate: float) -> train.Config:
    renderer = no_thinking_renderer(MODEL)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name=renderer,
        max_length=2048,
        batch_size=16,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    builder = FromConversationFileBuilder(common_config=common, file_path=DATA)
    return train.Config(
        log_path=log_path,
        model_name=MODEL,
        recipe_name="syco_med",
        renderer_name=renderer,
        dataset_builder=builder,
        learning_rate=float(learning_rate),
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=32,
        adam_beta2=0.95,
        adam_eps=1e-12,
        save_every=125,  # 6000/16 = 375 steps, so a checkpoint lands on the final step
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
    )


def main(log_path: str, learning_rate: float, dry_run: bool = False):
    load_dotenv()
    cfg = build_config(log_path, learning_rate)
    if dry_run:
        train_ds, _ = cfg.dataset_builder()
        print(f"OK config: model={cfg.model_name} renderer={cfg.renderer_name} lr={cfg.learning_rate} "
              f"batches={len(train_ds)} lora_rank={cfg.lora_rank}")
        return
    train.main(cfg)
    rec = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    print("FINAL_SAMPLER_PATH", rec.sampler_path if rec else None)


if __name__ == "__main__":
    fire.Fire(main)
