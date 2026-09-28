import os
import sys

sys.path.append("..")  # only needed once

def save_finetuned_model(trainer, output_dir):
    """
    Save the fine-tuned model and tokenizer.

    Mirrors the fine-tuning code's convention:
      * checkpoint directory is  <output_dir>/final
      * tokenizer is saved alongside the model

    Uses Trainer.save_model() so DeepSpeed ZeRO-3 weights are gathered
    correctly before writing to disk (stage3_gather_16bit_weights_on_model_save).
    """
    # 1. Match the fine-tuning code's directory name
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    # 2. Save the model via the Trainer (DeepSpeed-aware).
    #    Do NOT call unwrapped.save_pretrained() here – with ZeRO-3 that
    #    would only write the local shard, producing an unusable checkpoint.
    trainer.save_model(final_output_dir)

    # Ensure the tokenizer is saved (Trainer.save_model usually does this,
    #    but we do it explicitly for safety and future compatibility).
    trainer.tokenizer.save_pretrained(final_output_dir)
    print(f"Fine-tuned model saved")
    return final_output_dir
