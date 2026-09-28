import os
import sys

sys.path.append("..")  # only needed once

def save_model_and_tokenizer(trainer, output_dir):
    """
    Save the training model and tokenizer.

    Mirrors the convention of save_finetuned_model:
      * checkpoint directory is  <output_dir>/final
      * tokenizer is saved alongside the model

    Uses Trainer.save_model() so DeepSpeed ZeRO-3 weights are gathered
    correctly before writing to disk (stage3_gather_16bit_weights_on_model_save).
    """
    # Match the fine-tuning code's directory name
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    # Save the model via the Trainer (DeepSpeed-aware).
    trainer.save_model(final_output_dir)

    # Save the tokenizer
    trainer.tokenizer.save_pretrained(final_output_dir)

    print(f"Model saved")
    return final_output_dir