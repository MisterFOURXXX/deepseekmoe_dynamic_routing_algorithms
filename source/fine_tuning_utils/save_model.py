import os
import sys

sys.path.append("..")  # only needed once

def save_finetuned_model(trainer, output_dir):
    """
    Save the fine-tuned model and tokenizer.
    Handles DeepSpeed ZeRO-3 weight gathering automatically.
    """
    final_output_dir = os.path.join(output_dir, "fine_tuned_final")
    os.makedirs(final_output_dir, exist_ok=True)

    # Use Trainer.save_model() – it correctly gathers DeepSpeed ZeRO-3 weights
    #    and also saves the tokenizer if one is attached.
    trainer.save_model(final_output_dir)

    # Ensure the tokenizer is saved (Trainer.save_model usually does this,
    #    but we do it explicitly for safety and future compatibility).
    tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(final_output_dir)

    print(f"Fine-tuned model saved to {final_output_dir}")
    return final_output_dir