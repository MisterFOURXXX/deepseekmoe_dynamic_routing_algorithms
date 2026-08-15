import os
import sys

def save_model_and_tokenizer(trainer, output_dir):
    final_output_dir = os.path.join(output_dir, "final")
    unwrapped_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    unwrapped_model.save_pretrained(final_output_dir)
    trainer.tokenizer.save_pretrained(final_output_dir)
    return final_output_dir