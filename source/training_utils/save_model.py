import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import os

def save_model_and_tokenizer(trainer, output_dir):
    final_output_dir = os.path.join(output_dir, "final")
    unwrapped_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    unwrapped_model.save_pretrained(final_output_dir)
    trainer.tokenizer.save_pretrained(final_output_dir)
    return final_output_dir