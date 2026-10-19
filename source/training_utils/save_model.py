import os
import sys

sys.path.append("..") 

def save_model_and_tokenizer(trainer, output_dir):
    """
    Save the training model and tokenizer.
    """
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Save the model (DeepSpeed-aware if using DeepSpeed)
    trainer.save_model(final_output_dir)
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(final_output_dir)

    print(f"\nModel and tokenizer successfully saved to: {final_output_dir}")
    return final_output_dir