import os

def save_finetuned_model(trainer, output_dir):
    final_output_dir = os.path.join(output_dir, "fine_tuned_final")
    unwrapped_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    unwrapped_model.save_pretrained(final_output_dir)
    trainer.tokenizer.save_pretrained(final_output_dir)
    print(f"Fine-tuned model saved to {final_output_dir}")
    return final_output_dir