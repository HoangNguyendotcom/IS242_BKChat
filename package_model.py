from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import shutil

def package_model(
    model_path,  # Path to your trained model weights
    tokenizer_path,  # Path to your tokenizer
    output_dir="toxicity_model",  # Output directory for packaged model
    model_type="pytorch"  # or "sklearn" depending on your model type
):
    """
    Package a trained model into Hugging Face format
    
    Args:
        model_path: Path to your trained model weights
        tokenizer_path: Path to your tokenizer
        output_dir: Directory to save the packaged model
        model_type: Type of model ("pytorch" or "sklearn")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == "pytorch":
        # For PyTorch models
        model = torch.load(model_path)
        tokenizer = torch.load(tokenizer_path)
        
        # Save model and tokenizer in Hugging Face format
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    elif model_type == "sklearn":
        # For scikit-learn models
        from joblib import load
        model = load(model_path)
        
        # Convert sklearn model to PyTorch format
        # This is a simplified example - you'll need to adapt this based on your model
        from transformers import PreTrainedModel
        class SklearnToTransformer(PreTrainedModel):
            def __init__(self, sklearn_model):
                super().__init__()
                self.sklearn_model = sklearn_model
            
            def forward(self, input_ids, attention_mask=None):
                # Implement your model's forward pass here
                pass
        
        # Create and save the converted model
        converted_model = SklearnToTransformer(model)
        converted_model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained(output_dir)
    
    print(f"Model packaged successfully in {output_dir}")
    print(f"Contents of {output_dir}:")
    print(os.listdir(output_dir))

if __name__ == "__main__":
    # Example usage
    package_model(
        model_path="path/to/your/model.pt",
        tokenizer_path="path/to/your/tokenizer.pt",
        output_dir="toxicity_model",
        model_type="pytorch"  # or "sklearn"
    ) 