import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def convert_to_onnx(
    model_path: str,
    output_dir: str = "toxicity_model",
    model_name: str = "model.onnx"
):
    """
    Convert a PyTorch model to ONNX format
    
    Args:
        model_path: Path to your trained PyTorch model
        output_dir: Directory to save the ONNX model
        model_name: Name of the output ONNX file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your trained model
    model = torch.load(model_path)
    model.eval()  # Set to evaluation mode
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create dummy input for tracing
    dummy_input = tokenizer(
        "This is a dummy input for ONNX conversion",
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Export to ONNX format
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        os.path.join(output_dir, model_name),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=12
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model converted to ONNX format and saved to {output_dir}")
    print(f"Contents of {output_dir}:")
    print(os.listdir(output_dir))

if __name__ == "__main__":
    # Example usage
    convert_to_onnx(
        model_path="path/to/your/model.pt",
        output_dir="toxicity_model",
        model_name="model.onnx"
    ) 