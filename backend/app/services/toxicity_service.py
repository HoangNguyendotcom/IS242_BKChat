from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import logging
import onnxruntime
import numpy as np
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicityService:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'toxicity_model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.onnx_session: Optional[onnxruntime.InferenceSession] = None
        
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self._load_default_model()

    def _initialize_model(self):
        """Initialize the model with optimized settings"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Custom model not found at {self.model_path}")
            self._load_default_model()
            return

        # Try to load ONNX model first (most efficient)
        onnx_model_path = os.path.join(self.model_path, "model.onnx")
        if os.path.exists(onnx_model_path):
            logger.info("Loading ONNX model...")
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("ONNX model loaded successfully")
            return

        # Fallback to PyTorch model
        logger.info("Loading PyTorch model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Enable model optimization
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)  # PyTorch 2.0 optimization
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logger.info("PyTorch model loaded successfully")

    def _load_default_model(self):
        """Load the default model with optimizations"""
        logger.info("Loading default model...")
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Default model loaded successfully")

    def check_toxicity(self, text: str) -> bool:
        """
        Check if a message is toxic using the ML model
        Returns True if toxic, False otherwise
        """
        try:
            if self.onnx_session is not None:
                # Use ONNX runtime for inference
                inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
                ort_inputs = {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
                outputs = self.onnx_session.run(None, ort_inputs)
                logits = outputs[0]
                probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
                is_toxic = probabilities[0][1].item() > 0.5
            else:
                # Use PyTorch for inference
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    is_toxic = probabilities[0][1].item() > 0.5

            logger.debug(f"Toxicity check for '{text}': {is_toxic}")
            return is_toxic
        except Exception as e:
            logger.error(f"Error in toxicity check: {str(e)}")
            return False

# Create a singleton instance
toxicity_service = ToxicityService() 