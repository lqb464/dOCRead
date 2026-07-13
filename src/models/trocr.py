from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TrOCRWrapper:
    """
    Wrapper for HuggingFace TrOCR (Transformer-based Optical Character Recognition).
    Excellent for both Printed and Handwritten text recognition.
    """
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
    def predict(self, image: Image.Image) -> str:
        """
        Runs inference on a PIL Image.
        """
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
