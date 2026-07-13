import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.models.crnn import CRNN
from src.utils.metrics import calculate_cer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def train_crnn():
    logger.info("Initializing CRNN Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    imgH = 32
    nc = 1 # Grayscale
    nclass = 37 # 26 letters + 10 digits + 1 blank (for CTC)
    nh = 256
    
    model = CRNN(imgH, nc, nclass, nh).to(device)
    
    # CTC Loss requires specific input shapes: (T, N, C) for predictions
    criterion = nn.CTCLoss(blank=nclass-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    mlflow.set_experiment("DOCRead-CRNN-CTC")
    with mlflow.start_run():
        logger.info("Epoch 1/1 (Smoke Test)")
        
        # Dummy batch of 4 grayscale images (H=32, W=100)
        batch_size = 4
        images = torch.randn(batch_size, nc, imgH, 100).to(device)
        
        # Forward pass
        preds = model(images) # Shape: [Sequence Length, Batch Size, Num Classes]
        
        # Calculate loss (dummy targets)
        T, N, C = preds.size()
        preds_size = torch.full((N,), T, dtype=torch.int32)
        # Dummy target length (e.g. 5 chars per image)
        target_lengths = torch.full((N,), 5, dtype=torch.int32)
        # Dummy targets (flattened for CTC)
        targets = torch.randint(0, nclass-1, (N * 5,), dtype=torch.int32)
        
        # CTC Loss requires log_softmax
        log_preds = preds.log_softmax(2)
        loss = criterion(log_preds, targets, preds_size, target_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"Loss: {loss.item():.4f}")
        mlflow.log_metric("CTC_Loss", loss.item())
        
        # Example CER Calculation
        # In a real loop, you'd decode log_preds using Greedy Search or Beam Search
        # and compare against the actual ground truth text.
        dummy_pred_text = "helllo"
        dummy_true_text = "hello"
        cer = calculate_cer(dummy_pred_text, dummy_true_text)
        logger.info(f"CER: {cer:.4f}")
        mlflow.log_metric("CER", cer)
        
        logger.info("Training complete. (Smoke test finished)")

if __name__ == "__main__":
    train_crnn()
