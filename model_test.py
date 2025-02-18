import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_model_file(model_path):
    try:
        # Load the model file
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Print type of checkpoint
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        # If it's a dict, print keys
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            
            # If there's a state dict, examine it
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            logger.info(f"State dict keys: {state_dict.keys()}")
            
            # Print shapes of tensors
            for key, tensor in state_dict.items():
                logger.info(f"{key}: {tensor.shape}")
    
    except Exception as e:
        logger.error(f"Error examining model: {e}")

if __name__ == "__main__":
    model_path = "models/bert_model/saved_model/best_model.pth"
    examine_model_file(model_path)