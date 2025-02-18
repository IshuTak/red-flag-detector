import re
import torch
from typing import List, Dict, Any
from transformers import RobertaTokenizer

class TextProcessor:
    def __init__(self, max_length: int = 128):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet) indicators
        text = re.sub(r'^rt\s+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using RoBERTa tokenizer"""
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts"""
        # Clean and tokenize all texts
        encodings = [self.tokenize_text(text) for text in texts]
        
        # Combine into batches
        return {
            'input_ids': torch.cat([enc['input_ids'] for enc in encodings], dim=0),
            'attention_mask': torch.cat([enc['attention_mask'] for enc in encodings], dim=0)
        }

    def decode_predictions(self, predictions: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert model predictions to human-readable format"""
        probabilities = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            pred_idx = pred.item()
            confidence = prob[pred_idx].item()
            
            results.append({
                'flag_type': 'red_flag' if pred_idx == 1 else 'green_flag',
                'confidence': confidence,
                'probabilities': {
                    'green_flag': prob[0].item(),
                    'red_flag': prob[1].item()
                }
            })
            
        return results