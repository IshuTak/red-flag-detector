
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any
import logging
from src.utils.pattern_detector import PatternDetector
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlagDetectorModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        
        self.distilroberta = AutoModel.from_pretrained('distilroberta-base')
        
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # classifier.0
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)  # classifier.3
        )
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.distilroberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        
        return self.classifier(pooled_output)

class FlagDetector:
    def __init__(self, model_path: str = None, device: str = None):
        
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        
        self.model = FlagDetectorModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        
        self.pattern_detector = PatternDetector()
        
        
        self.recent_results = []
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        
        try:
            
            checkpoint = torch.load(model_path, map_location=self.device)
            logger.info("Model file loaded successfully")
            
            
            self.model.load_state_dict(checkpoint)
            logger.info(f"Model weights loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_model_prediction(self, text: str) -> Dict[str, Any]:
        
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = probs[0][prediction[0]].item()
        
        return {
            'prediction': "Red Flag" if prediction.item() == 1 else "Green Flag",
            'confidence': confidence * 100
        }
    
    def combine_predictions(self, model_pred: Dict[str, Any], pattern_pred: Dict[str, Any]) -> Dict[str, Any]:
        
        
        final_prediction = None
        final_confidence = 0.0
        reasons = []
        severity = "Low"
        
        
        has_toxic = bool(pattern_pred['toxic_matches'])
        has_positive = bool(pattern_pred['positive_matches'])
        
        
        high_priority_patterns = {
            'manipulation': [
                'nobody will ever love you',
                'no one will ever love you',
                'love you like i do'
            ],
            'threats': [
                'regret',
                'sorry',
                'teach you'
            ],
            'control_tactics': [
                'must do exactly',
                'have to tell me everything',
                'need to know where'
            ]
        }

        
        found_high_priority = False
        for category, patterns in high_priority_patterns.items():
            if category in pattern_pred['toxic_matches']:
                for pattern in patterns:
                    if any(pattern in match.lower() for match in pattern_pred['toxic_matches'][category]):
                        found_high_priority = True
                        severity = "Critical"
                        reasons.append(f"⛔ Critical {category.replace('_', ' ')} pattern detected")

        if found_high_priority:
            final_prediction = "Red Flag"
            final_confidence = 95.0
        elif has_toxic:
            final_prediction = "Red Flag"
            final_confidence = 90.0
            severity = "High"
            reasons.append("⚠️ Concerning patterns detected")
        elif has_positive:
            final_prediction = "Green Flag"
            final_confidence = 85.0
            reasons.extend([
                "✅ Healthy communication pattern detected",
                "💚 Shows respect and understanding"
            ])
        else:
            
            final_prediction = model_pred['prediction']
            final_confidence = model_pred['confidence']
            reasons.append("ℹ️ Based on model analysis")

        
        if has_toxic:
            for category, matches in pattern_pred['toxic_matches'].items():
                category_display = category.replace('_', ' ').title()
                reasons.append(f"📌 {category_display}: {', '.join(matches)}")
                
               
                if category == 'manipulation':
                    reasons.append("   💭 This is a form of emotional manipulation to control behavior")
                elif category == 'threats':
                    reasons.append("   ⚠️ This contains threatening language that may indicate abuse")
                elif category == 'control_tactics':
                    reasons.append("   🚫 This shows attempts to control and monitor behavior")

        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'severity': severity,
            'reasons': reasons,
            'pattern_analysis': {
                'toxic_patterns': pattern_pred['toxic_matches'],
                'positive_patterns': pattern_pred['positive_matches']
            },
            'detailed_analysis': {
                'model_confidence': model_pred['confidence'],
                'pattern_based_confidence': final_confidence,
                'has_toxic_patterns': has_toxic,
                'has_positive_patterns': has_positive,
                'severity_level': severity
            }
        }
    def _calculate_pattern_severity(self, toxic_matches: Dict) -> str:
        
        if not toxic_matches:
            return "None"
            
        
        category_weights = {
            'control_tactics': 3,
            'manipulation': 3,
            'gaslighting': 3,
            'threats': 4,  
            'emotional_abuse': 2,
            'isolation_tactics': 2,
            'financial_abuse': 2,
            'guilt_tripping': 1
        }
        
        
        total_score = 0
        for category, patterns in toxic_matches.items():
            weight = category_weights.get(category, 1)
            pattern_count = len(patterns)
            
            
            if category == 'threats' and any('regret' in p.lower() for p in patterns):
                weight += 1  # Extra weight for explicit threats
            
            total_score += pattern_count * weight
        
       
        if total_score >= 7:
            return "Critical"
        elif total_score >= 5:
            return "High"
        elif total_score >= 3:
            return "Medium"
        else:
            return "Low"
    


    def _contains_control_pattern(self, text: str) -> bool:
       
        control_patterns = [
            r'\b(must|have to|need to)\b.*',
            r'exactly what.*say',
            r'at all times',
            r'tell me where',
            r'need to know where'
        ]
    
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in control_patterns)

    def _assess_pattern_strength(self, toxic_matches: Dict) -> str:
       
        total_patterns = sum(len(matches) for matches in toxic_matches.values())
        if total_patterns >= 3:
            return "Strong"
        elif total_patterns >= 2:
            return "Moderate"
        return "Mild"
    
    def predict(self, text: str) -> Dict[str, Any]:
        
        try:
            
            model_pred = self._get_model_prediction(text)
            
           
            pattern_pred = self.pattern_detector.analyze_message(text)
            
            
            result = self.combine_predictions(model_pred, pattern_pred)
            
           
            result['text'] = text
            
            
            self.recent_results.append(result)
            if len(self.recent_results) > 5:  
                self.recent_results.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': str(e),
                'text': text,
                'prediction': "Error",
                'confidence': 0.0
            }
    
    def analyze_messages(self, messages: List[str]) -> Dict[str, Any]:
        
        try:
            results = []
            red_flags = 0
            green_flags = 0
            total_toxic_patterns = 0

            for message in messages:
                result = self.predict(message)
                results.append(result)
                
                if result['prediction'] == "Red Flag":
                    red_flags += 1
                else:
                    green_flags += 1
                
               
                if 'pattern_analysis' in result and 'toxic_matches' in result['pattern_analysis']:
                    total_toxic_patterns += len(result['pattern_analysis']['toxic_matches'])

            
            severity = self._calculate_severity(red_flags, len(messages))

            return {
                'messages': results,
                'overall_assessment': {
                    'total_messages': len(messages),
                    'red_flags': red_flags,
                    'green_flags': green_flags,
                    'severity': severity,
                    'total_toxic_patterns': total_toxic_patterns
                }
            }

        except Exception as e:
            logger.error(f"Error in message analysis: {e}")
            raise Exception(f"Analysis failed: {str(e)}")

    def _calculate_severity(self, red_flags: int, total: int) -> str:
        
        if total == 0:
            return "Unknown"
            
        ratio = red_flags / total
        
        
        pattern_strength = 0
        for result in self.recent_results[-3:]:  # Consider last 3 results
            if 'pattern_analysis' in result and 'toxic_matches' in result['pattern_analysis']:
                pattern_strength += len(result['pattern_analysis']['toxic_matches'])
        
        
        if ratio >= 0.4 or pattern_strength >= 3:
            return "High"
        elif ratio >= 0.25 or pattern_strength >= 2:
            return "Medium"
        else:
            return "Low"
