import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import RobertaTokenizer
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = 128
        
        
        self.negative_emotions = [
            'anger', 'disgust', 'fear', 'sadness', 'disappointment', 
            'disapproval', 'annoyance', 'grief'
        ]
    
    def process_hate_speech_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        processed_df = df.copy()
        
        
        processed_df['cleaned_text'] = processed_df['tweet'].apply(self.clean_text)
        
        
        processed_df['label'] = processed_df['class'].map({
            0: 'red_flag',    # hate speech
            1: 'red_flag',    # offensive language
            2: 'green_flag'   # neither
        })
        
        logger.info(f"Processed hate speech data with {len(processed_df)} rows")
        return processed_df[['cleaned_text', 'label']]
    
    def process_emotions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        processed_df = df.copy()
        
       
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_text)
        
        
        def determine_flag(row):
            
            has_negative = any(row[emotion] == 1 for emotion in self.negative_emotions)
            return 'red_flag' if has_negative else 'green_flag'
        
        processed_df['label'] = processed_df.apply(determine_flag, axis=1)
        
        logger.info(f"Processed emotions data with {len(processed_df)} rows")
        return processed_df[['cleaned_text', 'label']]
    
    def clean_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            return ""
        
        
        text = str(text).lower()
        
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        
        text = re.sub(r'@\w+', '', text)
        
        
        text = re.sub(r'#(\w+)', r'\1', text)
        
        
        text = re.sub(r'^rt\s+', '', text)
        
        
        text = ' '.join(text.split())
        
        return text
    
    def combine_datasets(self, hate_speech_df: pd.DataFrame, 
                        emotions_df: pd.DataFrame) -> pd.DataFrame:
        
        try:
            
            processed_hate_speech = self.process_hate_speech_data(hate_speech_df)
            processed_emotions = self.process_emotions_data(emotions_df)
            
            logger.info(f"Hate speech dataset shape: {processed_hate_speech.shape}")
            logger.info(f"Emotions dataset shape: {processed_emotions.shape}")
            
            
            combined_df = pd.concat(
                [processed_hate_speech, processed_emotions], 
                ignore_index=True
            )
            
            
            combined_df = combined_df.drop_duplicates(subset=['cleaned_text'])
            logger.info(f"Combined shape after removing duplicates: {combined_df.shape}")
            
           
            min_class_count = combined_df['label'].value_counts().min()
            balanced_df = pd.concat([
                combined_df[combined_df['label'] == label].sample(
                    min_class_count, 
                    random_state=42
                )
                for label in combined_df['label'].unique()
            ])
            
            logger.info(f"Final balanced shape: {balanced_df.shape}")
            logger.info("Class distribution:")
            logger.info(balanced_df['label'].value_counts())
            
            return balanced_df
            
        except Exception as e:
            logger.error(f"Error in combining datasets: {e}")
            raise
