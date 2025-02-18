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
        
        # Define negative emotions for Go Emotions dataset
        self.negative_emotions = [
            'anger', 'disgust', 'fear', 'sadness', 'disappointment', 
            'disapproval', 'annoyance', 'grief'
        ]
    
    def process_hate_speech_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process hate speech dataset"""
        processed_df = df.copy()
        
        # Clean text from 'tweet' column
        processed_df['cleaned_text'] = processed_df['tweet'].apply(self.clean_text)
        
        # Map class labels to red/green flags
        # class 0: hate speech, 1: offensive language, 2: neither
        processed_df['label'] = processed_df['class'].map({
            0: 'red_flag',    # hate speech
            1: 'red_flag',    # offensive language
            2: 'green_flag'   # neither
        })
        
        logger.info(f"Processed hate speech data with {len(processed_df)} rows")
        return processed_df[['cleaned_text', 'label']]
    
    def process_emotions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Go Emotions dataset"""
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_text)
        
        # Create label based on emotion columns
        def determine_flag(row):
            # Check if any negative emotion is present
            has_negative = any(row[emotion] == 1 for emotion in self.negative_emotions)
            return 'red_flag' if has_negative else 'green_flag'
        
        processed_df['label'] = processed_df.apply(determine_flag, axis=1)
        
        logger.info(f"Processed emotions data with {len(processed_df)} rows")
        return processed_df[['cleaned_text', 'label']]
    
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
    
    def combine_datasets(self, hate_speech_df: pd.DataFrame, 
                        emotions_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and balance datasets"""
        try:
            # Process each dataset
            processed_hate_speech = self.process_hate_speech_data(hate_speech_df)
            processed_emotions = self.process_emotions_data(emotions_df)
            
            logger.info(f"Hate speech dataset shape: {processed_hate_speech.shape}")
            logger.info(f"Emotions dataset shape: {processed_emotions.shape}")
            
            # Combine datasets
            combined_df = pd.concat(
                [processed_hate_speech, processed_emotions], 
                ignore_index=True
            )
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['cleaned_text'])
            logger.info(f"Combined shape after removing duplicates: {combined_df.shape}")
            
            # Balance classes
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
    
    def prepare_data_for_training(self, df: pd.DataFrame, 
                                test_size: float = 0.2) -> Dict:
        """Prepare data for model training"""
        try:
            # Encode labels
            df['encoded_label'] = self.label_encoder.fit_transform(df['label'])
            
            # Convert text to features
            encodings = self.tokenizer(
                df['cleaned_text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Create features and labels tensors
            features = {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            }
            labels = torch.tensor(df['encoded_label'].values)
            
            # Calculate split indices
            split_idx = int(len(df) * (1 - test_size))
            
            # Split data
            train_features = {
                'input_ids': features['input_ids'][:split_idx],
                'attention_mask': features['attention_mask'][:split_idx]
            }
            train_labels = labels[:split_idx]
            
            test_features = {
                'input_ids': features['input_ids'][split_idx:],
                'attention_mask': features['attention_mask'][split_idx:]
            }
            test_labels = labels[split_idx:]
            
            logger.info(f"Training set size: {len(train_labels)}")
            logger.info(f"Test set size: {len(test_labels)}")
            
            return {
                'train': (train_features, train_labels),
                'test': (test_features, test_labels),
                'label_encoder': self.label_encoder
            }
            
        except Exception as e:
            logger.error(f"Error in preparing data: {e}")
            raise
    
    def process_and_prepare_data(self) -> Dict:
        """Main function to process and prepare all data"""
        try:
            # Load datasets
            hate_speech_df = pd.read_csv('data/raw/hate_speech.csv')
            emotions_df = pd.read_csv('data/raw/go_emotions.csv')
            
            logger.info(f"Loaded hate speech dataset with {len(hate_speech_df)} rows")
            logger.info(f"Loaded emotions dataset with {len(emotions_df)} rows")
            
            # Combine datasets
            combined_df = self.combine_datasets(hate_speech_df, emotions_df)
            logger.info("Combined and balanced datasets")
            
            # Prepare for training
            prepared_data = self.prepare_data_for_training(combined_df)
            logger.info("Prepared data for training")
            
            # Save processed data
            combined_df.to_csv('data/processed/processed_combined_dataset.csv', 
                             index=False)
            logger.info("Saved processed dataset")
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise