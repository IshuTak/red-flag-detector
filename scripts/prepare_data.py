
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_processor import DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_hate_speech_data():
    
    try:
        df = pd.read_csv('data/raw/hate_speech.csv')
        logger.info(f"Loaded hate speech dataset with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading hate speech dataset: {e}")
        return None

def prepare_go_emotions_data():
    
    try:
        df = pd.read_csv('data/raw/go_emotions.csv')
        logger.info(f"Loaded Go Emotions dataset with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading Go Emotions dataset: {e}")
        return None

def main():
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    
    preprocessor = DataPreprocessor()
    
    
    hate_speech_df = prepare_hate_speech_data()
    emotions_df = prepare_go_emotions_data()
    
    if hate_speech_df is None or emotions_df is None:
        logger.error("Failed to load one or both datasets")
        return
    
    
    logger.info("Processing and merging datasets...")
    
    try:
        
        combined_df = preprocessor.combine_datasets(hate_speech_df, emotions_df)
        
        
        output_path = 'data/processed/processed_combined_dataset.csv'
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")
        
        
        
        
        
        
        logger.info("\nDataset Statistics:")
        logger.info(f"Total samples: {len(combined_df)}")
        logger.info("\nClass distribution:")
        logger.info(combined_df['label'].value_counts())
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    main()
