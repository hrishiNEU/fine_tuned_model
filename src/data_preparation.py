"""
Data Preparation Module for News Summarization with Bias Detection
This module handles dataset loading, preprocessing, and splitting.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataPreparation:
    """
    Handles data preparation for news summarization task including:
    - Dataset loading
    - Cleaning and preprocessing
    - Train/validation/test splitting
    - Tokenization
    - Bias keyword annotation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']
        )
        
        # Load bias keywords
        self.bias_keywords = self._load_bias_keywords()
        
    def _load_bias_keywords(self) -> Dict:
        """Load bias detection keywords."""
        bias_config = self.config['bias_detection']
        return bias_config['political_keywords']
    
    def load_dataset(self, dataset_name: str = None) -> DatasetDict:
        """
        Load dataset from Hugging Face or local files.
        
        Recommended datasets:
        - "cnn_dailymail": CNN/DailyMail news articles (recommended)
        - "EdinburghNLP/xsum": Extreme summarization of BBC articles
        """
        if dataset_name is None:
            dataset_name = self.data_config['dataset_name']
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load from Hugging Face with trust_remote_code=True for newer datasets
            if 'dataset_config' in self.data_config:
                dataset = load_dataset(
                    dataset_name, 
                    self.data_config['dataset_config'],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(dataset_name, trust_remote_code=True)
            
            logger.info(f"Dataset loaded successfully. Available splits: {dataset.keys()}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Trying alternative datasets...")
            
            # Try CNN/DailyMail as fallback
            try:
                logger.info("Trying cnn_dailymail dataset...")
                dataset = load_dataset("cnn_dailymail", "3.0.0", trust_remote_code=True)
                logger.info("Successfully loaded cnn_dailymail dataset")
                return dataset
            except Exception as e2:
                logger.error(f"Error loading fallback dataset: {e2}")
                raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere with tokenization
        # Keep basic punctuation for sentence structure
        
        return text.strip()
    
    def detect_bias_indicators(self, text: str) -> Dict:
        """
        Detect potential bias indicators in text.
        Returns counts of left-leaning and right-leaning keywords.
        """
        text_lower = text.lower()
        
        left_count = sum(
            1 for keyword in self.bias_keywords['left_leaning']
            if keyword in text_lower
        )
        
        right_count = sum(
            1 for keyword in self.bias_keywords['right_leaning']
            if keyword in text_lower
        )
        
        return {
            'left_bias_score': left_count,
            'right_bias_score': right_count,
            'bias_detected': left_count > 0 or right_count > 0
        }
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess examples for training.
        Handles tokenization and formatting.
        """
        # Get the article and summary fields (adjust based on dataset)
        # For cnn_dailymail: 'article' and 'highlights'
        # For xsum: 'document' and 'summary'
        
        if 'article' in examples:
            inputs = examples['article']
            targets = examples['highlights']
        elif 'document' in examples:
            inputs = examples['document']
            targets = examples['summary']
        else:
            raise ValueError(f"Unknown dataset format. Available keys: {examples.keys()}")
        
        # Clean texts
        inputs = [self.clean_text(text) for text in inputs]
        targets = [self.clean_text(text) for text in targets]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_config['max_input_length'],
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=self.data_config['max_target_length'],
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        
        # Add bias detection scores
        bias_scores = [self.detect_bias_indicators(text) for text in inputs]
        model_inputs['left_bias_score'] = [score['left_bias_score'] for score in bias_scores]
        model_inputs['right_bias_score'] = [score['right_bias_score'] for score in bias_scores]
        
        return model_inputs
    
    def split_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Split dataset into train, validation, and test sets.
        """
        logger.info("Splitting dataset...")
        
        # If dataset already has train/validation/test splits, use them
        if all(split in dataset for split in ['train', 'validation', 'test']):
            logger.info("Using existing splits")
            return dataset
        
        # Otherwise, create splits from train set
        train_test_split = dataset['train'].train_test_split(
            test_size=self.data_config['val_size'] + self.data_config['test_size'],
            seed=42
        )
        
        val_test_split = train_test_split['test'].train_test_split(
            test_size=self.data_config['test_size'] / (self.data_config['val_size'] + self.data_config['test_size']),
            seed=42
        )
        
        final_dataset = DatasetDict({
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
        
        logger.info(f"Train size: {len(final_dataset['train'])}")
        logger.info(f"Validation size: {len(final_dataset['validation'])}")
        logger.info(f"Test size: {len(final_dataset['test'])}")
        
        return final_dataset
    
    def prepare_datasets(self, sample_size: int = None) -> DatasetDict:
        """
        Complete pipeline: load, clean, split, and tokenize datasets.
        
        Args:
            sample_size: If provided, use only a sample of the data (for testing)
        """
        # Load dataset
        dataset = self.load_dataset()
        
        # Sample if requested (useful for quick testing)
        if sample_size:
            logger.info(f"Sampling {sample_size} examples from each split")
            if 'train' in dataset:
                dataset['train'] = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
            if 'validation' in dataset:
                dataset['validation'] = dataset['validation'].select(range(min(sample_size//2, len(dataset['validation']))))
            if 'test' in dataset:
                dataset['test'] = dataset['test'].select(range(min(sample_size//2, len(dataset['test']))))
        
        # Split dataset
        dataset = self.split_dataset(dataset)
        
        # Tokenize and preprocess
        logger.info("Tokenizing datasets...")
        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        # Save processed datasets
        output_dir = os.path.join(self.data_config['cache_dir'], 'processed')
        os.makedirs(output_dir, exist_ok=True)
        tokenized_datasets.save_to_disk(output_dir)
        logger.info(f"Processed datasets saved to {output_dir}")
        
        return tokenized_datasets
    
    def load_processed_datasets(self) -> DatasetDict:
        """Load previously processed datasets."""
        output_dir = os.path.join(self.data_config['cache_dir'], 'processed')
        logger.info(f"Loading processed datasets from {output_dir}")
        return DatasetDict.load_from_disk(output_dir)


def main():
    """Example usage of the data preparation module."""
    # Initialize data preparation
    data_prep = NewsDataPreparation(config_path='config.yaml')
    
    # Prepare datasets (use sample_size for testing, remove for full dataset)
    tokenized_datasets = data_prep.prepare_datasets(sample_size=1000)
    
    # Print dataset info
    print("\n=== Dataset Information ===")
    print(f"Train size: {len(tokenized_datasets['train'])}")
    print(f"Validation size: {len(tokenized_datasets['validation'])}")
    print(f"Test size: {len(tokenized_datasets['test'])}")
    
    # Show example
    print("\n=== Example from training set ===")
    example = tokenized_datasets['train'][0]
    print(f"Input IDs shape: {len(example['input_ids'])}")
    print(f"Labels shape: {len(example['labels'])}")
    print(f"Left bias score: {example['left_bias_score']}")
    print(f"Right bias score: {example['right_bias_score']}")


if __name__ == "__main__":
    main()