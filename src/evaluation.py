"""
Evaluation Module for News Summarization with Bias Detection
Comprehensive evaluation including ROUGE, BLEU, bias detection, and baseline comparison.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel
from datasets import DatasetDict
import evaluate
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for news summarization models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.eval_config = self.config['evaluation']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']
        )
        
        # Load evaluation metrics
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
        self.meteor_metric = evaluate.load('meteor')
        
        # Results storage
        self.results = {}
    
    def load_model(self, model_path: str, is_peft: bool = True):
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
            is_peft: Whether the model uses PEFT (LoRA)
        """
        logger.info(f"Loading model from {model_path}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            # Try to find actual checkpoint
            base_dir = os.path.dirname(model_path)
            if os.path.exists(base_dir):
                checkpoints = [d for d in os.listdir(base_dir) if d.startswith('checkpoint-')]
                if checkpoints:
                    # Use the latest checkpoint
                    latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
                    model_path = os.path.join(base_dir, latest)
                    logger.info(f"Using checkpoint: {model_path}")
        
        if is_peft:
            try:
                # Load base model
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_config['base_model']
                )
                # Load PEFT adapter
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()  # Merge LoRA weights
            except Exception as e:
                logger.warning(f"Failed to load as PEFT model: {e}")
                logger.info("Trying to load as regular model...")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        return model
    
    def load_baseline_model(self):
        """Load the baseline (non-fine-tuned) model for comparison."""
        logger.info("Loading baseline model (no fine-tuning)")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config['base_model']
        )
        return model
    
    def generate_summary(self, model, text: str) -> str:
        """Generate a summary for given text."""
        inputs = self.tokenizer(
            text,
            max_length=self.config['data']['max_input_length'],
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = inputs.to(device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                num_beams=self.config['inference']['num_beams'],
                max_length=self.config['inference']['max_length'],
                min_length=self.config['inference']['min_length'],
                length_penalty=self.config['inference']['length_penalty'],
                early_stopping=self.config['inference']['early_stopping']
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def detect_bias(self, text: str) -> Dict:
        """
        Detect potential bias in text using sentiment analysis and keyword matching.
        """
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Keyword-based bias detection
        bias_keywords = self.config['bias_detection']['political_keywords']
        text_lower = text.lower()
        
        left_count = sum(1 for kw in bias_keywords['left_leaning'] if kw in text_lower)
        right_count = sum(1 for kw in bias_keywords['right_leaning'] if kw in text_lower)
        
        return {
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'left_keywords': left_count,
            'right_keywords': right_count,
            'bias_score': abs(left_count - right_count),
            'is_subjective': sentiment.subjectivity > self.config['bias_detection']['subjectivity_threshold']
        }
    
    def evaluate_model(self, 
                      model, 
                      test_dataset: DatasetDict,
                      model_name: str = "fine-tuned") -> Dict:
        """
        Comprehensive evaluation of a model on test dataset.
        """
        logger.info(f"Evaluating {model_name} model...")
        
        predictions = []
        references = []
        bias_scores = []
        
        # Generate summaries for test set
        for i, example in enumerate(tqdm(test_dataset, desc="Generating summaries")):
            # Decode input text
            input_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            reference = self.tokenizer.decode(example['labels'], skip_special_tokens=True)
            
            # Generate prediction
            prediction = self.generate_summary(model, input_text)
            
            predictions.append(prediction)
            references.append(reference)
            
            # Detect bias
            bias_info = self.detect_bias(prediction)
            bias_scores.append(bias_info)
            
            # Limit evaluation size for speed (remove in production)
            if i >= 99:  # Evaluate first 100 examples
                break
        
        # Compute ROUGE scores
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # Compute BLEU score
        formatted_refs = [[ref] for ref in references]
        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=formatted_refs
        )
        
        # Compute METEOR score
        meteor_results = self.meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        
        # Aggregate bias scores
        avg_bias = {
            'avg_sentiment_polarity': np.mean([b['sentiment_polarity'] for b in bias_scores]),
            'avg_subjectivity': np.mean([b['sentiment_subjectivity'] for b in bias_scores]),
            'avg_bias_score': np.mean([b['bias_score'] for b in bias_scores]),
            'pct_subjective': np.mean([b['is_subjective'] for b in bias_scores]) * 100
        }
        
        # Combine all metrics
        results = {
            'model_name': model_name,
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'bleu': bleu_results['bleu'],
            'meteor': meteor_results['meteor'],
            'bias_metrics': avg_bias,
            'num_samples': len(predictions),
            'predictions': predictions[:10],  # Store first 10 for analysis
            'references': references[:10]
        }
        
        return results
    
    def compare_models(self, 
                      fine_tuned_model_path: str,
                      test_dataset: DatasetDict) -> Dict:
        """
        Compare fine-tuned model with baseline.
        """
        logger.info("Starting model comparison...")
        
        # Evaluate baseline model
        baseline_model = self.load_baseline_model()
        baseline_results = self.evaluate_model(
            baseline_model,
            test_dataset,
            model_name="baseline"
        )
        
        # Evaluate fine-tuned model
        fine_tuned_model = self.load_model(fine_tuned_model_path)
        fine_tuned_results = self.evaluate_model(
            fine_tuned_model,
            test_dataset,
            model_name="fine-tuned"
        )
        
        # Calculate improvements
        improvements = {
            'rouge1_improvement': (fine_tuned_results['rouge1'] - baseline_results['rouge1']) * 100,
            'rouge2_improvement': (fine_tuned_results['rouge2'] - baseline_results['rouge2']) * 100,
            'rougeL_improvement': (fine_tuned_results['rougeL'] - baseline_results['rougeL']) * 100,
            'bleu_improvement': (fine_tuned_results['bleu'] - baseline_results['bleu']) * 100,
            'meteor_improvement': (fine_tuned_results['meteor'] - baseline_results['meteor']) * 100
        }
        
        comparison = {
            'baseline': baseline_results,
            'fine_tuned': fine_tuned_results,
            'improvements': improvements
        }
        
        # Save results
        self._save_results(comparison)
        
        # Create visualizations
        self._create_comparison_plots(comparison)
        
        return comparison
    
    def _save_results(self, results: Dict):
        """Save evaluation results to file."""
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'evaluation_results.json')
        
        # Remove non-serializable items
        saveable_results = results.copy()
        for key in ['baseline', 'fine_tuned']:
            if key in saveable_results:
                saveable_results[key] = {
                    k: v for k, v in saveable_results[key].items()
                    if k not in ['predictions', 'references']
                }
        
        with open(output_file, 'w') as f:
            json.dump(saveable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _create_comparison_plots(self, comparison: Dict):
        """Create visualization comparing baseline and fine-tuned models."""
        results_dir = os.path.join(self.config['paths']['results_dir'], 'visualizations')
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract metrics for plotting
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']
        baseline_scores = [comparison['baseline'][m] for m in metrics]
        fine_tuned_scores = [comparison['fine_tuned'][m] for m in metrics]
        
        # Create comparison bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, fine_tuned_scores, width, label='Fine-tuned', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Scores', fontsize=12)
        ax.set_title('Model Performance Comparison: Baseline vs Fine-tuned', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300)
        logger.info(f"Comparison plot saved to {results_dir}")
        
        # Create improvement percentage plot
        fig, ax = plt.subplots(figsize=(10, 6))
        improvements = [comparison['improvements'][f'{m}_improvement'] for m in metrics]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        ax.barh(metrics, improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_title('Performance Improvement: Fine-tuned vs Baseline', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'improvement_chart.png'), dpi=300)
        
        plt.close('all')


def main():
    """Example usage of the evaluation module."""
    from data_preparation import NewsDataPreparation
    
    # Load test dataset
    data_prep = NewsDataPreparation()
    tokenized_datasets = data_prep.load_processed_datasets()
    test_dataset = tokenized_datasets['test']
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    # Replace with your actual model path
    fine_tuned_model_path = "./models/baseline/checkpoint-1000"
    
    comparison = evaluator.compare_models(
        fine_tuned_model_path,
        test_dataset
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"\nBaseline Model:")
    print(f"  ROUGE-1: {comparison['baseline']['rouge1']:.4f}")
    print(f"  ROUGE-2: {comparison['baseline']['rouge2']:.4f}")
    print(f"  ROUGE-L: {comparison['baseline']['rougeL']:.4f}")
    print(f"  BLEU: {comparison['baseline']['bleu']:.4f}")
    
    print(f"\nFine-tuned Model:")
    print(f"  ROUGE-1: {comparison['fine_tuned']['rouge1']:.4f}")
    print(f"  ROUGE-2: {comparison['fine_tuned']['rouge2']:.4f}")
    print(f"  ROUGE-L: {comparison['fine_tuned']['rougeL']:.4f}")
    print(f"  BLEU: {comparison['fine_tuned']['bleu']:.4f}")
    
    print(f"\nImprovements:")
    for metric, value in comparison['improvements'].items():
        print(f"  {metric}: {value:+.2f}%")


if __name__ == "__main__":
    main()