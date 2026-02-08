"""
Error Analysis Module for News Summarization
Analyzes model failures, identifies error patterns, and suggests improvements.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from datasets import DatasetDict
import evaluate
from collections import Counter, defaultdict
from textblob import TextBlob
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Performs detailed error analysis on model predictions.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize error analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']
        )
        
        # Load evaluation metrics
        self.rouge_metric = evaluate.load('rouge')
        
        # Error categories
        self.error_patterns = defaultdict(list)
    
    def load_model(self, model_path: str, is_peft: bool = True):
        """Load a fine-tuned model."""
        logger.info(f"Loading model from {model_path}")
        
        if is_peft:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config['base_model']
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        return model
    
    def generate_summary(self, model, text: str) -> str:
        """Generate a summary for given text."""
        inputs = self.tokenizer(
            text,
            max_length=self.config['data']['max_input_length'],
            truncation=True,
            return_tensors="pt"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = inputs.to(device)
        
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
    
    def compute_example_score(self, prediction: str, reference: str) -> Dict:
        """Compute ROUGE scores for a single example."""
        result = self.rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True
        )
        return result
    
    def categorize_error(self, 
                        input_text: str, 
                        prediction: str, 
                        reference: str, 
                        rouge_score: float) -> List[str]:
        """
        Categorize the type of error based on analysis.
        
        Error categories:
        - length_mismatch: Summary too short or too long
        - factual_error: Missing key facts from input
        - redundancy: Repetitive content
        - bias: Subjective or biased language
        - coherence: Lacks logical flow
        - coverage: Missing important information
        """
        categories = []
        
        # Length analysis
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        
        if pred_len < ref_len * 0.5:
            categories.append("too_short")
        elif pred_len > ref_len * 1.5:
            categories.append("too_long")
        
        # Repetition detection
        words = prediction.lower().split()
        word_counts = Counter(words)
        avg_word_freq = np.mean(list(word_counts.values()))
        if avg_word_freq > 1.5:
            categories.append("redundancy")
        
        # Bias detection
        blob = TextBlob(prediction)
        if blob.sentiment.subjectivity > 0.6:
            categories.append("high_subjectivity")
        
        # Coverage analysis (simple version)
        # Check if key entities/numbers from input appear in prediction
        input_numbers = set([w for w in input_text.split() if w.isdigit()])
        pred_numbers = set([w for w in prediction.split() if w.isdigit()])
        
        if input_numbers and len(input_numbers & pred_numbers) / len(input_numbers) < 0.5:
            categories.append("missing_key_facts")
        
        # Low ROUGE score
        if rouge_score < 0.2:
            categories.append("low_quality")
        
        return categories if categories else ["other"]
    
    def analyze_errors(self, model, test_dataset: DatasetDict, num_examples: int = 100):
        """
        Perform comprehensive error analysis on test set.
        """
        logger.info(f"Analyzing errors on {num_examples} examples...")
        
        error_examples = []
        all_scores = []
        
        for i, example in enumerate(test_dataset):
            if i >= num_examples:
                break
            
            # Decode texts
            input_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            reference = self.tokenizer.decode(example['labels'], skip_special_tokens=True)
            
            # Generate prediction
            prediction = self.generate_summary(model, input_text)
            
            # Compute scores
            scores = self.compute_example_score(prediction, reference)
            rouge2_score = scores['rouge2']
            
            all_scores.append(rouge2_score)
            
            # Identify poorly performing examples (bottom 20%)
            if rouge2_score < 0.15:  # Threshold for poor performance
                error_categories = self.categorize_error(
                    input_text, prediction, reference, rouge2_score
                )
                
                error_example = {
                    'index': i,
                    'input_text': input_text[:500] + "...",  # Truncate for readability
                    'reference': reference,
                    'prediction': prediction,
                    'rouge2_score': rouge2_score,
                    'error_categories': error_categories
                }
                
                error_examples.append(error_example)
                
                # Track error patterns
                for category in error_categories:
                    self.error_patterns[category].append(error_example)
        
        # Generate error analysis report
        report = self._generate_error_report(error_examples, all_scores)
        
        # Create visualizations
        self._create_error_visualizations(error_examples, all_scores)
        
        return report
    
    def _generate_error_report(self, error_examples: List[Dict], all_scores: List[float]) -> Dict:
        """Generate comprehensive error analysis report."""
        
        # Count error categories
        category_counts = Counter()
        for example in error_examples:
            for category in example['error_categories']:
                category_counts[category] += 1
        
        # Statistics
        report = {
            'total_examples_analyzed': len(all_scores),
            'poor_performance_count': len(error_examples),
            'poor_performance_rate': len(error_examples) / len(all_scores) * 100,
            'avg_rouge2_score': np.mean(all_scores),
            'median_rouge2_score': np.median(all_scores),
            'min_rouge2_score': np.min(all_scores),
            'max_rouge2_score': np.max(all_scores),
            'error_category_distribution': dict(category_counts),
            'top_3_error_categories': category_counts.most_common(3),
            'sample_error_examples': error_examples[:5]  # Top 5 worst
        }
        
        # Save report
        self._save_error_report(report)
        
        return report
    
    def _save_error_report(self, report: Dict):
        """Save error analysis report to file."""
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Save JSON report
        output_file = os.path.join(results_dir, 'error_analysis.json')
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        text_file = os.path.join(results_dir, 'error_analysis.txt')
        
        with open(text_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Examples Analyzed: {report['total_examples_analyzed']}\n")
            f.write(f"Poor Performance Count: {report['poor_performance_count']}\n")
            f.write(f"Poor Performance Rate: {report['poor_performance_rate']:.2f}%\n\n")
            
            f.write(f"Average ROUGE-2 Score: {report['avg_rouge2_score']:.4f}\n")
            f.write(f"Median ROUGE-2 Score: {report['median_rouge2_score']:.4f}\n")
            f.write(f"Score Range: {report['min_rouge2_score']:.4f} - {report['max_rouge2_score']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("ERROR CATEGORY DISTRIBUTION\n")
            f.write("="*80 + "\n")
            for category, count in report['top_3_error_categories']:
                f.write(f"  {category}: {count} occurrences\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("SAMPLE ERROR EXAMPLES\n")
            f.write("="*80 + "\n\n")
            
            for i, example in enumerate(report['sample_error_examples'][:3], 1):
                f.write(f"\nExample {i} (ROUGE-2: {example['rouge2_score']:.4f})\n")
                f.write(f"Error Categories: {', '.join(example['error_categories'])}\n")
                f.write(f"\nReference: {example['reference']}\n")
                f.write(f"\nPrediction: {example['prediction']}\n")
                f.write("-"*80 + "\n")
        
        logger.info(f"Error analysis reports saved to {results_dir}")
    
    def _create_error_visualizations(self, error_examples: List[Dict], all_scores: List[float]):
        """Create visualizations for error analysis."""
        results_dir = os.path.join(self.config['paths']['results_dir'], 'visualizations')
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Score distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0.15, color='red', linestyle='--', label='Poor Performance Threshold')
        ax.set_xlabel('ROUGE-2 Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of ROUGE-2 Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'score_distribution.png'), dpi=300)
        
        # 2. Error category distribution
        category_counts = Counter()
        for example in error_examples:
            for category in example['error_categories']:
                category_counts[category] += 1
        
        if category_counts:
            fig, ax = plt.subplots(figsize=(12, 6))
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            
            ax.barh(categories, counts, color='coral', alpha=0.7)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_title('Error Category Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'error_categories.png'), dpi=300)
        
        plt.close('all')
        logger.info(f"Error visualizations saved to {results_dir}")
    
    def suggest_improvements(self, report: Dict) -> List[str]:
        """
        Suggest improvements based on error analysis.
        """
        suggestions = []
        
        top_errors = dict(report['top_3_error_categories'])
        
        if 'too_short' in top_errors:
            suggestions.append(
                "Increase min_length parameter in generation config to produce longer summaries"
            )
        
        if 'too_long' in top_errors:
            suggestions.append(
                "Decrease max_length or adjust length_penalty to produce more concise summaries"
            )
        
        if 'redundancy' in top_errors:
            suggestions.append(
                "Implement repetition penalty in generation config (e.g., repetition_penalty=1.2)"
            )
        
        if 'high_subjectivity' in top_errors:
            suggestions.append(
                "Fine-tune with more objective examples or add bias penalty during training"
            )
        
        if 'missing_key_facts' in top_errors:
            suggestions.append(
                "Increase training data size or adjust training objectives to emphasize factual accuracy"
            )
        
        if 'low_quality' in top_errors:
            suggestions.append(
                "Consider longer training, higher learning rate, or different base model architecture"
            )
        
        # General suggestions
        suggestions.extend([
            "Implement beam search with higher num_beams for better quality",
            "Use diverse dataset augmentation to improve generalization",
            "Consider ensemble methods combining multiple checkpoints"
        ])
        
        return suggestions


def main():
    """Example usage of error analysis module."""
    from data_preparation import NewsDataPreparation
    
    # Load test dataset
    data_prep = NewsDataPreparation()
    tokenized_datasets = data_prep.load_processed_datasets()
    test_dataset = tokenized_datasets['test']
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer()
    
    # Load model
    model_path = "./models/baseline/checkpoint-1000"  # Replace with actual path
    model = analyzer.load_model(model_path)
    
    # Perform error analysis
    report = analyzer.analyze_errors(model, test_dataset, num_examples=100)
    
    # Get improvement suggestions
    suggestions = analyzer.suggest_improvements(report)
    
    # Print summary
    print("\n=== Error Analysis Summary ===")
    print(f"Poor Performance Rate: {report['poor_performance_rate']:.2f}%")
    print(f"\nTop Error Categories:")
    for category, count in report['top_3_error_categories']:
        print(f"  - {category}: {count}")
    
    print("\n=== Suggested Improvements ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")


if __name__ == "__main__":
    main()