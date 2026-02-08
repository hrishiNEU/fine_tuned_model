"""
Model Training Module for News Summarization with Bias Detection
Handles model loading, fine-tuning with different configurations, and checkpointing.
"""

import os
import yaml
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import DatasetDict
import evaluate
import logging
from typing import Dict
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSummarizationTrainer:
    """
    Manages the fine-tuning process for news summarization models.
    Supports multiple hyperparameter configurations and PEFT methods.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.data_config = self.config['data']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']
        )
        
        # Load evaluation metrics
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
        
        # Initialize wandb if configured
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                config=self.config
            )
    
    def load_base_model(self):
        """Load the pre-trained base model."""
        logger.info(f"Loading base model: {self.model_config['base_model']}")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config['base_model'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        return model
    
    def setup_peft_model(self, model):
        """
        Configure PEFT (Parameter-Efficient Fine-Tuning) using LoRA.
        This significantly reduces memory usage and training time.
        """
        if not self.model_config['use_peft']:
            logger.info("PEFT disabled, using full fine-tuning")
            return model
        
        logger.info("Setting up LoRA for parameter-efficient fine-tuning")
        
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics (ROUGE, BLEU) for the model.
        """
        predictions, labels = eval_pred
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert to numpy array if needed
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Get argmax for predictions if they are logits (3D)
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        rouge_result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Compute BLEU score
        # Format references as list of lists for BLEU
        formatted_refs = [[label] for label in decoded_labels]
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=formatted_refs
        )
        
        # Combine metrics
        result = {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            'bleu': bleu_result['bleu']
        }
        
        return result
    
    def train_model(self, 
                   tokenized_datasets: DatasetDict, 
                   config_name: str = "training_baseline",
                   experiment_name: str = None):
        """
        Train the model with specified configuration.
        
        Args:
            tokenized_datasets: Preprocessed and tokenized datasets
            config_name: Name of training configuration in config.yaml
            experiment_name: Optional name for this training run
        """
        logger.info(f"Starting training with configuration: {config_name}")
        
        # Load model
        model = self.load_base_model()
        
        # Apply PEFT if configured
        model = self.setup_peft_model(model)
        
        # Get training configuration
        training_config = self.config[config_name]
        
        # Update experiment name in output directory
        if experiment_name:
            training_config['output_dir'] = os.path.join(
                training_config['output_dir'], 
                experiment_name
            )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            fp16=training_config['fp16'] and torch.cuda.is_available(),
            eval_strategy=training_config['eval_strategy'],  # Changed from evaluation_strategy
            report_to="wandb" if self.config['logging']['use_wandb'] else "none",
            run_name=experiment_name or config_name
        )
        
        # Data collator for padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {training_config['output_dir']}")
        trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer, model
    
    def run_hyperparameter_comparison(self, tokenized_datasets: DatasetDict):
        """
        Train models with different hyperparameter configurations and compare results.
        """
        logger.info("Running hyperparameter comparison...")
        
        configs = ['training_baseline', 'training_config1', 'training_config2', 'training_config3']
        results = {}
        
        for config_name in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training with: {config_name}")
            logger.info(f"{'='*60}\n")
            
            try:
                trainer, model = self.train_model(
                    tokenized_datasets,
                    config_name=config_name,
                    experiment_name=config_name.replace('training_', '')
                )
                
                # Get final evaluation metrics
                eval_result = trainer.evaluate()
                results[config_name] = eval_result
                
                logger.info(f"Results for {config_name}:")
                logger.info(f"ROUGE-1: {eval_result['eval_rouge1']:.4f}")
                logger.info(f"ROUGE-2: {eval_result['eval_rouge2']:.4f}")
                logger.info(f"ROUGE-L: {eval_result['eval_rougeL']:.4f}")
                logger.info(f"BLEU: {eval_result['eval_bleu']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training with {config_name}: {e}")
                results[config_name] = None
        
        # Save comparison results
        self._save_comparison_results(results)
        
        return results
    
    def _save_comparison_results(self, results: Dict):
        """Save hyperparameter comparison results."""
        import json
        
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, 'hyperparameter_comparison.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_file}")


def main():
    """Example usage of the training module."""
    from data_preparation import NewsDataPreparation
    
    # Load processed datasets
    data_prep = NewsDataPreparation()
    try:
        tokenized_datasets = data_prep.load_processed_datasets()
    except:
        logger.info("Processed datasets not found. Preparing new datasets...")
        tokenized_datasets = data_prep.prepare_datasets(sample_size=1000)
    
    # Initialize trainer
    trainer = NewsSummarizationTrainer()
    
    # Option 1: Train with single configuration
    trainer.train_model(
        tokenized_datasets,
        config_name='training_baseline',
        experiment_name='baseline_run'
    )
    
    # Option 2: Run hyperparameter comparison
    # results = trainer.run_hyperparameter_comparison(tokenized_datasets)


if __name__ == "__main__":
    main()