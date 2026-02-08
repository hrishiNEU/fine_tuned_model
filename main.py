"""
Main execution script for News Summarization with Bias Detection
Run this script to execute the complete pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preparation import NewsDataPreparation
from model_training import NewsSummarizationTrainer
from evaluation import ModelEvaluator
from error_analysis import ErrorAnalyzer
from inference import NewsSummarizer, create_gradio_interface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(args):
    """
    Execute the complete pipeline from data prep to inference.
    """
    logger.info("="*80)
    logger.info("STARTING FULL PIPELINE")
    logger.info("="*80)
    
    # Step 1: Data Preparation
    if args.prepare_data:
        logger.info("\n[STEP 1/5] Data Preparation")
        logger.info("-"*80)
        data_prep = NewsDataPreparation()
        
        sample_size = args.sample_size if args.quick_test else None
        tokenized_datasets = data_prep.prepare_datasets(sample_size=sample_size)
        logger.info("✓ Data preparation completed")
    else:
        logger.info("\n[STEP 1/5] Loading Pre-processed Data")
        logger.info("-"*80)
        data_prep = NewsDataPreparation()
        tokenized_datasets = data_prep.load_processed_datasets()
        logger.info("✓ Data loaded")
    
    # Step 2: Model Training
    if args.train:
        logger.info("\n[STEP 2/5] Model Training")
        logger.info("-"*80)
        trainer = NewsSummarizationTrainer()
        
        if args.hyperparameter_search:
            logger.info("Running hyperparameter comparison...")
            results = trainer.run_hyperparameter_comparison(tokenized_datasets)
        else:
            logger.info("Training with baseline configuration...")
            trainer.train_model(
                tokenized_datasets,
                config_name='training_baseline',
                experiment_name='baseline_run'
            )
        logger.info("✓ Training completed")
    else:
        logger.info("\n[STEP 2/5] Skipping Training (using existing model)")
        logger.info("-"*80)
    
    # Step 3: Evaluation
    if args.evaluate:
        logger.info("\n[STEP 3/5] Model Evaluation")
        logger.info("-"*80)
        evaluator = ModelEvaluator()
        
        # Find the actual model path
        model_path = args.model_path
        if not os.path.exists(model_path):
            # Try to find the actual checkpoint in the baseline directory
            base_dir = './models/baseline'
            if os.path.exists(base_dir):
                # Look for checkpoint directories
                checkpoints = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and 'checkpoint' in d.lower()]
                if checkpoints:
                    # Use the latest checkpoint or the one with highest number
                    try:
                        latest = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]) if '-' in x else 0)[-1]
                        model_path = os.path.join(base_dir, latest)
                        logger.info(f"Found checkpoint: {model_path}")
                    except:
                        model_path = os.path.join(base_dir, checkpoints[0])
                        logger.info(f"Using checkpoint: {model_path}")
                else:
                    # Check if there's a baseline_run directory
                    if os.path.exists(os.path.join(base_dir, 'baseline_run')):
                        model_path = os.path.join(base_dir, 'baseline_run')
                        logger.info(f"Using baseline_run: {model_path}")
        
        comparison = evaluator.compare_models(
            model_path,
            tokenized_datasets['test']
        )
        
        logger.info("\n--- Evaluation Results ---")
        logger.info(f"Baseline ROUGE-2: {comparison['baseline']['rouge2']:.4f}")
        logger.info(f"Fine-tuned ROUGE-2: {comparison['fine_tuned']['rouge2']:.4f}")
        logger.info(f"Improvement: {comparison['improvements']['rouge2_improvement']:.2f}%")
        logger.info("✓ Evaluation completed")
    else:
        logger.info("\n[STEP 3/5] Skipping Evaluation")
        logger.info("-"*80)
    
    # Step 4: Error Analysis
    if args.error_analysis:
        logger.info("\n[STEP 4/5] Error Analysis")
        logger.info("-"*80)
        analyzer = ErrorAnalyzer()
        model = analyzer.load_model(args.model_path)
        
        report = analyzer.analyze_errors(
            model,
            tokenized_datasets['test'],
            num_examples=100
        )
        
        suggestions = analyzer.suggest_improvements(report)
        
        logger.info("\n--- Error Analysis Summary ---")
        logger.info(f"Poor Performance Rate: {report['poor_performance_rate']:.2f}%")
        logger.info(f"Top Error Category: {report['top_3_error_categories'][0][0]}")
        logger.info("✓ Error analysis completed")
    else:
        logger.info("\n[STEP 4/5] Skipping Error Analysis")
        logger.info("-"*80)
    
    # Step 5: Inference
    if args.inference:
        logger.info("\n[STEP 5/5] Launching Inference Interface")
        logger.info("-"*80)
        interface = create_gradio_interface(args.model_path)
        interface.launch(
            share=args.share,
            server_name="0.0.0.0" if args.public else "127.0.0.1",
            server_port=args.port
        )
    else:
        logger.info("\n[STEP 5/5] Skipping Inference")
        logger.info("-"*80)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="News Summarization with Bias Detection - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with quick test
  python main.py --all --quick-test
  
  # Prepare data and train model
  python main.py --prepare-data --train
  
  # Run hyperparameter search
  python main.py --train --hyperparameter-search
  
  # Evaluate existing model
  python main.py --evaluate --model-path ./models/baseline/checkpoint-1000
  
  # Launch inference interface
  python main.py --inference --model-path ./models/baseline/checkpoint-1000
  
  # Complete pipeline (data prep → train → evaluate → analyze → inference)
  python main.py --all --model-path ./models/baseline/checkpoint-1000
        """
    )
    
    # Pipeline steps
    parser.add_argument('--all', action='store_true',
                       help='Run all pipeline steps')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Prepare and preprocess datasets')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--error-analysis', action='store_true',
                       help='Perform error analysis')
    parser.add_argument('--inference', action='store_true',
                       help='Launch inference interface')
    
    # Training options
    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Run hyperparameter comparison (trains 4 configurations)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use small dataset sample for quick testing')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for quick testing (default: 1000)')
    
    # Model options
    parser.add_argument('--model-path', type=str,
                       default='./models/baseline/checkpoint-best',
                       help='Path to model for evaluation/inference')
    
    # Inference options
    parser.add_argument('--share', action='store_true',
                       help='Create shareable Gradio link')
    parser.add_argument('--public', action='store_true',
                       help='Make interface publicly accessible')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for Gradio interface (default: 7860)')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.prepare_data = True
        args.train = True
        args.evaluate = True
        args.error_analysis = True
        args.inference = True
    
    # Check if at least one step is specified
    if not any([args.prepare_data, args.train, args.evaluate, 
                args.error_analysis, args.inference]):
        parser.print_help()
        print("\nError: Please specify at least one pipeline step or use --all")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_full_pipeline(args)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()