"""
Inference Pipeline for News Summarization with Bias Detection
Provides user-friendly interface for using the fine-tuned model.
"""

import os
import yaml
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import PeftModel
from textblob import TextBlob
import gradio as gr
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSummarizer:
    """
    Production-ready inference pipeline for news summarization.
    """
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize the summarizer.
        
        Args:
            model_path: Path to the fine-tuned model
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.inference_config = self.config['inference']
        self.bias_config = self.config['bias_detection']
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']
        )
        self.model = self._load_model(model_path)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {model_path}")
        
        # Check if adapter_config.json exists (PEFT/LoRA model)
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        
        if os.path.exists(adapter_config_path):
            # This is a PEFT model with LoRA adapters
            try:
                logger.info("Loading as PEFT model with LoRA adapters")
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_config['base_model']
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()
            except Exception as e:
                logger.error(f"Error loading PEFT model: {e}")
                logger.info("Falling back to loading as regular model")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            # This is a regular fine-tuned model (full weights saved)
            logger.info("Loading as regular fine-tuned model (no LoRA adapters found)")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Loading base model as fallback")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_config['base_model']
                )
        
        model.eval()
        return model
    
    def summarize(self, article_text: str) -> Dict[str, any]:
        """
        Generate summary for a news article.
        
        Args:
            article_text: The full news article text
            
        Returns:
            Dictionary containing summary and bias analysis
        """
        # Tokenize input
        inputs = self.tokenizer(
            article_text,
            max_length=self.config['data']['max_input_length'],
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=self.inference_config['num_beams'],
                max_length=self.inference_config['max_length'],
                min_length=self.inference_config['min_length'],
                length_penalty=self.inference_config['length_penalty'],
                early_stopping=self.inference_config['early_stopping'],
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Analyze bias
        bias_analysis = self._detect_bias(summary)
        
        return {
            'summary': summary,
            'bias_analysis': bias_analysis,
            'input_length': len(article_text.split()),
            'summary_length': len(summary.split())
        }
    
    def _detect_bias(self, text: str) -> Dict:
        """
        Detect potential bias in generated summary.
        """
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Keyword-based bias detection
        text_lower = text.lower()
        
        left_keywords = self.bias_config['political_keywords']['left_leaning']
        right_keywords = self.bias_config['political_keywords']['right_leaning']
        
        left_count = sum(1 for kw in left_keywords if kw in text_lower)
        right_count = sum(1 for kw in right_keywords if kw in text_lower)
        
        # Determine bias direction
        if left_count > right_count:
            bias_direction = "Left-leaning"
        elif right_count > left_count:
            bias_direction = "Right-leaning"
        else:
            bias_direction = "Neutral"
        
        # Determine objectivity
        is_objective = sentiment.subjectivity < self.bias_config['subjectivity_threshold']
        
        return {
            'sentiment_polarity': round(sentiment.polarity, 3),
            'sentiment_subjectivity': round(sentiment.subjectivity, 3),
            'bias_direction': bias_direction,
            'left_keywords_count': left_count,
            'right_keywords_count': right_count,
            'is_objective': is_objective,
            'objectivity_score': round(1 - sentiment.subjectivity, 3)
        }
    
    def batch_summarize(self, articles: list) -> list:
        """
        Summarize multiple articles efficiently.
        
        Args:
            articles: List of article texts
            
        Returns:
            List of results dictionaries
        """
        results = []
        
        for article in articles:
            result = self.summarize(article)
            results.append(result)
        
        return results


def create_gradio_interface(model_path: str):
    """
    Create a Gradio web interface for the summarizer.
    
    Args:
        model_path: Path to the fine-tuned model
    """
    # Initialize summarizer
    summarizer = NewsSummarizer(model_path)
    
    def summarize_and_analyze(article_text: str) -> Tuple[str, str]:
        """Wrapper function for Gradio interface."""
        if not article_text.strip():
            return "Please enter an article to summarize.", ""
        
        result = summarizer.summarize(article_text)
        
        # Format bias analysis
        bias_info = result['bias_analysis']
        bias_report = f"""
**Bias Analysis:**
- Sentiment Polarity: {bias_info['sentiment_polarity']} (-1 negative, +1 positive)
- Subjectivity: {bias_info['sentiment_subjectivity']} (0 objective, 1 subjective)
- Bias Direction: {bias_info['bias_direction']}
- Objectivity Score: {bias_info['objectivity_score']}
- Is Objective: {'✓ Yes' if bias_info['is_objective'] else '✗ No'}

**Statistics:**
- Original Length: {result['input_length']} words
- Summary Length: {result['summary_length']} words
- Compression Ratio: {result['input_length'] / max(result['summary_length'], 1):.1f}x
        """
        
        return result['summary'], bias_report
    
    # Example articles for testing
    example_articles = [
        ["""
The Federal Reserve announced today a quarter-point increase in interest rates, marking the fifth 
consecutive hike this year as officials continue their battle against inflation. Chair Jerome Powell 
stated that the central bank remains committed to bringing inflation down to its 2% target, despite 
concerns about potential economic slowdown. Markets reacted negatively to the news, with the S&P 500 
dropping 1.2% in afternoon trading. Economists remain divided on whether the Fed's aggressive stance 
will successfully curb inflation without triggering a recession.
        """],
        ["""
A groundbreaking study published in Nature today reveals that a new AI-powered drug discovery platform 
has identified potential treatments for rare diseases in a fraction of the traditional time. The system, 
developed by researchers at Stanford University, uses machine learning algorithms to predict molecular 
interactions and has already shown promising results in early trials. The technology could revolutionize 
pharmaceutical research and dramatically reduce the time and cost required to bring new medications to 
market. Industry experts are calling it a major breakthrough in computational biology.
        """]
    ]
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=summarize_and_analyze,
        inputs=gr.Textbox(
            label="News Article",
            placeholder="Paste your news article here...",
            lines=10
        ),
        outputs=[
            gr.Textbox(label="Summary", lines=5),
            gr.Markdown(label="Bias Analysis & Statistics")
        ],
        title="📰 News Article Summarizer with Bias Detection",
        description="""
        This tool generates concise summaries of news articles and analyzes them for potential bias.
        Simply paste a news article and click Submit to get an objective summary with bias metrics.
        """,
        examples=example_articles
    )
    
    return interface


def main():
    """Example usage of the inference pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="News Summarization Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--interface",
        action="store_true",
        help="Launch Gradio web interface"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Article text to summarize (for CLI mode)"
    )
    
    args = parser.parse_args()
    
    if args.interface:
        # Launch web interface
        logger.info("Launching Gradio interface...")
        interface = create_gradio_interface(args.model_path)
        interface.launch(share=True)
    else:
        # CLI mode
        summarizer = NewsSummarizer(args.model_path)
        
        if args.text:
            article = args.text
        else:
            # Example article
            article = """
            The Supreme Court issued a landmark ruling today on voting rights, striking down 
            provisions of a controversial state law that critics argued would suppress minority 
            turnout. In a 6-3 decision, the justices found that the law violated constitutional 
            protections. Civil rights groups celebrated the decision, while state officials 
            expressed disappointment and vowed to pursue alternative measures to ensure election 
            integrity. The ruling is expected to have significant implications for similar laws 
            in other states ahead of next year's elections.
            """
        
        logger.info("Generating summary...")
        result = summarizer.summarize(article)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(result['summary'])
        
        print("\n" + "="*80)
        print("BIAS ANALYSIS")
        print("="*80)
        bias = result['bias_analysis']
        print(f"Sentiment: {bias['sentiment_polarity']:.3f} (polarity), {bias['sentiment_subjectivity']:.3f} (subjectivity)")
        print(f"Bias Direction: {bias['bias_direction']}")
        print(f"Objectivity Score: {bias['objectivity_score']:.3f}")
        print(f"Is Objective: {'Yes' if bias['is_objective'] else 'No'}")
        
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Original Length: {result['input_length']} words")
        print(f"Summary Length: {result['summary_length']} words")
        print(f"Compression: {result['input_length'] / result['summary_length']:.1f}x")


if __name__ == "__main__":
    main()