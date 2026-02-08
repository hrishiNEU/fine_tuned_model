"""
Generate a comprehensive results summary for the assignment report.
This script extracts all metrics and creates formatted tables.
"""

import json
import os
from pathlib import Path

def load_results():
    """Load evaluation results from JSON file."""
    results_file = "results/evaluation_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("\nRun evaluation first:")
        print("python main.py --evaluate --model-path models/baseline/checkpoint-4")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def print_table(data):
    """Print a nicely formatted table."""
    # Print header
    print("\n" + "="*100)
    print("PERFORMANCE METRICS TABLE")
    print("="*100)
    print()
    
    # Table header
    print(f"{'Model':<30} | {'ROUGE-1':>10} | {'ROUGE-2':>10} | {'ROUGE-L':>10} | {'BLEU':>10} | {'METEOR':>10}")
    print("-" * 100)
    
    # Print rows
    for row in data:
        print(f"{row['model']:<30} | {row['rouge1']:>10.4f} | {row['rouge2']:>10.4f} | "
              f"{row['rougeL']:>10.4f} | {row['bleu']:>10.4f} | {row['meteor']:>10.4f}")
    
    print("="*100)

def print_markdown_table(data):
    """Print table in Markdown format for report."""
    print("\n" + "="*100)
    print("MARKDOWN FORMAT (Copy to your report)")
    print("="*100)
    print()
    
    print("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |")
    print("|-------|---------|---------|---------|------|--------|")
    
    for row in data:
        print(f"| {row['model']} | {row['rouge1']:.4f} | {row['rouge2']:.4f} | "
              f"{row['rougeL']:.4f} | {row['bleu']:.4f} | {row['meteor']:.4f} |")
    
    print()

def print_latex_table(data):
    """Print table in LaTeX format."""
    print("\n" + "="*100)
    print("LATEX FORMAT (For academic papers)")
    print("="*100)
    print()
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance Comparison of Models}")
    print("\\begin{tabular}{|l|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{BLEU} & \\textbf{METEOR} \\\\")
    print("\\hline")
    
    for row in data:
        print(f"{row['model']} & {row['rouge1']:.4f} & {row['rouge2']:.4f} & "
              f"{row['rougeL']:.4f} & {row['bleu']:.4f} & {row['meteor']:.4f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

def print_improvements(results):
    """Print improvement metrics."""
    print("\n" + "="*100)
    print("IMPROVEMENT ANALYSIS")
    print("="*100)
    print()
    
    improvements = results.get('improvements', {})
    
    print("Fine-tuned Model vs Baseline:")
    print(f"  • ROUGE-1 Improvement: {improvements.get('rouge1_improvement', 0):+.2f}%")
    print(f"  • ROUGE-2 Improvement: {improvements.get('rouge2_improvement', 0):+.2f}%")
    print(f"  • ROUGE-L Improvement: {improvements.get('rougeL_improvement', 0):+.2f}%")
    print(f"  • BLEU Improvement:    {improvements.get('bleu_improvement', 0):+.2f}%")
    print(f"  • METEOR Improvement:  {improvements.get('meteor_improvement', 0):+.2f}%")
    print()

def print_bias_metrics(results):
    """Print bias detection metrics."""
    print("\n" + "="*100)
    print("BIAS DETECTION METRICS")
    print("="*100)
    print()
    
    baseline_bias = results.get('baseline', {}).get('bias_metrics', {})
    fine_tuned_bias = results.get('fine_tuned', {}).get('bias_metrics', {})
    
    print(f"{'Metric':<30} | {'Baseline':>15} | {'Fine-tuned':>15}")
    print("-" * 70)
    print(f"{'Avg Sentiment Polarity':<30} | {baseline_bias.get('avg_sentiment_polarity', 0):>15.4f} | {fine_tuned_bias.get('avg_sentiment_polarity', 0):>15.4f}")
    print(f"{'Avg Subjectivity':<30} | {baseline_bias.get('avg_subjectivity', 0):>15.4f} | {fine_tuned_bias.get('avg_subjectivity', 0):>15.4f}")
    print(f"{'Avg Bias Score':<30} | {baseline_bias.get('avg_bias_score', 0):>15.4f} | {fine_tuned_bias.get('avg_bias_score', 0):>15.4f}")
    print(f"{'% Subjective Summaries':<30} | {baseline_bias.get('pct_subjective', 0):>14.2f}% | {fine_tuned_bias.get('pct_subjective', 0):>14.2f}%")
    print()

def print_examples(results):
    """Print example predictions."""
    print("\n" + "="*100)
    print("EXAMPLE PREDICTIONS (First 3)")
    print("="*100)
    
    baseline_preds = results.get('baseline', {}).get('predictions', [])
    baseline_refs = results.get('baseline', {}).get('references', [])
    fine_tuned_preds = results.get('fine_tuned', {}).get('predictions', [])
    
    for i in range(min(3, len(baseline_preds))):
        print(f"\n--- Example {i+1} ---")
        print(f"\nReference Summary:")
        print(f"  {baseline_refs[i][:200]}...")
        print(f"\nBaseline Prediction:")
        print(f"  {baseline_preds[i][:200]}...")
        print(f"\nFine-tuned Prediction:")
        print(f"  {fine_tuned_preds[i][:200]}...")
        print()

def save_to_file(results):
    """Save formatted results to a text file."""
    output_file = "results/RESULTS_SUMMARY.txt"
    
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # Generate all output
    print_results_summary(results)
    
    # Get the output
    output = sys.stdout.getvalue()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n✅ Results saved to: {output_file}")

def print_results_summary(results):
    """Print complete results summary."""
    
    # Extract data for table
    baseline = results.get('baseline', {})
    fine_tuned = results.get('fine_tuned', {})
    
    table_data = [
        {
            'model': 'Baseline (No Fine-tuning)',
            'rouge1': baseline.get('rouge1', 0),
            'rouge2': baseline.get('rouge2', 0),
            'rougeL': baseline.get('rougeL', 0),
            'bleu': baseline.get('bleu', 0),
            'meteor': baseline.get('meteor', 0)
        },
        {
            'model': 'Fine-tuned (checkpoint-4)',
            'rouge1': fine_tuned.get('rouge1', 0),
            'rouge2': fine_tuned.get('rouge2', 0),
            'rougeL': fine_tuned.get('rougeL', 0),
            'bleu': fine_tuned.get('bleu', 0),
            'meteor': fine_tuned.get('meteor', 0)
        }
    ]
    
    # Print all sections
    print_table(table_data)
    print_improvements(results)
    print_bias_metrics(results)
    print_markdown_table(table_data)
    print_latex_table(table_data)
    print_examples(results)

def main():
    """Main function."""
    print("\n" + "="*100)
    print("NEWS SUMMARIZATION WITH BIAS DETECTION - RESULTS SUMMARY")
    print("="*100)
    
    # Load results
    results = load_results()
    
    if results is None:
        return
    
    # Print summary
    print_results_summary(results)
    
    # Save to file
    save_to_file(results)
    
    print("\n" + "="*100)
    print("SUMMARY COMPLETE")
    print("="*100)
    print("\nFiles generated:")
    print("  • results/RESULTS_SUMMARY.txt - Complete formatted results")
    print("  • results/evaluation_results.json - Raw JSON data")
    print("\nVisualization files:")
    print("  • results/visualizations/model_comparison.png")
    print("  • results/visualizations/improvement_chart.png")
    print("\nUse these in your report and presentation!")
    print()

if __name__ == "__main__":
    main()