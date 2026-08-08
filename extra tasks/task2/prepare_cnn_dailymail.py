import os
import pandas as pd
from datasets import load_dataset

def prepare_cnn_dailymail(sample_size=1000, seed=42):
    print("--> Loading cnn_dailymail dataset (version 3.0.0) from HuggingFace...")
    # Load the test split
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    
    print(f"--> Shuffling and selecting {sample_size} random samples (seed={seed})...")
    # Shuffle and select the required number of samples
    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    
    # Convert to Pandas DataFrame
    df = sampled_dataset.to_pandas()
    
    # Keep only necessary columns and rename them to match the research methodology
    df = df[['highlights', 'article']]
    df = df.rename(columns={
        'highlights': 'Prompt',
        'article': 'Human_story'
    })
    
    # Clean the text (remove unnecessary newlines if any)
    df['Prompt'] = df['Prompt'].str.replace('\n', ' ').str.strip()
    df['Human_story'] = df['Human_story'].str.replace('\n', ' ').str.strip()
    
    # Create output directory if it doesn't exist
    output_dir = "results/extra/task2/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cnn_dailymail_1k.csv")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Saved {len(df)} articles to {output_path}")
    print("\nData Preview:")
    print(df.head(2).to_string())

if __name__ == "__main__":
    prepare_cnn_dailymail()