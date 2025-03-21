"""
Fine-tuning Module
----------------
Scripts for fine-tuning the LLM on medical text data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class RadiologyReportDataset(Dataset):
    """Dataset for fine-tuning the model on radiology reports."""
    
    def __init__(self, reports: List[str], labels: List[str], tokenizer):
        """
        Initialize the dataset.
        
        Args:
            reports: List of radiology report texts
            labels: List of corresponding labels ("positive" or "negative")
            tokenizer: Tokenizer for the model
        """
        self.tokenizer = tokenizer
        self.reports = reports
        self.labels = labels
        
    def __len__(self):
        return len(self.reports)
    
    def __getitem__(self, idx):
        report = self.reports[idx]
        label = self.labels[idx]
        
        # Create prompt similar to what we'll use in inference
        prompt = f"""
        Task: Analyze this head CT radiology report and determine if there is any mention of intracranial hemorrhage.
        
        Report: {report}
        
        First, identify any phrases related to intracranial bleeding or hemorrhage.
        Then, determine if the report is positive or negative for intracranial hemorrhage.
        Answer with 'positive' if any type of intracranial hemorrhage is present, and 'negative' if there is no hemorrhage or if hemorrhage is explicitly ruled out.
        """
        prompt = prompt.strip()
        
        # Tokenize inputs and labels
        inputs = self.tokenizer(
            prompt, 
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target = self.tokenizer(
            label,
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }


def prepare_data(data_file: str, test_size: float = 0.2) -> Dict[str, HFDataset]:
    """
    Prepare and split the data for fine-tuning.
    
    Args:
        data_file: Path to the CSV file with reports and labels
        test_size: Proportion of data to use for testing
        
    Returns:
        dict: Dictionary with train and test datasets
    """
    # Load data
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records from {data_file}")
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        raise
    
    # Ensure required columns exist
    required_cols = ["report_text", "hemorrhage_present"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data file must contain columns: {required_cols}")
    
    # Convert labels to strings
    df["label"] = df["hemorrhage_present"].apply(
        lambda x: "positive" if x else "negative"
    )
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    
    logger.info(f"Training data: {len(train_df)} samples")
    logger.info(f"Testing data: {len(test_df)} samples")
    
    # Convert to HuggingFace datasets
    train_data = HFDataset.from_pandas(train_df)
    test_data = HFDataset.from_pandas(test_df)
    
    return {"train": train_data, "test": test_data}


def fine_tune_model(
    model_name: str = "facebook/bart-large-cnn",
    data_file: str = "../data/fine_tuning_data/reports.csv",
    output_dir: str = "./fine_tuned",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """
    Fine-tune the LLM on radiology report data.
    
    Args:
        model_name: Base model to use
        data_file: Path to the CSV file with reports and labels
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    logger.info(f"Starting fine-tuning process for model: {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load and prepare data
    datasets = prepare_data(data_file)
    
    # Create torch datasets
    train_data = RadiologyReportDataset(
        datasets["train"]["report_text"],
        datasets["train"]["label"],
        tokenizer
    )
    test_data = RadiologyReportDataset(
        datasets["test"]["report_text"],
        datasets["test"]["label"],
        tokenizer
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="none",  # Disable reporting to third-party services
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Fine-tune the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning complete!")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example code to fine-tune the model - would need a real dataset
    print("This script provides functions to fine-tune the LLM on radiology reports.")
    print("You would need to prepare a CSV file with 'report_text' and 'hemorrhage_present' columns.")
    print("Example usage: fine_tune_model(data_file='path/to/data.csv', epochs=5)")
    
    # Uncomment to run fine-tuning with real data
    # fine_tune_model(
    #     data_file="../data/fine_tuning_data/reports.csv",
    #     epochs=5,
    #     batch_size=4
    # )
