"""
LLM Processor Module
------------------
Handles NLP processing of radiology reports using an open-source LLM.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Processes radiology reports using a fine-tuned LLM to extract diagnoses.
    Uses the facebook/bart-large-cnn model as a base, which can be fine-tuned
    for medical text analysis.
    """
    
    # Default model - an open source model from HuggingFace
    DEFAULT_MODEL = "facebook/bart-large-cnn"
    
    # Path to saved fine-tuned model (if available)
    FINE_TUNED_MODEL_PATH = os.path.join(
        Path(__file__).parent.absolute(), "fine_tuned"
    )
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the LLM processor.
        
        Args:
            model_path (str, optional): Path to a fine-tuned model directory.
                                       If None, uses the default model.
        """
        self.model_path = model_path or self.DEFAULT_MODEL
        
        # Check if fine-tuned model exists and use it instead
        if not model_path and os.path.exists(self.FINE_TUNED_MODEL_PATH):
            self.model_path = self.FINE_TUNED_MODEL_PATH
            logger.info(f"Using fine-tuned model from: {self.FINE_TUNED_MODEL_PATH}")
        
        logger.info(f"Initializing LLM with model: {self.model_path}")
        
        # Load model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.nlp_pipeline = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze_report(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze a radiology report to determine if it indicates intracranial hemorrhage.
        
        Args:
            report_text (str): The radiologist report text
            
        Returns:
            dict: Analysis results including hemorrhage detection and confidence
        """
        # Clean the text (remove excessive whitespace, etc.)
        cleaned_text = self._preprocess_text(report_text)
        
        # Create the prompt with instructions for the model
        prompt = self._create_prompt(cleaned_text)
        
        # Generate the analysis
        try:
            result = self._generate_analysis(prompt)
            hemorrhage_detected, confidence = self._interpret_response(result)
            
            return {
                "hemorrhage_detected": hemorrhage_detected,
                "confidence": confidence,
                "model_name": self.model_path,
                "analysis_text": result,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error analyzing report: {e}")
            return {
                "hemorrhage_detected": None,
                "confidence": 0.0,
                "model_name": self.model_path,
                "analysis_text": f"Error: {str(e)}",
                "success": False
            }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess report text for analysis.
        
        Args:
            text (str): Report text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive newlines and whitespace
        cleaned = " ".join(text.split())
        return cleaned
    
    def _create_prompt(self, text: str) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            text (str): Preprocessed report text
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""
        Task: Analyze this head CT radiology report and determine if there is any mention of intracranial hemorrhage.
        
        Report: {text}
        
        First, identify any phrases related to intracranial bleeding or hemorrhage.
        Then, determine if the report is positive or negative for intracranial hemorrhage.
        Answer with 'positive' if any type of intracranial hemorrhage is present, and 'negative' if there is no hemorrhage or if hemorrhage is explicitly ruled out.
        """
        return prompt.strip()
    
    def _generate_analysis(self, prompt: str) -> str:
        """
        Generate text using the NLP pipeline.
        
        Args:
            prompt (str): Formatted prompt
            
        Returns:
            str: Generated response
        """
        result = self.nlp_pipeline(
            prompt,
            max_length=150,
            min_length=20,
            do_sample=False,
            num_return_sequences=1
        )
        
        # Extract and return the generated text
        return result[0]["generated_text"].strip()
    
    def _interpret_response(self, response: str) -> Tuple[Optional[bool], float]:
        """
        Interpret the model's response to determine if hemorrhage is detected.
        
        Args:
            response (str): Model-generated response
            
        Returns:
            tuple: (hemorrhage_detected, confidence)
                  hemorrhage_detected is True if positive, False if negative, None if unclear
                  confidence is a float value between 0.0 and 1.0
        """
        response = response.lower()
        
        # Check for clear positive/negative statements
        if "positive" in response and "negative" not in response:
            return True, 0.9
        elif "negative" in response and "positive" not in response:
            return False, 0.9
        
        # Check for hemorrhage-related terms
        hemorrhage_terms = [
            "hemorrhage", "hematoma", "bleed", "blood", "subdural", 
            "epidural", "subarachnoid", "intraventricular", "intraparenchymal"
        ]
        
        # Check for negation terms
        negation_terms = [
            "no evidence of", "without", "no", "not", "none", "absent"
        ]
        
        # Simple heuristic for detecting hemorrhage mentions and negations
        hemorrhage_mentioned = any(term in response for term in hemorrhage_terms)
        negated = any(term in response for term in negation_terms)
        
        if hemorrhage_mentioned:
            if negated:
                return False, 0.7
            else:
                return True, 0.7
        
        # Unclear result
        return None, 0.3


# Example usage
if __name__ == "__main__":
    # Sample reports
    positive_report = """
    CT HEAD WITHOUT CONTRAST
    
    CLINICAL INDICATION: Fall
    
    TECHNIQUE: Axial images were obtained through the head without intravenous contrast.
    
    COMPARISON: None.
    
    FINDINGS: There is an acute subdural hematoma along the right frontal convexity measuring approximately 5 mm in thickness. No significant mass effect or midline shift. The ventricles are normal in size and configuration. Gray-white matter differentiation is preserved.
    
    IMPRESSION: Acute right frontal subdural hematoma without significant mass effect.
    """
    
    negative_report = """
    CT HEAD WITHOUT CONTRAST
    
    CLINICAL INDICATION: Headache
    
    TECHNIQUE: Axial images were obtained through the head without intravenous contrast.
    
    COMPARISON: None.
    
    FINDINGS: There is no evidence of acute intracranial hemorrhage, mass effect, or midline shift. Ventricles and sulci are normal in size and configuration for age. Gray-white matter differentiation is preserved. No acute infarct is seen.
    
    IMPRESSION: No evidence of acute intracranial hemorrhage or mass effect.
    """
    
    try:
        processor = LLMProcessor()
        
        # Analyze positive report
        positive_result = processor.analyze_report(positive_report)
        print(f"Positive report analysis: {positive_result}")
        
        # Analyze negative report
        negative_result = processor.analyze_report(negative_report)
        print(f"Negative report analysis: {negative_result}")
        
    except Exception as e:
        print(f"Error during initialization or processing: {e}")
