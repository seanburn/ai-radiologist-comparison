"""
Comparison Engine Module
-----------------------
Contains logic to compare radiologist diagnoses with AI model outputs.
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Any, Union

class ComparisonResult(Enum):
    """Enum representing possible comparison results."""
    AGREEMENT = "agreement"
    FALSE_POSITIVE = "false_positive"  # AI detected hemorrhage, radiologist did not
    FALSE_NEGATIVE = "false_negative"  # AI missed hemorrhage, radiologist detected
    INCONCLUSIVE = "inconclusive"  # Unable to determine agreement/disagreement
    ERROR = "error"  # Error in processing

class ComparisonEngine:
    """
    Engine to compare radiologist report diagnoses with AI model outputs.
    """
    
    def __init__(self):
        """Initialize the comparison engine."""
        pass
    
    def compare_diagnoses(self, 
                         radiologist_hemorrhage: Optional[bool], 
                         ai_hemorrhage: Optional[bool]) -> Tuple[ComparisonResult, str]:
        """
        Compare radiologist diagnosis with AI model output.
        
        Args:
            radiologist_hemorrhage (bool or None): True if radiologist detected hemorrhage,
                                                  False if not, None if inconclusive
            ai_hemorrhage (bool or None): True if AI detected hemorrhage,
                                         False if not, None if inconclusive
                                         
        Returns:
            tuple: (ComparisonResult, explanation message)
        """
        # Handle inconclusive or missing data
        if radiologist_hemorrhage is None:
            return (ComparisonResult.INCONCLUSIVE, 
                    "Inconclusive: Unable to determine radiologist's diagnosis.")
            
        if ai_hemorrhage is None:
            return (ComparisonResult.INCONCLUSIVE, 
                    "Inconclusive: Unable to determine AI model's diagnosis.")
        
        # Compare diagnoses
        if radiologist_hemorrhage == ai_hemorrhage:
            if radiologist_hemorrhage:
                return (ComparisonResult.AGREEMENT, 
                        "Agreement: Both radiologist and AI detected intracranial hemorrhage.")
            else:
                return (ComparisonResult.AGREEMENT, 
                        "Agreement: Neither radiologist nor AI detected intracranial hemorrhage.")
        else:
            if ai_hemorrhage and not radiologist_hemorrhage:
                return (ComparisonResult.FALSE_POSITIVE, 
                        "False Positive: AI detected hemorrhage, but radiologist did not.")
            else:  # radiologist_hemorrhage and not ai_hemorrhage
                return (ComparisonResult.FALSE_NEGATIVE, 
                        "False Negative: AI did not detect hemorrhage, but radiologist did.")
    
    def generate_detailed_report(self, 
                               radiologist_data: Dict[str, Any],
                               ai_data: Dict[str, Any],
                               comparison_result: ComparisonResult,
                               message: str) -> Dict[str, Any]:
        """
        Generate a detailed report of the comparison.
        
        Args:
            radiologist_data (dict): Data from the radiologist report
            ai_data (dict): Data from the AI model
            comparison_result (ComparisonResult): Result of the comparison
            message (str): Explanation message
            
        Returns:
            dict: Detailed report
        """
        return {
            "comparison_result": comparison_result.value,
            "explanation": message,
            "radiologist_diagnosis": {
                "hemorrhage_detected": radiologist_data.get("hemorrhage_detected"),
                "report_text": radiologist_data.get("report_text", ""),
                "accession_number": radiologist_data.get("accession_number", ""),
                "study_date": radiologist_data.get("study_date", ""),
            },
            "ai_diagnosis": {
                "hemorrhage_detected": ai_data.get("hemorrhage_detected"),
                "confidence": ai_data.get("confidence", 0.0),
                "model_name": ai_data.get("model_name", ""),
                "model_version": ai_data.get("model_version", ""),
            },
            "study_info": {
                "patient_id": radiologist_data.get("patient_id", ""),
                "study_description": radiologist_data.get("study_description", ""),
                "comparison_timestamp": ai_data.get("prediction_timestamp", ""),
            }
        }


# Example usage
if __name__ == "__main__":
    engine = ComparisonEngine()
    
    # Example 1: Agreement (both negative)
    result, message = engine.compare_diagnoses(False, False)
    print(f"Example 1: {result.value} - {message}")
    
    # Example 2: Agreement (both positive)
    result, message = engine.compare_diagnoses(True, True)
    print(f"Example 2: {result.value} - {message}")
    
    # Example 3: False Positive
    result, message = engine.compare_diagnoses(False, True)
    print(f"Example 3: {result.value} - {message}")
    
    # Example 4: False Negative
    result, message = engine.compare_diagnoses(True, False)
    print(f"Example 4: {result.value} - {message}")
