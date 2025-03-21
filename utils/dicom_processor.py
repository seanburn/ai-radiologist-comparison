"""
DICOM Processing Module
----------------------
Handles extraction and validation of DICOM headers for head CT studies.
"""

import pydicom
import logging
import re

logger = logging.getLogger(__name__)

class DicomProcessor:
    """
    Processes DICOM files to extract metadata and verify head CT studies.
    """
    
    # Common head CT study description patterns
    HEAD_CT_PATTERNS = [
        r"HEAD CT",
        r"HEAD\s+CT\s+W/?O",  # Match "HEAD CT W/O", "HEAD CT WO", etc.
        r"NONCONTRAST\s+HEAD\s+CT",
        r"CT\s+HEAD",
        r"BRAIN\s+CT",
        r"HEAD\s+NCCT",  # Non-contrast CT
        r"CRANIAL\s+CT",
    ]
    
    def __init__(self):
        """Initialize the DICOM processor with compiled regex patterns."""
        self.head_ct_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.HEAD_CT_PATTERNS]
    
    def extract_metadata(self, dicom_file_path):
        """
        Reads a DICOM file and extracts key metadata.
        
        Args:
            dicom_file_path (str): Path to the DICOM file
            
        Returns:
            dict: Dictionary containing DICOM metadata
        """
        try:
            ds = pydicom.dcmread(dicom_file_path)
            
            # Extract key elements
            metadata = {
                "study_description": self._get_dicom_element(ds, (0x0008, 0x1030), ""),
                "series_description": self._get_dicom_element(ds, (0x0008, 0x103E), ""),
                "patient_id": self._get_dicom_element(ds, (0x0010, 0x0020), ""),
                "accession_number": self._get_dicom_element(ds, (0x0008, 0x0050), ""),
                "study_date": self._get_dicom_element(ds, (0x0008, 0x0020), ""),
                "modality": self._get_dicom_element(ds, (0x0008, 0x0060), ""),
                "manufacturer": self._get_dicom_element(ds, (0x0008, 0x0070), ""),
                "institution_name": self._get_dicom_element(ds, (0x0008, 0x0080), ""),
                "referring_physician": self._get_dicom_element(ds, (0x0008, 0x0090), ""),
            }
            
            # Add validation
            metadata["is_head_ct"] = self.is_head_ct(metadata)
            metadata["is_ct"] = metadata.get("modality", "").upper() == "CT"
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error reading DICOM file: {e}")
            return {"error": str(e)}
    
    def _get_dicom_element(self, dataset, tag, default=""):
        """
        Safely extract a DICOM element value.
        
        Args:
            dataset: pydicom dataset
            tag: DICOM tag tuple (group, element)
            default: Default value if the tag doesn't exist
            
        Returns:
            Value of the DICOM element or default
        """
        try:
            if tag in dataset:
                return str(dataset[tag].value)
            return default
        except Exception:
            return default
    
    def is_head_ct(self, metadata):
        """
        Check if the study is a head CT based on study description.
        
        Args:
            metadata (dict): DICOM metadata dictionary
            
        Returns:
            bool: True if the study is a head CT, False otherwise
        """
        # First check if it's a CT
        if metadata.get("modality", "").upper() != "CT":
            return False
        
        # Check study and series descriptions
        study_desc = metadata.get("study_description", "").upper()
        series_desc = metadata.get("series_description", "").upper()
        
        # Combined description for pattern matching
        combined_desc = f"{study_desc} {series_desc}"
        
        for pattern in self.head_ct_patterns:
            if pattern.search(study_desc) or pattern.search(series_desc):
                return True
                
        return False
    
    def validate_head_ct(self, dicom_file_path):
        """
        Convenient method to check if a DICOM file is a head CT.
        
        Args:
            dicom_file_path (str): Path to the DICOM file
            
        Returns:
            bool: True if the DICOM is a head CT, False otherwise
        """
        metadata = self.extract_metadata(dicom_file_path)
        return metadata.get("is_head_ct", False)


# Usage example
if __name__ == "__main__":
    processor = DicomProcessor()
    # Example path - adjust as needed
    sample_path = "../data/sample.dcm"
    try:
        metadata = processor.extract_metadata(sample_path)
        print(f"Study description: {metadata.get('study_description')}")
        print(f"Is head CT: {metadata.get('is_head_ct')}")
    except FileNotFoundError:
        print("Sample file not found. This is just an example.")
