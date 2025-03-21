"""
API Module
---------
Implements the Flask API for the AI vs Radiologist comparison system.
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import modules
from utils.dicom_processor import DicomProcessor
from utils.ris_connector import RisConnector
from utils.comparison_engine import ComparisonEngine
from models.llm_processor import LLMProcessor

# Initialize the Flask app
app = Flask(__name__)

# Initialize modules
dicom_processor = DicomProcessor()
ris_connector = RisConnector()
comparison_engine = ComparisonEngine()

# LLM processor is initialized on demand to save resources
llm_processor = None

# In-memory store for real-time monitoring
# In a production environment, this should be replaced with a proper database
real_time_monitoring = {
    "studies": [],  # List of studies being monitored
    "latest_results": [],  # Latest comparison results
    "statistics": {
        "total_studies": 0,
        "agreement_count": 0,
        "false_positive_count": 0,
        "false_negative_count": 0
    }
}

# Create a lock for thread-safe access to real-time monitoring data
monitoring_lock = threading.Lock()

def get_llm_processor():
    """Get or initialize the LLM processor."""
    global llm_processor
    if llm_processor is None:
        llm_processor = LLMProcessor()
    return llm_processor

def update_monitoring_statistics(comparison_result: Dict[str, Any]):
    """Update the monitoring statistics with a new comparison result."""
    with monitoring_lock:
        # Update study counts
        real_time_monitoring["statistics"]["total_studies"] += 1
        
        # Update result counts based on comparison result
        result_type = comparison_result.get("comparison_result")
        if result_type == "agreement":
            real_time_monitoring["statistics"]["agreement_count"] += 1
        elif result_type == "false_positive":
            real_time_monitoring["statistics"]["false_positive_count"] += 1
        elif result_type == "false_negative":
            real_time_monitoring["statistics"]["false_negative_count"] += 1
        
        # Add to latest results, keeping only most recent 100
        real_time_monitoring["latest_results"].insert(0, {
            "timestamp": datetime.now().isoformat(),
            "study_info": comparison_result.get("study_info", {}),
            "comparison_result": result_type,
            "explanation": comparison_result.get("explanation", "")
        })
        
        # Keep only the 100 most recent results
        if len(real_time_monitoring["latest_results"]) > 100:
            real_time_monitoring["latest_results"] = real_time_monitoring["latest_results"][:100]

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route("/api/analyze-report", methods=["POST"])
def analyze_report():
    """Analyze a radiology report for hemorrhage detection."""
    data = request.json
    
    if not data or "report_text" not in data:
        return jsonify({"error": "Missing required field: report_text"}), 400
    
    report_text = data["report_text"]
    
    try:
        # Get LLM processor instance
        processor = get_llm_processor()
        
        # Analyze the report
        result = processor.analyze_report(report_text)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing report: {str(e)}")
        return jsonify({"error": f"Failed to analyze report: {str(e)}"}), 500

@app.route("/api/validate-dicom", methods=["POST"])
def validate_dicom():
    """Validate a DICOM file to ensure it's a head CT study."""
    data = request.json
    
    if not data or "dicom_path" not in data:
        return jsonify({"error": "Missing required field: dicom_path"}), 400
    
    dicom_path = data["dicom_path"]
    
    try:
        # Extract metadata
        metadata = dicom_processor.extract_metadata(dicom_path)
        
        # Validate if it's a head CT
        is_head_ct = dicom_processor.is_head_ct(metadata)
        
        return jsonify({
            "is_head_ct": is_head_ct,
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Error validating DICOM: {str(e)}")
        return jsonify({"error": f"Failed to validate DICOM: {str(e)}"}), 500

@app.route("/api/get-report", methods=["POST"])
def get_report():
    """Retrieve a report from the RIS system."""
    data = request.json
    
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Check for required fields
    if "accession_number" not in data and "patient_id" not in data:
        return jsonify({"error": "Missing required field: accession_number or patient_id"}), 400
    
    try:
        # Get report from RIS
        if "accession_number" in data:
            report_data = ris_connector.get_report_by_accession(data["accession_number"])
        else:
            study_date = data.get("study_date")  # Optional
            report_data = ris_connector.get_report_by_patient_id(data["patient_id"], study_date)
        
        if report_data:
            return jsonify(report_data)
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving report: {str(e)}")
        return jsonify({"error": f"Failed to retrieve report: {str(e)}"}), 500

@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare radiologist diagnosis with AI model output."""
    data = request.json
    
    if not data or "report_data" not in data or "ai_data" not in data:
        return jsonify({"error": "Missing required fields: report_data and ai_data"}), 400
    
    report_data = data["report_data"]
    ai_data = data["ai_data"]
    
    # Validate report data
    if "report_text" not in report_data:
        return jsonify({"error": "Missing required field: report_text in report_data"}), 400
    
    # Validate AI data
    if "hemorrhage_detected" not in ai_data:
        return jsonify({"error": "Missing required field: hemorrhage_detected in ai_data"}), 400
    
    try:
        # Get LLM processor instance
        processor = get_llm_processor()
        
        # First, analyze the radiologist report
        radiologist_analysis = processor.analyze_report(report_data["report_text"])
        
        # Then, prepare the AI diagnosis data
        ai_diagnosis = {
            "hemorrhage_detected": ai_data["hemorrhage_detected"],
            "confidence": ai_data.get("confidence", 0.0),
            "model_name": ai_data.get("model_name", "Unknown AI Model"),
            "prediction_timestamp": ai_data.get("prediction_timestamp", datetime.now().isoformat())
        }
        
        # Next, validate with DICOM if provided
        dicom_validation = None
        if "dicom_path" in data:
            try:
                dicom_metadata = dicom_processor.extract_metadata(data["dicom_path"])
                is_head_ct = dicom_processor.is_head_ct(dicom_metadata)
                dicom_validation = {
                    "is_head_ct": is_head_ct,
                    "metadata": dicom_metadata
                }
            except Exception as e:
                logger.warning(f"DICOM validation failed: {str(e)}")
                dicom_validation = {
                    "is_head_ct": False,
                    "error": str(e)
                }
        
        # Compare the diagnoses
        comparison_result = comparison_engine.compare_diagnoses(
            radiologist_diagnosis=radiologist_analysis,
            ai_diagnosis=ai_diagnosis,
            dicom_validation=dicom_validation
        )
        
        # Add study information to the result
        study_info = {
            "accession_number": report_data.get("accession_number", "Unknown"),
            "patient_id": report_data.get("patient_id", "Unknown"),
            "study_date": report_data.get("study_date", "Unknown"),
            "report_date": report_data.get("report_date", "Unknown")
        }
        comparison_result["study_info"] = study_info
        
        # Update monitoring statistics
        update_monitoring_statistics(comparison_result)
        
        return jsonify(comparison_result)
    except Exception as e:
        logger.error(f"Error comparing diagnoses: {str(e)}")
        return jsonify({"error": f"Failed to compare diagnoses: {str(e)}"}), 500

@app.route("/api/real-time/webhook", methods=["POST"])
def ai_webhook():
    """Webhook endpoint to receive real-time AI results."""
    data = request.json
    
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Required fields for AI webhook
    required_fields = ["study_uid", "accession_number", "hemorrhage_detected", "confidence", "model_name"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
    
    try:
        # Store the incoming AI result for processing
        study_uid = data["study_uid"]
        accession_number = data["accession_number"]
        
        # Check if we already have the radiologist report for this study
        with monitoring_lock:
            # Find the study in our monitoring list
            study_index = next((i for i, s in enumerate(real_time_monitoring["studies"]) 
                              if s.get("study_uid") == study_uid), None)
            
            if study_index is not None:
                # Update the existing study with AI data
                real_time_monitoring["studies"][study_index]["ai_data"] = {
                    "hemorrhage_detected": data["hemorrhage_detected"],
                    "confidence": data["confidence"],
                    "model_name": data["model_name"],
                    "prediction_timestamp": datetime.now().isoformat()
                }
                
                # If we have both radiologist and AI data, compare them
                if "report_data" in real_time_monitoring["studies"][study_index]:
                    # Prepare the comparison payload
                    comparison_payload = {
                        "report_data": real_time_monitoring["studies"][study_index]["report_data"],
                        "ai_data": real_time_monitoring["studies"][study_index]["ai_data"]
                    }
                    
                    # Process in a separate thread to avoid blocking the webhook
                    threading.Thread(
                        target=process_comparison, 
                        args=(comparison_payload, study_index)
                    ).start()
                    
                    return jsonify({"status": "processing", "message": "Comparison in progress"})
                else:
                    return jsonify({"status": "waiting", "message": "Waiting for radiologist report"})
            else:
                # New study, add to monitoring list
                real_time_monitoring["studies"].append({
                    "study_uid": study_uid,
                    "accession_number": accession_number,
                    "timestamp": datetime.now().isoformat(),
                    "ai_data": {
                        "hemorrhage_detected": data["hemorrhage_detected"],
                        "confidence": data["confidence"],
                        "model_name": data["model_name"],
                        "prediction_timestamp": datetime.now().isoformat()
                    }
                })
                return jsonify({"status": "waiting", "message": "Waiting for radiologist report"})
    except Exception as e:
        logger.error(f"Error processing AI webhook: {str(e)}")
        return jsonify({"error": f"Failed to process AI data: {str(e)}"}), 500

@app.route("/api/real-time/report", methods=["POST"])
def report_webhook():
    """Webhook endpoint to receive real-time radiologist reports."""
    data = request.json
    
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Required fields for report webhook
    required_fields = ["study_uid", "accession_number", "report_text"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
    
    try:
        # Store the incoming report for processing
        study_uid = data["study_uid"]
        accession_number = data["accession_number"]
        report_text = data["report_text"]
        
        # Prepare report data
        report_data = {
            "report_text": report_text,
            "accession_number": accession_number,
            "patient_id": data.get("patient_id", "Unknown"),
            "study_date": data.get("study_date", "Unknown"),
            "report_date": datetime.now().isoformat()
        }
        
        with monitoring_lock:
            # Find the study in our monitoring list
            study_index = next((i for i, s in enumerate(real_time_monitoring["studies"]) 
                              if s.get("study_uid") == study_uid), None)
            
            if study_index is not None:
                # Update the existing study with report data
                real_time_monitoring["studies"][study_index]["report_data"] = report_data
                
                # If we have both radiologist and AI data, compare them
                if "ai_data" in real_time_monitoring["studies"][study_index]:
                    # Prepare the comparison payload
                    comparison_payload = {
                        "report_data": real_time_monitoring["studies"][study_index]["report_data"],
                        "ai_data": real_time_monitoring["studies"][study_index]["ai_data"]
                    }
                    
                    # Process in a separate thread to avoid blocking the webhook
                    threading.Thread(
                        target=process_comparison, 
                        args=(comparison_payload, study_index)
                    ).start()
                    
                    return jsonify({"status": "processing", "message": "Comparison in progress"})
                else:
                    return jsonify({"status": "waiting", "message": "Waiting for AI result"})
            else:
                # New study, add to monitoring list
                real_time_monitoring["studies"].append({
                    "study_uid": study_uid,
                    "accession_number": accession_number,
                    "timestamp": datetime.now().isoformat(),
                    "report_data": report_data
                })
                return jsonify({"status": "waiting", "message": "Waiting for AI result"})
    except Exception as e:
        logger.error(f"Error processing report webhook: {str(e)}")
        return jsonify({"error": f"Failed to process report data: {str(e)}"}), 500

@app.route("/api/real-time/status", methods=["GET"])
def get_monitoring_status():
    """Get the current status of real-time monitoring."""
    try:
        with monitoring_lock:
            # Create a copy of the monitoring data to avoid race conditions
            status_data = {
                "statistics": dict(real_time_monitoring["statistics"]),
                "latest_results": list(real_time_monitoring["latest_results"][:10]),  # Only return the 10 most recent
                "pending_studies": sum(1 for s in real_time_monitoring["studies"] 
                                      if "report_data" not in s or "ai_data" not in s)
            }
            
            # Calculate additional statistics
            total = status_data["statistics"]["total_studies"]
            if total > 0:
                agreement_rate = (status_data["statistics"]["agreement_count"] / total) * 100
                status_data["statistics"]["agreement_rate"] = round(agreement_rate, 2)
            else:
                status_data["statistics"]["agreement_rate"] = 0
            
            return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        return jsonify({"error": f"Failed to get monitoring status: {str(e)}"}), 500

def process_comparison(comparison_payload: Dict[str, Any], study_index: int):
    """Process a comparison in a separate thread."""
    try:
        # Call the comparison function
        result = compare()
        
        # If the comparison was successful, remove the study from monitoring
        if result.status_code == 200:
            with monitoring_lock:
                # Only remove if the index is still valid
                if study_index < len(real_time_monitoring["studies"]):
                    # Remove the processed study from the monitoring list
                    real_time_monitoring["studies"].pop(study_index)
    except Exception as e:
        logger.error(f"Error in background comparison processing: {str(e)}")

def start_cleanup_thread():
    """Start a thread to periodically clean up old studies."""
    def cleanup_old_studies():
        while True:
            try:
                # Sleep for 1 hour between cleanups
                time.sleep(3600)
                
                with monitoring_lock:
                    # Get current time
                    now = datetime.now()
                    
                    # Remove studies older than 24 hours
                    real_time_monitoring["studies"] = [
                        s for s in real_time_monitoring["studies"]
                        if (now - datetime.fromisoformat(s["timestamp"])).total_seconds() < 86400
                    ]
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}")
                # Sleep for a short period before retrying
                time.sleep(60)
    
    # Start the thread
    cleanup_thread = threading.Thread(target=cleanup_old_studies, daemon=True)
    cleanup_thread.start()

# Start the cleanup thread when the module is imported
start_cleanup_thread()

if __name__ == "__main__":
    # Run the app
    app.run(debug=True, host="0.0.0.0", port=5000)
