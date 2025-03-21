# AI vs Radiologist: Head CT Hemorrhage Detection Comparison

A Python application that compares the final diagnostic head CT radiology report from the RIS with the output of an AI model detecting intracranial hemorrhage.

## Features

- Integration with both the RIS and the AI model output (with DICOM header recognition)
- NLP component driven by an open-source LLM fine-tuned for medical language
- Python-based back end with Flask API
- Simple, user-friendly Streamlit front end

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
ai-radiologist-comparison/
├── app/
│   ├── api.py         # Flask API implementation
│   ├── streamlit_app.py  # Streamlit front end
├── models/
│   ├── llm_processor.py  # NLP model for report analysis
│   ├── fine_tuning.py    # Scripts for fine-tuning the LLM
├── data/
│   ├── sample_reports/   # Sample data for testing
│   └── fine_tuning_data/ # Data for model fine-tuning
├── utils/
│   ├── dicom_processor.py  # DICOM header parsing
│   ├── ris_connector.py    # RIS integration
│   └── comparison_engine.py  # Compare radiologist & AI outputs
├── tests/
│   └── unit_tests/      # Test scripts
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

### Running the API
```
cd ai-radiologist-comparison
python -m app.api
```

### Running the Streamlit Interface
```
cd ai-radiologist-comparison
streamlit run app/streamlit_app.py
```

## Development

This project uses an open-source LLM (from Hugging Face Transformers) that is fine-tuned on medical text data for analyzing radiology reports.
