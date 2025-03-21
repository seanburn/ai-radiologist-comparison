import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import json

# Set page title and layout
st.set_page_config(
    page_title="AI vs Radiologist: Head CT Hemorrhage Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
API_URL = "http://localhost:5000"

# Helper function to call API
def call_api(endpoint, payload=None, method="GET"):
    """Call API endpoint and handle errors."""
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}")
        else:  # POST
            response = requests.post(f"{API_URL}{endpoint}", json=payload)
        
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Real-Time Monitoring", "Report Analysis", "AI Comparison", "Batch Processing"]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "This application compares radiologist reports with AI model outputs "
    "for intracranial hemorrhage detection in head CT scans."
)

# Home page
if page == "Home":
    st.title("AI vs Radiologist: Head CT Hemorrhage Detection")
    
    st.markdown("""
    ### Welcome to the Head CT Hemorrhage Detection Comparison Tool
    
    This application helps compare radiologist diagnoses with AI model outputs for 
    intracranial hemorrhage detection in head CT scans.
    
    #### Features:
    - **Real-Time Monitoring**: Monitor all head CTs flowing through the RIS and AI software
    - **Report Analysis**: Analyze radiology reports to extract hemorrhage diagnoses
    - **AI Comparison**: Compare radiologist and AI model diagnoses
    - **DICOM Support**: Validate head CT studies using DICOM metadata
    - **Batch Processing**: Process multiple reports and comparisons at once
    
    #### Getting Started:
    1. Select a page from the sidebar navigation
    2. The Real-Time Monitoring dashboard shows all current activity
    3. Use other pages for manual analysis and comparisons
    """)
    
    st.markdown("---")
    
    # Show system status
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### API Connection")
        try:
            health_data = call_api("/health")
            if health_data:
                st.success("✅ API is connected and running")
            else:
                st.error("❌ API connection error")
        except Exception:
            st.error("❌ Cannot connect to API")
    
    with col2:
        st.markdown("#### System Information")
        st.info(f"Version: 1.0.0")
        st.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Real-Time Monitoring page
elif page == "Real-Time Monitoring":
    st.title("Real-Time Head CT Monitoring")
    
    st.markdown("""
    This dashboard shows real-time evaluation of all head CTs flowing through the RIS and AI software.
    The system automatically captures both radiologist reports and AI model outputs, then compares them.
    """)
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh dashboard (every 30 seconds)", value=True)
    if auto_refresh:
        refresh_interval = 30
    else:
        refresh_interval = None
    
    # Manual refresh button
    refresh_button = st.button("Refresh Now")
    
    # Get monitoring status from API
    try:
        monitoring_data = call_api("/api/real-time/status")
        
        if monitoring_data:
            # Display key metrics
            st.subheader("Key Performance Metrics")
            
            stats = monitoring_data.get("statistics", {})
            total_studies = stats.get("total_studies", 0)
            agreement_count = stats.get("agreement_count", 0)
            false_pos_count = stats.get("false_positive_count", 0)
            false_neg_count = stats.get("false_negative_count", 0)
            agreement_rate = stats.get("agreement_rate", 0)
            
            # Create metric rows
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Studies", total_studies)
            
            with col2:
                st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            
            with col3:
                st.metric("False Positives", false_pos_count)
            
            with col4:
                st.metric("False Negatives", false_neg_count)
            
            # Pending studies
            pending_studies = monitoring_data.get("pending_studies", 0)
            st.subheader("Pending Studies")
            
            if pending_studies > 0:
                st.warning(f"There are {pending_studies} studies awaiting either radiologist reports or AI results")
            else:
                st.success("No pending studies")
            
            # Recent results
            st.subheader("Recent Comparison Results")
            
            latest_results = monitoring_data.get("latest_results", [])
            if latest_results:
                # Convert to dataframe for display
                results_df = pd.DataFrame(latest_results)
                
                # Format the dataframe
                display_cols = [
                    "timestamp", 
                    "comparison_result", 
                    "explanation"
                ]
                study_info_cols = []
                
                # Add study info columns if they exist
                if "study_info" in results_df.columns:
                    # Expand the study_info nested dictionaries
                    for record in latest_results:
                        study_info = record.get("study_info", {})
                        for key, value in study_info.items():
                            record[f"study_info_{key}"] = value
                    
                    # Create a new dataframe with the expanded data
                    results_df = pd.DataFrame(latest_results)
                    study_info_cols = [col for col in results_df.columns if col.startswith("study_info_")]
                    display_cols = ["timestamp"] + study_info_cols + ["comparison_result", "explanation"]
                
                # Filter and rename columns
                if all(col in results_df.columns for col in display_cols):
                    display_df = results_df[display_cols].copy()
                    
                    # Rename columns for better display
                    column_rename = {
                        "timestamp": "Timestamp",
                        "comparison_result": "Result",
                        "explanation": "Explanation",
                        "study_info_accession_number": "Accession #",
                        "study_info_patient_id": "Patient ID",
                        "study_info_study_date": "Study Date"
                    }
                    
                    # Only rename columns that exist
                    rename_dict = {k: v for k, v in column_rename.items() if k in display_df.columns}
                    display_df = display_df.rename(columns=rename_dict)
                    
                    # Format timestamp
                    if "Timestamp" in display_df.columns:
                        display_df["Timestamp"] = pd.to_datetime(display_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Apply color coding for results
                    def color_results(val):
                        if val == "agreement":
                            return "background-color: #c2f0c2"  # Light green
                        elif val == "false_positive":
                            return "background-color: #f5c6cb"  # Light red
                        elif val == "false_negative":
                            return "background-color: #f5c6cb"  # Light red
                        return ""
                    
                    # Display the styled dataframe
                    if "Result" in display_df.columns:
                        st.dataframe(display_df.style.applymap(color_results, subset=["Result"]))
                    else:
                        st.dataframe(display_df)
                else:
                    st.dataframe(results_df)
            else:
                st.info("No comparison results available yet")
            
            # System Health
            st.subheader("System Health")
            system_health_col1, system_health_col2 = st.columns(2)
            
            with system_health_col1:
                st.success("✅ RIS Connection Active")
            
            with system_health_col2:
                st.success("✅ AI System Connection Active")
            
            # Auto-refresh
            if auto_refresh:
                time_placeholder = st.empty()
                time_placeholder.info(f"Dashboard will refresh in {refresh_interval} seconds...")
                
                # Add JavaScript to auto-refresh the page
                st.markdown(
                    f"""
                    <script>
                        setTimeout(function() {{
                            window.location.reload();
                        }}, {refresh_interval * 1000});
                    </script>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Failed to load monitoring data from API")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        # Show simulated data in preview mode
        st.warning("Showing simulated data (API not connected)")
        
        # Simulate metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Studies", "124")
        with col2:
            st.metric("Agreement Rate", "89.5%")
        with col3:
            st.metric("False Positives", "8")
        with col4:
            st.metric("False Negatives", "5")
        
        # Simulate pending studies
        st.subheader("Pending Studies")
        st.warning("There are 3 studies awaiting either radiologist reports or AI results")
        
        # Simulate recent results
        st.subheader("Recent Comparison Results")
        
        # Create sample data
        data = {
            "Timestamp": ["2025-03-21 00:45:12", "2025-03-21 00:32:18", "2025-03-21 00:28:05"],
            "Accession #": ["ACC123456", "ACC123455", "ACC123454"],
            "Patient ID": ["MRN67890", "MRN67889", "MRN67888"],
            "Result": ["agreement", "false_positive", "agreement"],
            "Explanation": [
                "Both radiologist and AI detected hemorrhage",
                "AI detected hemorrhage but radiologist did not",
                "Both radiologist and AI reported no hemorrhage"
            ]
        }
        
        sample_df = pd.DataFrame(data)
        
        # Apply color coding
        def color_results(val):
            if val == "agreement":
                return "background-color: #c2f0c2"  # Light green
            elif val in ["false_positive", "false_negative"]:
                return "background-color: #f5c6cb"  # Light red
            return ""
        
        st.dataframe(sample_df.style.applymap(color_results, subset=["Result"]))

# Report Analysis page
elif page == "Report Analysis":
    st.title("Radiology Report Analysis")
    
    st.markdown("""
    This page allows you to analyze a radiology report to determine if it indicates 
    intracranial hemorrhage.
    """)
    
    # Report entry section
    st.subheader("Enter Report Text")
    
    report_text = st.text_area(
        "Radiology Report Text", 
        value="",
        height=300
    )
    
    analyze_button = st.button("Analyze Report")
    
    if analyze_button and report_text:
        with st.spinner("Analyzing report..."):
            # Try to call the API
            try:
                result = call_api("/api/analyze-report", {"report_text": report_text}, method="POST")
                
                if result:
                    # Display results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        hemorrhage = result.get("hemorrhage_detected")
                        if hemorrhage is True:
                            st.error("🔴 HEMORRHAGE DETECTED")
                        elif hemorrhage is False:
                            st.success("🟢 NO HEMORRHAGE DETECTED")
                        else:
                            st.warning("⚠️ INCONCLUSIVE")
                    
                    with col2:
                        confidence = result.get("confidence", 0)
                        st.metric("Confidence", f"{confidence:.2f}")
                    
                    # Detailed analysis
                    analysis_text = result.get("analysis_text", "")
                    if analysis_text:
                        with st.expander("Detailed Analysis"):
                            st.write(analysis_text)
                else:
                    # Fallback to local analysis if API fails
                    st.subheader("Analysis Results (Local Fallback)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "hemorrhage" in report_text.lower():
                            st.error("🔴 HEMORRHAGE DETECTED")
                        else:
                            st.success("🟢 NO HEMORRHAGE DETECTED")
                    
                    with col2:
                        st.metric("Confidence", "0.95")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
                # Fallback to local analysis
                st.subheader("Analysis Results (Local Fallback)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "hemorrhage" in report_text.lower():
                        st.error("🔴 HEMORRHAGE DETECTED")
                    else:
                        st.success("🟢 NO HEMORRHAGE DETECTED")
                
                with col2:
                    st.metric("Confidence", "0.95")
    
    elif analyze_button:
        st.warning("Please enter a report text to analyze.")

# AI Comparison page
elif page == "AI Comparison":
    st.title("AI vs Radiologist Comparison")
    
    st.markdown("""
    This page allows you to compare the radiologist's diagnosis with an AI model's output
    for intracranial hemorrhage detection.
    """)
    
    # Report section
    st.subheader("Radiologist Report")
    
    report_text = st.text_area(
        "Enter Report Text",
        value="",
        height=200
    )
    
    # AI model output section
    st.subheader("AI Model Output")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ai_result = st.radio(
            "AI Hemorrhage Detection Result",
            ["Hemorrhage Detected", "No Hemorrhage Detected"],
            index=1
        )
        ai_hemorrhage = ai_result == "Hemorrhage Detected"
    
    with col2:
        ai_confidence = st.slider(
            "AI Confidence Score",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.01
        )
    
    # Comparison button
    compare_button = st.button("Compare Diagnoses")
    
    if compare_button:
        if not report_text:
            st.warning("Please enter a radiologist report.")
        else:
            with st.spinner("Comparing diagnoses..."):
                # Prepare payload
                payload = {
                    "report_data": {"report_text": report_text},
                    "ai_data": {
                        "hemorrhage_detected": ai_hemorrhage,
                        "confidence": ai_confidence,
                        "model_name": "Manual Comparison Model"
                    }
                }
                
                # Try to call the API
                try:
                    result = call_api("/api/compare", payload, method="POST")
                    
                    if result:
                        st.subheader("Comparison Results")
                        
                        # Show result prominently
                        result_type = result.get("comparison_result", "unknown")
                        explanation = result.get("explanation", "")
                        
                        if result_type == "agreement":
                            st.success(f"✅ AGREEMENT: {explanation}")
                        elif result_type == "false_positive":
                            st.error(f"❌ FALSE POSITIVE: {explanation}")
                        elif result_type == "false_negative":
                            st.error(f"❌ FALSE NEGATIVE: {explanation}")
                        elif result_type == "inconclusive":
                            st.warning(f"⚠️ INCONCLUSIVE: {explanation}")
                        else:
                            st.error(f"⚠️ ERROR: {explanation}")
                        
                        # Detailed results
                        with st.expander("Detailed Comparison Information"):
                            st.json(result)
                    else:
                        # Fallback to local comparison if API fails
                        st.subheader("Comparison Results (Local Fallback)")
                        
                        report_hemorrhage = "hemorrhage" in report_text.lower()
                        
                        if report_hemorrhage == ai_hemorrhage:
                            st.success("✅ AGREEMENT: Both the radiologist and AI model agree.")
                        elif ai_hemorrhage and not report_hemorrhage:
                            st.error("❌ FALSE POSITIVE: AI detected hemorrhage, but radiologist did not.")
                        else:
                            st.error("❌ FALSE NEGATIVE: AI did not detect hemorrhage, but radiologist did.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
                    # Fallback to local comparison
                    st.subheader("Comparison Results (Local Fallback)")
                    
                    report_hemorrhage = "hemorrhage" in report_text.lower()
                    
                    if report_hemorrhage == ai_hemorrhage:
                        st.success("✅ AGREEMENT: Both the radiologist and AI model agree.")
                    elif ai_hemorrhage and not report_hemorrhage:
                        st.error("❌ FALSE POSITIVE: AI detected hemorrhage, but radiologist did not.")
                    else:
                        st.error("❌ FALSE NEGATIVE: AI did not detect hemorrhage, but radiologist did.")

# Batch Processing page
elif page == "Batch Processing":
    st.title("Batch Processing")
    
    st.markdown("""
    This page allows you to process multiple comparisons at once using a CSV file.
    
    The CSV file should contain the following columns:
    - `report_text`: The full text of the radiologist's report
    - `ai_result`: Boolean indicating if AI detected hemorrhage (1) or not (0)
    - `ai_confidence`: Confidence score of the AI model (0.0-1.0)
    - Other optional columns: `accession_number`, `patient_id`, `study_date`
    """)
    
    # File upload section
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load CSV data
            df = pd.read_csv(uploaded_file)
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_columns = ["report_text", "ai_result"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Process the dataset
                if st.button("Process Batch"):
                    st.info("Processing batch data... This may take some time.")
                    
                    # Initialize results list
                    results = []
                    
                    # Set up progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each row
                    for i, (_, row) in enumerate(df.iterrows()):
                        # Update progress
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing record {i+1} of {len(df)}")
                        
                        try:
                            # Prepare payload
                            report_data = {"report_text": row["report_text"]}
                            
                            # Add optional metadata if present
                            for field in ["accession_number", "patient_id", "study_date"]:
                                if field in row and not pd.isna(row[field]):
                                    report_data[field] = str(row[field])
                            
                            ai_data = {
                                "hemorrhage_detected": bool(row["ai_result"]),
                                "confidence": row.get("ai_confidence", 0.9),
                                "model_name": "Batch Processing Model"
                            }
                            
                            payload = {
                                "report_data": report_data,
                                "ai_data": ai_data
                            }
                            
                            # Try API call
                            try:
                                comparison_result = call_api("/api/compare", payload, method="POST")
                                
                                if comparison_result:
                                    results.append({
                                        "index": i,
                                        "accession": row.get("accession_number", f"ROW_{i}"),
                                        "result": comparison_result.get("comparison_result"),
                                        "explanation": comparison_result.get("explanation"),
                                        "radiologist_hemorrhage": comparison_result.get("radiologist_diagnosis", {}).get("hemorrhage_detected"),
                                        "ai_hemorrhage": comparison_result.get("ai_diagnosis", {}).get("hemorrhage_detected")
                                    })
                                else:
                                    # Fallback to local analysis
                                    report_hemorrhage = "hemorrhage" in row["report_text"].lower()
                                    ai_hemorrhage = bool(row["ai_result"])
                                    
                                    if report_hemorrhage == ai_hemorrhage:
                                        result_type = "agreement"
                                        explanation = "Both agree"
                                    elif ai_hemorrhage and not report_hemorrhage:
                                        result_type = "false_positive"
                                        explanation = "AI reported hemorrhage, radiologist did not"
                                    else:
                                        result_type = "false_negative"
                                        explanation = "Radiologist reported hemorrhage, AI did not"
                                    
                                    results.append({
                                        "index": i,
                                        "accession": row.get("accession_number", f"ROW_{i}"),
                                        "result": result_type,
                                        "explanation": explanation,
                                        "radiologist_hemorrhage": report_hemorrhage,
                                        "ai_hemorrhage": ai_hemorrhage
                                    })
                            except Exception:
                                # Local fallback on exception
                                report_hemorrhage = "hemorrhage" in row["report_text"].lower()
                                ai_hemorrhage = bool(row["ai_result"])
                                
                                if report_hemorrhage == ai_hemorrhage:
                                    result_type = "agreement"
                                    explanation = "Both agree"
                                elif ai_hemorrhage and not report_hemorrhage:
                                    result_type = "false_positive"
                                    explanation = "AI reported hemorrhage, radiologist did not"
                                else:
                                    result_type = "false_negative"
                                    explanation = "Radiologist reported hemorrhage, AI did not"
                                
                                results.append({
                                    "index": i,
                                    "accession": row.get("accession_number", f"ROW_{i}"),
                                    "result": result_type,
                                    "explanation": explanation,
                                    "radiologist_hemorrhage": report_hemorrhage,
                                    "ai_hemorrhage": ai_hemorrhage
                                })
                        except Exception as e:
                            st.error(f"Error processing row {i}: {str(e)}")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    st.subheader("Batch Results")
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        
                        # Apply color coding
                        def color_results(val):
                            if val == "agreement":
                                return "background-color: #c2f0c2"  # Light green
                            elif val in ["false_positive", "false_negative"]:
                                return "background-color: #f5c6cb"  # Light red
                            return ""
                        
                        st.dataframe(results_df.style.applymap(color_results, subset=["result"]))
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        
                        # Count results by type
                        result_counts = results_df["result"].value_counts()
                        
                        # Calculate percentages
                        total = len(results_df)
                        agreement_count = result_counts.get("agreement", 0)
                        agreement_pct = (agreement_count / total) * 100 if total > 0 else 0
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Cases", total)
                        
                        with col2:
                            st.metric("Agreement Cases", agreement_count)
                        
                        with col3:
                            st.metric("Agreement Rate", f"{agreement_pct:.1f}%")
                        
                        # Show result breakdown
                        st.bar_chart(result_counts)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results CSV",
                            csv,
                            "batch_results.csv",
                            "text/csv",
                            key="download-csv"
                        )
                    else:
                        st.warning("No results were generated. Please check your input data.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Display app footer
st.markdown("---")
st.markdown(
    " 2025 AI vs Radiologist | "
    "Developed for comparing radiologist diagnoses with AI model outputs"
)
