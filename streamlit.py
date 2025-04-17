import streamlit as st
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import json

# Set page configuration
st.set_page_config(
    page_title="Smart Trash Classification Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path to the uploads folder (same as in Flask app)
UPLOAD_FOLDER = 'static/uploads'

# Path to store counter data
DATA_FILE = 'trash_data.json'

# Function to load the latest image
def get_latest_image():
    if not os.path.exists(UPLOAD_FOLDER):
        return None, "No images directory found"
    
    files = os.listdir(UPLOAD_FOLDER)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        return None, "No images found"
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
    latest_image = image_files[0]
    
    # Extract prediction from filename if available
    prediction = "Unknown"
    if "_" in latest_image:
        parts = latest_image.split("_")
        if len(parts) > 1:
            prediction = parts[1].split(".")[0]
    
    image_path = os.path.join(UPLOAD_FOLDER, latest_image)
    timestamp = datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%Y-%m-%d %H:%M:%S')
    
    return image_path, timestamp, prediction

def load_counter_data():
    """Load the counter data from the JSON file if available"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {DATA_FILE}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    # Return the default counter data if loading fails or file doesn't exist
    return {"Organic": 0, "Other": 0, "Plastic": 0, "history": []}


# Function to create a bar chart of trash counts
def create_bar_chart(counter_data):
    categories = list(counter_data.keys())
    categories = [cat for cat in categories if cat != "history"]  # Exclude history
    counts = [counter_data[cat] for cat in categories]
    
    chart_data = pd.DataFrame({
        'Category': categories,
        'Count': counts
    })
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Category', sort=None),
        y='Count',
        color=alt.Color('Category', scale=alt.Scale(
            domain=['Organic', 'Other', 'Plastic'],
            range=['#27ae60', '#3498db', '#f39c12']
        ))
    ).properties(
        title='Trash Classification Counts'
    )
    
    return chart

# Function to create a line chart of trash counts over time
def create_time_chart(history_data):
    if not history_data or len(history_data) < 2:
        return None
    
    # Convert history data to DataFrame
    df = pd.DataFrame(history_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create line chart
    chart = alt.Chart(df).transform_fold(
        ['Organic', 'Other', 'Plastic'],
        as_=['Category', 'Count']
    ).mark_line().encode(
        x='timestamp:T',
        y='Count:Q',
        color=alt.Color('Category:N', scale=alt.Scale(
            domain=['Organic', 'Other', 'Plastic'],
            range=['#27ae60', '#3498db', '#f39c12']
        ))
    ).properties(
        title='Trash Classification Over Time'
    )
    
    return chart

# Main dashboard
def main():
    # Add title and description
    st.title("Smart Trash Classification Dashboard")
    st.markdown("Real-time monitoring of trash classification from ESP32-CAM")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    # Load data
    counter_data = load_counter_data()
    image_path, timestamp, prediction = get_latest_image()
    
    # Display latest image and prediction in the left column
    with col1:
        st.subheader("Latest Captured Image")
        if image_path:
            image = Image.open(image_path)
            st.image(image, caption=f"Captured at: {timestamp}", use_container_width=True)
            
            # Display prediction with colored box
            prediction_color = {
                "Organic": "#27ae60",
                "Other": "#3498db",
                "Plastic": "#f39c12"
            }.get(prediction, "#7f8c8d")
            
            st.markdown(
                f"""
                <div style="background-color: {prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center; font-size: 24px; margin: 10px 0;">
                    Prediction: {prediction}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No images available")
    
    # Display counters and charts in the right column
    with col2:
        st.subheader("Classification Counters")
        
        # Create three columns for the counters
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(
                """
                <div style="background-color: #abebc6; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3 style="margin: 0;">Organic</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<h1 style='text-align: center;'>{counter_data.get('Organic', 0)}</h1>", unsafe_allow_html=True)
            
        with c2:
            st.markdown(
                """
                <div style="background-color: #d6eaf8; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3 style="margin: 0;">Other</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<h1 style='text-align: center;'>{counter_data.get('Other', 0)}</h1>", unsafe_allow_html=True)
            
        with c3:
            st.markdown(
                """
                <div style="background-color: #f9e79f; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3 style="margin: 0;">Plastic</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<h1 style='text-align: center;'>{counter_data.get('Plastic', 0)}</h1>", unsafe_allow_html=True)
    
    # Display bar chart
    st.subheader("Classification Distribution")
    bar_chart = create_bar_chart(counter_data)
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Display time chart if history data is available
    if "history" in counter_data and len(counter_data["history"]) > 1:
        st.subheader("Classification Trend Over Time")
        time_chart = create_time_chart(counter_data["history"])
        if time_chart:
            st.altair_chart(time_chart, use_container_width=True)
        else:
            st.info("Not enough data points for time chart yet")
    
    # Add auto-refresh
    st.markdown(
        """
        <meta http-equiv="refresh" content="5">
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()