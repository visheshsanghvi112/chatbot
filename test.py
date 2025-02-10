import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from time import sleep
from datetime import datetime
import numpy as np
from io import StringIO
import base64
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Specify the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="AI Milestone - Enhanced GPT Chatbot",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS STYLING ----
st.markdown("""
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Courier New', monospace;
    }
    .header {
        background-color: #1e1e1e;
        color: #00FF00;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0, 255, 0, 0.4);
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 20px;
        border-radius: 10px;
        background-color: #1a1a1a;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
    }
    .user-message {
        background-color: #333333;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #00FF00;
    }
    .assistant-message {
        background-color: #444444;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
    }
    .timestamp {
        font-size: 12px;
        color: #999999;
        text-align: right;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #999999;
        margin-top: 20px;
    }
    .footer a {
        color: #00FF00;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .btn-clear {
        background-color: #ff0000;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 14px;
        cursor: pointer;
    }
    .btn-clear:hover {
        background-color: #cc0000;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown('<div class="header">👽 AI Milestone - Vishesh GPT Chatbot</div>', unsafe_allow_html=True)

# ---- API LOGIC ----
def get_response_from_api(messages):
    """
    Sends the conversation context to the AI API and retrieves a response.
    """
    api_key = "AIzaSyARDiJ0B2jIGeTm9-L9ay0mPNu3PTO1G7A"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{
                "text": "\n".join([msg['content'] for msg in messages])
            }]
        }]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except KeyError:
            return "Error: Unable to parse the response from the AI."
    else:
        return "Error: Failed to connect to the AI server."

# ---- COMMAND HANDLING ----
def handle_commands(user_input):
    """
    Processes commands like /help, /clear, /date, or creator-related queries.
    """
    command = user_input.lower().strip()
    if command in ["/help", "help"]:
        return (
            "**Available Commands:**\n"
            "- `/help`: Show help.\n"
            "- `/clear`: Clear chat history.\n"
            "- `/date`: Get current date and time.\n"
            "- `/about`: Learn about the chatbot.\n"
            "- `/creator`: Know who created me.\n"
            "- `/analyze`: Analyze uploaded data.\n"
            "- `/visualize`: Create visualizations.\n"
            "- `/export`: Export chat history.\n"
            "- `/theme`: Toggle dark/light mode.\n"
            "- `/columns`: List columns in the uploaded data.\n"
            "- `/stats`: Get basic statistics of the uploaded data.\n"
            "- `/summary`: Get a summary of the uploaded data.\n"
            "- `/analyze_pdf`: Analyze uploaded PDF files.\n"
            "- `/analyze_image`: Analyze uploaded images."
        )
    elif command in ["/clear", "clear"]:
        st.session_state.messages = []
        return "Chat history cleared!"
    elif command in ["/date", "date"]:
        return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elif command in ["/about", "about"]:
        return (
            "I'm a chatbot powered by Vishesh GPT, designed to assist you with context-aware conversations."
        )
    elif "who created you" in command or "/creator" in command:
        return (
            "I was created by Vishesh Sanghvi using cutting-edge AI technologies. "
            "You can learn more about him on his [LinkedIn profile](https://www.linkedin.com/in/vishesh-sanghvi-96b16a237/)."
        )
    elif command == "/analyze":
        return handle_data_analysis()
    elif command == "/visualize":
        return handle_visualization()
    elif command == "/export":
        return export_chat_history()
    elif command == "/theme":
        toggle_theme()
        return "Theme updated!"
    elif command == "/columns":
        return list_columns()
    elif command == "/stats":
        return get_basic_stats()
    elif command == "/summary":
        return get_data_summary()
    elif command == "/analyze_pdf":
        return analyze_pdf()
    elif command == "/analyze_image":
        return analyze_image()
    else:
        return None

def list_columns():
    if 'data' not in st.session_state:
        return "Please upload a data file first!"
    
    df = st.session_state.data
    columns = list(df.columns)
    return f"Columns in the uploaded data:\n{columns}"

def get_basic_stats():
    if 'data' not in st.session_state:
        return "Please upload a data file first!"
    
    df = st.session_state.data
    stats = df.describe().to_dict()
    return f"Basic Statistics of the uploaded data:\n```json\n{json.dumps(stats, indent=2)}\n```"

def get_data_summary():
    if 'data' not in st.session_state:
        return "Please upload a data file first!"
    
    df = st.session_state.data
    summary = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Data Types": df.dtypes.astype(str).to_dict(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Head": df.head().to_dict(orient='records')
    }
    return f"Data Summary:\n```json\n{json.dumps(summary, indent=2)}\n```"

def analyze_pdf():
    if 'pdf' not in st.session_state:
        return "Please upload a PDF file first!"
    
    pdf_text = ""
    for page_num in range(len(st.session_state.pdf)):
        page = st.session_state.pdf[page_num]
        pdf_text += page.get_text()
    
    return f"PDF Analysis:\n{pdf_text[:1000]}..."  # Displaying only the first 1000 characters for brevity

def analyze_image():
    if 'image' not in st.session_state:
        return "Please upload an image file first!"
    
    image = st.session_state.image
    
    # Perform OCR
    with st.spinner("Performing OCR..."):
        text = pytesseract.image_to_string(image)

    return f"Image Analysis (Extracted Text):\n{text}"

# ---- DATA ANALYSIS FUNCTIONS ----
def handle_data_analysis():
    if 'data' not in st.session_state:
        return "Please upload a data file first!"
    
    df = st.session_state.data
    analysis = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Basic Stats": df.describe().to_dict()
    }
    return f"Data Analysis Results:\n```json\n{json.dumps(analysis, indent=2)}\n```"

def handle_visualization():
    if 'data' not in st.session_state:
        return "Please upload a data file first!"
    
    df = st.session_state.data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 1:
        return "No numeric columns found for visualization!"
    
    fig = px.scatter_matrix(df[numeric_cols], title='Scatter Matrix of Numeric Columns')
    st.plotly_chart(fig)
    return "Visualization created!"

def export_chat_history():
    if not st.session_state.messages:
        return "No chat history to export!"
    
    chat_export = ""
    for msg in st.session_state.messages:
        timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        chat_export += f"{msg['role']} ({timestamp}): {msg['content']}\n"
    
    b64 = base64.b64encode(chat_export.encode()).decode()
    return f"Download link: <a href='data:text/plain;base64,{b64}' download='chat_history.txt'>Download Chat History</a>"

def toggle_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# ---- REAL-TIME DATA INTEGRATION FUNCTION ----
def get_real_time_data():
    """
    Fetch real-time data from an API.
    """
    url = 'https://api.example.com/real-time-data'  # Replace with a valid API URL
    response = requests.get(url)
    data = response.json()
    return data

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "data" not in st.session_state:
    st.session_state.data = None

if "pdf" not in st.session_state:
    st.session_state.pdf = None

if "image" not in st.session_state:
    st.session_state.image = None

# ---- SIDEBAR ----
with st.sidebar:
    st.title("Settings & Tools")
    
    # Hints Dropdown
    with st.expander("Hints & Commands"):
        st.markdown("""
        **Available Commands:**
        - `/help`: Show help.
        - `/clear`: Clear chat history.
        - `/date`: Get current date and time.
        - `/about`: Learn about the chatbot.
        - `/creator`: Know who created me.
        - `/analyze`: Analyze uploaded data.
        - `/visualize`: Create visualizations.
        - `/export`: Export chat history.
        - `/theme`: Toggle dark/light mode.
        - `/columns`: List columns in the uploaded data.
        - `/stats`: Get basic statistics of the uploaded data.
        - `/summary`: Get a summary of the uploaded data.
        - `/analyze_pdf`: Analyze uploaded PDF files.
        - `/analyze_image`: Analyze uploaded images.
        """)

    uploaded_file = st.file_uploader("Upload Data File (CSV, XLSX, PDF, Image)", type=['csv', 'xlsx', 'pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"CSV file '{uploaded_file.name}' uploaded successfully!")
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.data = pd.read_excel(uploaded_file)
                st.success(f"Excel file '{uploaded_file.name}' uploaded successfully!")
            elif uploaded_file.name.endswith('.pdf'):
                st.session_state.pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                st.success(f"PDF file '{uploaded_file.name}' uploaded successfully!")
            elif uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
                st.session_state.image = Image.open(uploaded_file)
                st.success(f"Image file '{uploaded_file.name}' uploaded successfully!")
            
            if st.checkbox("Show Data Preview") and st.session_state.data is not None:
                st.dataframe(st.session_state.data.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ---- DISPLAY CHAT ----
def display_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-message"><b>You:</b> {msg["content"]}</div><div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)
        elif msg['role'] == 'assistant':
            st.markdown(f'<div class="assistant-message"><b>AI:</b> {msg["content"]}</div><div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- INPUT FORM ----
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask me anything:",
        placeholder="Type your question or command (e.g., /help)...",
        help="Ask questions or use commands like /help, /clear, /creator."
    )
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # Handle commands
        command_response = handle_commands(user_input.strip())
        if command_response:
            st.session_state.messages.append({"role": "assistant", "content": command_response, "timestamp": datetime.now()})
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_input.strip(), "timestamp": datetime.now()})
            
            with st.spinner("AI is thinking..."):
                sleep(1)  # Simulate processing
                ai_response = get_response_from_api(st.session_state.messages)
            # Append AI response
            st.session_state.messages.append({"role": "assistant", "content": ai_response, "timestamp": datetime.now()})

# ---- DISPLAY CHAT ----
display_chat()

# ---- FOOTER ----
st.markdown(
    """
    <div class="footer">
    <hr>
    Created with ❤️ by Vishesh GPT | <a href="https://www.linkedin.com/in/vishesh-sanghvi-96b16a237/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
