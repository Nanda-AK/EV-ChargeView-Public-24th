import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import openai

# Load environment variables
load_dotenv()

st.set_page_config(page_title="EV Charging Station Insights", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

# Inject modern, professional dark theme CSS
st.markdown(
    """
    <style>
    /* Base dark background */
    body, .stApp {
        background-color: #111217;
        color: #FFF;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    /* Headings - bold, modern, high contrast */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, h1, h2, h3, h4, h5, h6 {
        color: #FFF;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    /* Sidebar styling */
    .stSidebar {
        background-color: #18191F !important;
        box-shadow: 2px 0 8px 0 rgba(0,0,0,0.25);
    }
    /* Inputs and text areas */
    .stTextInput, .stTextArea, .stSelectbox, .stRadio, .stDataFrame, .stAlert, .stMarkdown {
        color: #FFF !important;
        background-color: #18191F !important;
        border-radius: 8px;
        border: 1px solid #23232B !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    }
    /* Button styling */
    .stButton>button {
        color: #FFF;
        background: linear-gradient(90deg, #FF8800 0%, #FFB347 100%);
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(255,136,0,0.10);
        font-weight: 600;
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FFB347 0%, #FF8800 100%);
        box-shadow: 0 4px 16px rgba(255,136,0,0.18);
    }
    /* Highlight for selected prompts/questions */
    .stRadio [aria-checked="true"] {
        background: #FF8800 !important;
        color: #FFF !important;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 1px 8px rgba(255,136,0,0.10);
    }
    /* Chart backgrounds and accent border */
    .element-container svg, .stPlotlyChart, .stAltairChart, .stVegaLiteChart, .stPyplot {
        background-color: #18191F !important;
        border-radius: 10px;
        border: 2px solid #00B8A9;
        box-shadow: 0 2px 12px rgba(0,184,169,0.10);
        padding: 12px;
    }
    /* Accent color for links and highlights */
    a, .accent, .stMarkdown a {
        color: #00B8A9 !important;
        font-weight: 600;
    }
    /* Subtle shadow for cards/containers */
    .stContainer, .stAlert, .stDataFrame {
        box-shadow: 0 2px 16px rgba(0,0,0,0.18);
        border-radius: 10px;
    }
    /* Make all Streamlit label, header, and caption text white */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
    .stSidebar .stRadio label, .stSidebar .stSelectbox label, .stSidebar label,
    .stSidebar .stMarkdown, .stSidebar .stTextInput label, .stSidebar .stTextArea label,
    .stSidebar .stCaption, .stSidebar p, .stSidebar span,
    label[data-testid="stSidebarLabel"],
    .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label,
    .stMarkdown label, .stCaption, p, span, label {
        color: #FFF !important;
    }
    /* Also force white for all streamlit widget labels in main area */
    label, .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label, .stCaption, .stMarkdown label, p, span {
        color: #FFF !important;
    }
    </style>
    <link href="https://fonts.googleapis.com/css?family=Inter:400,600,700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# Show logo at the top-right using st.image and columns for compatibility
logo_path = os.path.join(os.path.dirname(__file__), 'EV ChargeInsight logo.png')
logo_col1, logo_col2 = st.columns([8, 1])
with logo_col2:
    st.image(logo_path, use_container_width=False, width=72, caption=None, output_format="auto")

# Optional: add a little CSS to float the logo right and add shadow
st.markdown(
    """
    <style>
    .element-container img[alt="EV ChargeInsight Logo"] {
        float: right;
        margin-top: -16px;
        margin-bottom: 8px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,184,169,0.16);
        background: #23272F;
        padding: 4px;
        height: 56px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("EV Charging Station Insights (Google Maps Data)")

# API Key setup (use .env or Streamlit secrets)
OPENAI_API_KEY = st.secrets['OpenAI_API_KEY']
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in a .env file or Streamlit secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# File loader: load from repo instead of upload
DATA_PATH = os.path.join(os.path.dirname(__file__), "cleaned_ev_data.csv")
if os.path.exists(DATA_PATH):
    EV_df = pd.read_csv(DATA_PATH)
    # Removed st.success message as requested
else:
    st.error(f"Could not find {DATA_PATH}. Please ensure the file exists in the repository.")
    st.stop()

system_instruction = """
You are a data analyst and cleanup assistant for an EV charging station SmartDataframe chatbot.

Your job includes two parts:
1. Refine the user's question to be specific, concise, and suitable for querying a PandasAI SmartDataframe.
2. Clean the LLM output by removing unnecessary logs, errors, warnings, tracebacks, and retries.

Guidelines:
- Only reference relevant columns: EV Vendor, city, state, rank, totalScore, reviewsCount, categoryName, address.
- Avoid complex or restricted matplotlib functions like 'gca()', 'tight_layout()', etc.
- Prefer simple aggregations like counts, averages, and basic bar/line plots.
- If a table or chart is present, present it cleanly.
- Never output raw error messages like: ERROR, WARNING, Traceback, Retry, ModuleNotFoundError.

Return only the clean, corrected, final prompt ready for querying the SmartDataframe.
Do NOT explain what you are doing. Just output the refined version.
"""

system_prompt = """
You are a data assistant for an electric vehicle (EV) charging station dashboard.

You must ONLY access and query the columns required to answer the user's question. Do NOT scan or consider the full dataset unless absolutely necessary.

Available columns include:
- EV Vendor
- city
- state
- address
- totalScore
- reviewsCount
- categoryName
- rank
- location

Ignore any large or nested fields like: reviews, reviewsDistribution, popularTimesHistogram, or detailed JSONs unless specifically asked.

Avoid including the full DataFrame or long context in your output. Respond clearly, concisely, and use only what’s needed.

Charts or tables must reference filtered data, not full dumps.

Your job is to reduce token usage while delivering actionable insights.
"""

llm = OpenAI(Model="GPT-4o")
#EV_SmartDF = SmartDataframe(EV_df, config={"llm": llm, "system_message": system_prompt})
EV_SmartDF = SmartDataframe(EV_df, config={"llm": llm, "system_message": system_instruction})

def refine_prompt(user_prompt):
    client = OpenAI()
    system_instruction = """
You are a data analyst assistant helping refine user questions for a SmartDataframe
containing EV charging station data from Google Maps.

Make the prompt specific, concise, and compatible with PandasAI.
Avoid using advanced or restricted matplotlib methods like 'tight_layout' or 'gca'.
Prefer simple operations like group by, counts, averages, bar/line plots.
Use field names like: EV Vendor, city, state, rank, totalScore, reviewsCount, etc.
Format the prompt clearly and naturally for LLM data querying.
Return only the improved prompt. Do NOT explain or comment.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def clean_llm_output(raw_output):
    client = OpenAI()
    system_instruction = """
You are a cleanup assistant for an AI data chatbot.

You are given raw LLM output which might include logs, errors, tracebacks, retries, or system warnings.
Your job is to extract and return only the clean, relevant answer for the user.

If a chart or table is included, describe it briefly or return it cleanly.
Ignore any lines like:
- ERROR:...
- WARNING:...
- Traceback...
- Retry number...
- ModuleNotFoundError...
- Any internal PandasAI or matplotlib error logs

Do not explain what you did — just return the clean result.
"""
    safe_output = str(raw_output)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": safe_output}
        ]
    )
    return response.choices[0].message.content.strip()

# Sidebar with predefined questions
st.sidebar.header("Predefined Analysis Questions")

predefined_questions = {
    "Location-Based Analysis": [
        "Which cities have the highest number of EV stations?",
        "List all vendors with station count in San Jose, CA, and show it in a bar chart.",
        "What is the average rank of stations in each city?",
    ],
    "Vendor Performance Insights": [
        "Which EV vendor has the most stations overall?",
        "Show average review score for each EV vendor in descending order.",
        "List all vendors and their total review count.",
    ],
    "Quality & Ranking": [
        "Which vendor has the best average rank across all locations?",
        "Which stations have the most user reviews? List top 5 with vendor and location.",
    ],
    "Risk/Complaint Indicators": [
        "Summarize common user complaints based on reviews.",
    ],
    "Trends & Strategy": [
        "Create a bar chart comparing total stations by vendor in California.",
    ],
}

category = st.sidebar.selectbox("Select Category", list(predefined_questions.keys()))
question = st.sidebar.radio(
    "Choose a question:",
    predefined_questions[category],
    key="predefined_question"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Or type your own question")

# Main UI prompt box (pre-filled with selected question)
user_prompt = st.text_area(
    "Ask a question about the EV charging station data:",
    value=question if question else "",
    key="main_prompt_box"
)

if st.button("Submit Query") and user_prompt:
    with st.spinner("Processing your query..."):
        refined = refine_prompt(user_prompt)
        st.info(f"Refined User Question: {refined}")  # Display the refined prompt in the Streamlit UI
        #response = EV_SmartDF.chat(refined)
        final_response = EV_SmartDF.chat(refined)
        #final_response = clean_llm_output(response)
    st.subheader("LLM Response:")
    st.write(final_response)
    # Try to display chart if present
    if hasattr(response, 'chart') and response.chart is not None:
        try:
            import matplotlib.figure
            if isinstance(response.chart, matplotlib.figure.Figure):
                st.pyplot(response.chart)
            elif isinstance(response.chart, str) and (response.chart.endswith('.png') or response.chart.endswith('.jpg')):
                from PIL import Image
                import os
                if os.path.exists(response.chart):
                    img = Image.open(response.chart)
                    st.image(img, caption="Generated Chart")
                else:
                    st.warning(f"Chart image file not found: {response.chart}")
        except Exception as e:
            st.warning(f"Could not display chart: {e}")
