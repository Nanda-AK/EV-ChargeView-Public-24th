import streamlit as st
from openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI
import pandas as pd

# Load Data
EV_df = pd.read_csv("cleaned_ev_data.csv")

# Sidebar for API key input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Prompt input
st.title("🔌 EV Charging Insight Chatbot")
user_prompt = st.text_area("Ask a question about the EV station dataset")

# Function: Refine Prompt and detect if chart is needed
def refine_prompt(user_prompt):
    client = OpenAI(api_key=api_key)

    # Detect if prompt requests a chart
    chart_keywords = ["chart", "graph", "plot", "visualize", "bar chart", "line chart"]
    wants_chart = any(keyword in user_prompt.lower() for keyword in chart_keywords)

    if wants_chart:
        system_instruction = """
You are a data analyst assistant refining user questions for an EV charging dataset.

If the user asks for a chart, generate a refined prompt that will:
- Extract clear X and Y values for plotting
- Avoid using restricted matplotlib calls like 'gca' or 'tight_layout'
- Prefer simple bar or line plots

Avoid logging, retries, or error tracebacks.
"""
    else:
        system_instruction = """
You are a data analyst assistant refining user questions for an EV charging dataset.

Return a concise and clean prompt optimized for querying a SmartDataframe.
Avoid any mention of charts or graphs.
Exclude all logs, retries, tracebacks, or internal errors.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip(), wants_chart

# Process prompt on button click
if st.button("Submit Query") and user_prompt:
    with st.spinner("Refining and querying..."):
        refined, wants_chart = refine_prompt(user_prompt)
        st.info(f"🔍 Refined Prompt: {refined}")

        # Create SmartDataframe
        llm = PandasAI_OpenAI(api_token=api_key)
        EV_SmartDF = SmartDataframe(EV_df, config={"llm": llm, "verbose": True})

        try:
            response = EV_SmartDF.chat(refined)
            st.subheader("🤖 Chatbot Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error during chat: {e}")
            response = None

        # Show chart if requested
        if wants_chart and response is not None:
            import matplotlib.figure
            try:
                if hasattr(response, 'chart') and response.chart is not None:
                    if isinstance(response.chart, matplotlib.figure.Figure):
                        st.pyplot(response.chart)
                    elif isinstance(response.chart, str) and response.chart.endswith(('.png', '.jpg')):
                        from PIL import Image
                        import os
                        if os.path.exists(response.chart):
                            img = Image.open(response.chart)
                            st.image(img, caption="Generated Chart")
                        else:
                            st.warning(f"Chart image file not found: {response.chart}")
            except Exception as e:
                st.warning(f"Chart rendering failed: {e}")
