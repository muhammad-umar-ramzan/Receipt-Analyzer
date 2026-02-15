import streamlit as st
import pandas as pd
import os
from groq import Groq  
from utils import extract_text, parse_receipt, categorize

# --- PAGE CONFIG & THEME ---
st.set_page_config(page_title="AI Receipt Analyzer", layout="wide", page_icon="💰")

# --- HACKATHON MOTIVATION ---
st.sidebar.info("💡 *'You don't have to see the whole staircase, just take the first step.'*")

st.title("AI-Powered Receipt Analyzer 💰")
st.markdown("""
    **Hackathon Objective:** Extract data from receipts, categorize expenses, 
    and get AI-driven budgeting advice for the Pakistani market.
""")

# --- 1. RECEIPT IMAGE PROCESSING (UI) ---
uploaded_file = st.file_uploader(
    "Step 1: Upload your grocery receipt", 
    type=["png", "jpg", "jpeg"],
    help="Ensure the receipt is well-lit for better OCR accuracy."
)

if uploaded_file is not None:
    # Save temporary file locally for OCR processing
    temp_path = "temp_receipt.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Receipt", width=400)

    with st.spinner("Processing Pipeline: OCR ➡️ Parsing ➡️ Categorizing..."):
        try:
            # --- 2. DATA EXTRACTION & PARSING ---
            # Calling functions from utils.py (Image processing happens inside extract_text)
            raw_text = extract_text(temp_path)
            
            with st.expander("🔍 View Raw OCR Text (Debug Mode)"):
                st.text(raw_text)

            df = parse_receipt(raw_text)

            if df.empty:
                st.error("Could not find any items or prices. Please try a clearer photo.")
            else:
                # --- 3. EXPENSE CATEGORIZATION ---
                df, category_totals = categorize(df)
                total_spend = category_totals.sum()

                # --- 4. SPENDING ANALYSIS (Hackathon Requirement) ---
                # Calculating percentages and identifying anomalies
                analysis_df = pd.DataFrame(category_totals).reset_index()
                analysis_df.columns = ['Category', 'Amount']
                analysis_df['Percentage'] = (analysis_df['Amount'] / total_spend) * 100

                # Layout for Visuals
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🛒 Itemized Breakdown")
                    st.table(df[['Item', 'Price', 'Category']])

                with col2:
                    st.subheader("📊 Spending Analysis")
                    st.bar_chart(category_totals)
                    
                    # Displaying Percentage Breakdown
                    for index, row in analysis_df.iterrows():
                        st.write(f"**{row['Category']}:** {row['Percentage']:.1f}% of total")

                st.metric(label="Total Bill", value=f"Rs. {total_spend:,.2f}")

                # --- 5. LLM INTEGRATION FOR FINANCIAL ADVICE ---
                st.divider()
                st.subheader("🤖 AI Budgeting Consultant")

                try:
                    # Groq SDK Implementation
                   
                    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                    # Humanizing the prompt with context
                    context_data = analysis_df.to_string(index=False)
                    prompt = f"""
                    Context: A user in Islamabad, Pakistan just uploaded a grocery receipt.
                    Data:
                    {context_data}
                    Total: Rs. {total_spend:.2f}

                    Task:
                    1. Identify if any category (like Snacks or Meat) is too high (anomaly detection).
                    2. Give 4-5 practical 'bachat' (saving) tips.
                    3. Mention local Islamabad context (e.g., shopping at Sunday Bazaars or checking fuel prices).
                    4. Keep the tone encouraging. .
                    """

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a smart financial advisor specialized in Pakistani household economy."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.6 # Lower temperature for more focused advice
                    )
                    
                    advice = chat_completion.choices[0].message.content
                    st.success("Personalized Advice Generated:")
                    st.write(advice)

                except Exception as groq_err:
                    st.warning("LLM Advice currently unavailable. Check API Key.")
                    st.error(str(groq_err))

        except Exception as e:
            st.error(f"An error occurred in the pipeline: {e}")
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    # Landing Page State
    st.write("---")
    st.info("Waiting for a receipt upload to start the logical pipeline.")
