import streamlit as st
import pandas as pd
import os
from groq import Groq  
from utils import extract_text, parse_receipt, categorize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Receipt Analyzer", layout="wide", page_icon="💰")

st.title("AI-Powered Receipt Analyzer")
st.markdown("""
    **Overview:** Upload a receipt to automatically extract items and prices, categorize expenses, 
    and receive AI-driven financial insights.
""")

# --- UPLOAD RECEIPT ---
uploaded_file = st.file_uploader(
    "Upload a receipt image", 
    type=["png", "jpg", "jpeg"],
    help="Ensure the receipt is clear and well-lit for optimal OCR accuracy."
)

if uploaded_file is not None:
    temp_path = "temp_receipt.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Receipt", width=400)

    with st.spinner("Processing receipt..."):
        try:
            # --- OCR EXTRACTION ---
            raw_text = extract_text(temp_path)
            
            with st.expander("View Extracted Text"):
                st.text(raw_text)

            # --- PARSE RECEIPT ---
            df = parse_receipt(raw_text)

            if df.empty:
                st.error("No items or prices detected. Please upload a clearer image.")
            else:
                # --- CATEGORIZE EXPENSES ---
                df, category_totals = categorize(df)
                total_spend = category_totals.sum()

                # --- EXPENSE ANALYSIS ---
                analysis_df = pd.DataFrame(category_totals).reset_index()
                analysis_df.columns = ['Category', 'Amount']
                analysis_df['Percentage'] = (analysis_df['Amount'] / total_spend) * 100

                # Layout columns for visuals
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Itemized Breakdown")
                    st.table(df[['Item', 'Price', 'Category']])

                with col2:
                    st.subheader("Expense Analysis")
                    st.bar_chart(category_totals)

                    for _, row in analysis_df.iterrows():
                        st.write(f"**{row['Category']}:** {row['Percentage']:.1f}% of total")

                st.metric(label="Total Bill", value=f"${total_spend:,.2f}")

                # --- AI FINANCIAL ADVICE ---
                st.divider()
                st.subheader("AI Financial Consultant")

                try:
                    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                    context_data = analysis_df.to_string(index=False)
                    prompt = f"""
                    A user uploaded a receipt. Analyze the data below:
                    {context_data}
                    Total: ${total_spend:.2f}

                    Task:
                    1. Identify any category with unusually high expenses.
                    2. Provide 4-5 practical cost-saving suggestions.
                    3. Keep the tone professional and actionable.
                    """

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a professional financial advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.6
                    )
                    
                    advice = chat_completion.choices[0].message.content
                    st.success("AI Financial Advice:")
                    st.write(advice)

                except Exception as e:
                    st.warning("AI advice is currently unavailable.")
                    st.error(str(e))

        except Exception as e:
            st.error(f"Pipeline error: {e}")
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    st.write("---")
    st.info("Please upload a receipt to begin the analysis.")
