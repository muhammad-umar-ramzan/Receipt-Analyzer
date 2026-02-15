import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from groq import Groq  
import pytesseract
import re
from typing import Tuple
import tempfile
from PIL import Image

# ----------------------------
# IMAGE PREPROCESSING FUNCTION
# ----------------------------
def preprocess_image(image_path: str):
    """
    Preprocess receipt image for OCR
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase image size for better OCR
    height, width = gray.shape
    if height < 1000:
        scale_factor = 2.0
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)
    
    # Remove noise
    gray = cv2.medianBlur(gray, 1)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# ----------------------------
# OCR EXTRACTION FUNCTION
# ----------------------------
def extract_text(image_path: str) -> str:
    """
    Extract text from receipt image
    """
    preprocessed = preprocess_image(image_path)
    
    # Try different OCR modes
    configs = [
        r'--oem 3 --psm 6',  # Assume uniform block
        r'--oem 3 --psm 4',  # Assume variable text
        r'--oem 3 --psm 3',  # Fully automatic
    ]
    
    best_text = ""
    max_length = 0
    
    for config in configs:
        text = pytesseract.image_to_string(preprocessed, config=config)
        if len(text.strip()) > max_length:
            max_length = len(text.strip())
            best_text = text
    
    return best_text

# ----------------------------
# RECEIPT PARSING FUNCTION
# ----------------------------
def parse_receipt(text: str) -> pd.DataFrame:
    """
    Parse OCR text to extract items and prices
    """
    data = []
    
    # Price patterns
    price_patterns = [
        r'[€$£]\s*(\d+(?:\.\d{2})?)',  # €60, $60, £60
        r'(\d+(?:\.\d{2})?)\s*[€$£]',  # 60€, 60$, 60£
        r'(\d+(?:\.\d{2})?)\s*$',  # Number at end
        r'(\d+\.\d{2})',  # Decimal number
    ]
    
    # Skip header lines
    skip_patterns = [
        r'total', r'subtotal', r'tax', r'change', r'thank', r'balance',
        r'date', r'tel', r'cashier', r'receipt', r'store', r'shop',
        r'bill', r'invoice', r'payment', r'cash', r'card',
        r'materials', r'tools', r'price', r'list', r'bom', r'item',
        r'description', r'estimated', r'---', r'\|', r'table'
    ]
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Skip lines with skip patterns
        lower_line = line.lower()
        if any(re.search(pattern, lower_line) for pattern in skip_patterns):
            continue
        
        # Find price
        price = None
        price_match_end = -1
        currency = '€'  # Default currency
        
        for pattern in price_patterns:
            matches = list(re.finditer(pattern, line, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                price_str = last_match.group(1) if last_match.groups() else last_match.group(0)
                
                try:
                    # Detect currency
                    if '€' in line:
                        currency = '€'
                    elif '$' in line:
                        currency = '$'
                    elif '£' in line:
                        currency = '£'
                    
                    # Clean price
                    price_str = re.sub(r'[^\d.]', '', price_str)
                    if price_str:
                        price = float(price_str)
                        price_match_end = last_match.end()
                        break
                except ValueError:
                    continue
        
        if price is None or price == 0:
            continue
        
        # Extract item name
        if price_match_end > 0:
            item = line[:price_match_end - len(price_str)].strip()
        else:
            item = re.sub(r'[\d\s.,€$£]+$', '', line).strip()
        
        # Clean item
        item = re.sub(r'[^\w\s\-.]', ' ', item)
        item = re.sub(r'\s+', ' ', item).strip()
        
        # Validate
        if (len(item) >= 2 and 
            not item.isdigit() and 
            0.01 <= price <= 10000):
            data.append({
                'Item': item,
                'Price': price,
                'Currency': currency
            })
    
    # If no data found, try alternative parsing for BOM format
    if not data:
        data = parse_bom_format(lines)
    
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Price', 'Currency'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Item', 'Price'])
    
    return df

def parse_bom_format(lines):
    """Parse Bill of Materials format"""
    data = []
    
    for line in lines:
        line = line.strip()
        # Split on multiple spaces
        parts = re.split(r'\s{2,}', line)
        
        if len(parts) >= 2:
            last_part = parts[-1]
            price_match = re.search(r'[€$£]?\s*(\d+(?:\.\d{2})?)\s*[€$£]?', last_part)
            
            if price_match:
                try:
                    price = float(price_match.group(1))
                    item = parts[0].strip()
                    
                    # Detect currency
                    currency = '€'
                    if '$' in last_part:
                        currency = '$'
                    elif '£' in last_part:
                        currency = '£'
                    
                    if item and 0 < price < 1000:
                        data.append({
                            'Item': item,
                            'Price': price,
                            'Currency': currency
                        })
                except:
                    continue
    
    return data

# ----------------------------
# CATEGORIZATION FUNCTION
# ----------------------------
def categorize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Categorize items into expense categories
    """
    if df.empty:
        return df, pd.Series(dtype=float)
    
    # Category keywords
    category_keywords = {
        'Electronics': [
            'raspberry', 'pi', 'arduino', 'led', 'lcd', 'display', 
            'camera', 'c920', 'logitech', 'sensor', 'module', 'board',
            'microcontroller', 'processor', 'power supply'
        ],
        'Components': [
            'resistor', 'capacitor', 'diode', 'transistor', 'ic', 'chip',
            'jumper', 'wire', 'cable', 'connector', 'breadboard'
        ],
        'Hardware': [
            'screw', 'nut', 'bolt', 'mount', 'bracket', 'enclosure',
            'box', 'case', 'multiplex', 'acrylic', 'wood'
        ],
        'Tools': [
            'soldering', 'iron', 'multimeter', 'pliers', 'cutter',
            'screwdriver', 'drill', 'glue', 'tape', 'adhesive', 'gorilla'
        ],
        'Groceries': [
            'milk', 'bread', 'egg', 'cheese', 'vegetable', 'fruit',
            'meat', 'chicken', 'rice', 'oil', 'sugar', 'salt'
        ],
        'Beverages': [
            'water', 'juice', 'soda', 'cola', 'tea', 'coffee', 'drink'
        ],
        'Household': [
            'soap', 'shampoo', 'detergent', 'cleaner', 'tissue',
            'paper', 'towel', 'sponge'
        ]
    }
    
    def get_category(item: str) -> str:
        item_lower = item.lower()
        for cat, keywords in category_keywords.items():
            if any(k in item_lower for k in keywords):
                return cat
        return 'Other'
    
    df['Category'] = df['Item'].apply(get_category)
    category_totals = df.groupby('Category')['Price'].sum().round(2)
    
    return df, category_totals

# ----------------------------
# STREAMLIT APP
# ----------------------------

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Receipt Analyzer", 
    layout="wide", 
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 AI-Powered Receipt Analyzer</h1>
    <p>Upload a receipt to automatically extract items, categorize expenses, and receive AI-driven financial insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI-powered tool helps you:
    - 📸 Extract text from receipt images
    - 📊 Automatically categorize expenses
    - 💡 Get personalized financial advice
    
    **Supported currencies:** €, $, £
    """)
    
    st.divider()
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload a clear receipt image
    2. Wait for OCR processing
    3. Review extracted items
    4. Get AI financial insights
    """)
    
    st.divider()
    
    # Example data
    if st.button("📋 Load Example BOM"):
        st.session_state['example_data'] = pd.DataFrame({
            'Item': ['Raspberry Pi 5', 'RGB LED', 'LCD Display', 'Logitech C920', 'Power Supply'],
            'Price': [60.00, 2.00, 5.00, 70.00, 10.00],
            'Currency': ['€', '€', '€', '€', '€'],
            'Category': ['Electronics', 'Components', 'Electronics', 'Electronics', 'Electronics']
        })
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Receipt Image", 
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        help="Ensure the receipt is clear and well-lit for optimal OCR accuracy"
    )

with col2:
    st.markdown("### 📊 Quick Stats")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.metric("Total Items", len(df))
        st.metric("Total Amount", f"€{df['Price'].sum():.2f}")
        st.metric("Categories", len(df['Category'].unique()))

# Check for example data in session state
if 'example_data' in st.session_state:
    df = st.session_state['example_data']
    category_totals = df.groupby('Category')['Price'].sum()
    total_spend = category_totals.sum()
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Itemized Breakdown")
        st.dataframe(
            df[['Item', 'Price', 'Currency', 'Category']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Expense Analysis")
        
        # Create pie chart
        import plotly.express as px
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title="Expense Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.metric(
        label="💰 Total Bill", 
        value=f"€{total_spend:,.2f}",
        delta=f"{len(df)} items"
    )
    
    # Clear example button
    if st.button("Clear Example"):
        del st.session_state['example_data']
        st.rerun()

elif uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Receipt", width=400)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔄 Processing receipt..."):
        try:
            # Step 1: OCR Extraction
            status_text.text("📝 Extracting text from image...")
            progress_bar.progress(25)
            raw_text = extract_text(temp_path)
            
            with st.expander("🔍 View Extracted Text"):
                st.text(raw_text)

            # Step 2: Parse Receipt
            status_text.text("🔎 Parsing items and prices...")
            progress_bar.progress(50)
            df = parse_receipt(raw_text)

            if df.empty:
                st.error("❌ No items or prices detected. Please upload a clearer image.")
                
                # Offer manual entry option
                with st.expander("📝 Manual Entry"):
                    st.markdown("Enter items manually:")
                    item_name = st.text_input("Item Name")
                    item_price = st.number_input("Price (€)", min_value=0.0, step=0.01)
                    if st.button("Add Item"):
                        st.info("Manual entry feature coming soon!")
            else:
                # Step 3: Categorize
                status_text.text("📊 Categorizing expenses...")
                progress_bar.progress(75)
                df, category_totals = categorize(df)
                
                # Store in session state
                st.session_state['df'] = df
                st.session_state['category_totals'] = category_totals
                
                total_spend = category_totals.sum()

                # Layout columns for visuals
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📋 Itemized Breakdown")
                    
                    # Format price display
                    display_df = df[['Item', 'Price', 'Currency', 'Category']].copy()
                    display_df['Price'] = display_df.apply(
                        lambda x: f"{x['Currency']}{x['Price']:.2f}", axis=1
                    )
                    
                    st.dataframe(
                        display_df[['Item', 'Price', 'Category']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Item": "Item Name",
                            "Price": "Amount",
                            "Category": "Category"
                        }
                    )

                with col2:
                    st.subheader("📊 Expense Analysis")
                    
                    # Create bar chart
                    chart_data = pd.DataFrame({
                        'Category': category_totals.index,
                        'Amount': category_totals.values
                    })
                    
                    st.bar_chart(chart_data.set_index('Category'))
                    
                    # Category breakdown
                    analysis_df = pd.DataFrame(category_totals).reset_index()
                    analysis_df.columns = ['Category', 'Amount']
                    analysis_df['Percentage'] = (analysis_df['Amount'] / total_spend) * 100
                    
                    for _, row in analysis_df.iterrows():
                        st.markdown(f"""
                        <div style='padding: 10px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px;'>
                            <strong>{row['Category']}:</strong> €{row['Amount']:.2f} ({row['Percentage']:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)

                st.metric(
                    label="💰 Total Bill", 
                    value=f"€{total_spend:,.2f}",
                    delta=f"{len(df)} items"
                )

                # Step 4: AI Financial Advice
                progress_bar.progress(90)
                status_text.text("🤖 Generating AI insights...")
                
                st.divider()
                st.subheader("🤖 AI Financial Consultant")

                # Check for Groq API key
                groq_api_key = os.getenv("GROQ_API_KEY")
                
                if not groq_api_key:
                    st.warning("⚠️ Groq API key not found. Please set the GROQ_API_KEY environment variable for AI insights.")
                    
                    # Demo advice
                    with st.expander("🔍 View Demo Analysis"):
                        st.markdown("""
                        **Sample Financial Insights:**
                        
                        1. **Electronics (€60)** - Consider buying in bulk for discounts
                        2. **Components (€5)** - Look for starter kits that bundle items
                        3. **Tools (€5)** - Check if these tools are available for borrowing
                        
                        *Set up Groq API key to get personalized AI advice!*
                        """)
                else:
                    try:
                        client = Groq(api_key=groq_api_key)

                        context_data = analysis_df.to_string(index=False)
                        
                        # Create a more detailed prompt
                        prompt = f"""
                        Analyze this receipt data and provide financial advice:

                        RECEIPT SUMMARY:
                        {context_data}
                        
                        Total Spent: €{total_spend:.2f}
                        
                        ITEMS PURCHASED:
                        {df[['Item', 'Price', 'Category']].to_string(index=False)}

                        Please provide:
                        1. A brief analysis of spending patterns
                        2. Identify the top expense category
                        3. Suggest 3-5 practical money-saving tips
                        4. Any recommendations for future purchases

                        Keep the response concise, professional, and actionable.
                        """

                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a professional financial advisor. Provide clear, practical advice."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama-3.1-8b-instant",
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        advice = chat_completion.choices[0].message.content
                        
                        # Display advice in a nice box
                        st.success("💡 AI Financial Advice:")
                        st.markdown(f"""
                        <div style='padding: 20px; background-color: #e8f4f8; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                            {advice}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add download button for analysis
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis as CSV",
                            data=csv,
                            file_name="receipt_analysis.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.warning("⚠️ AI advice is currently unavailable.")
                        if st.checkbox("Show error details"):
                            st.error(str(e))

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            if st.checkbox("Show detailed error"):
                st.exception(e)
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            status_text.empty()
            progress_bar.empty()

else:
    # Welcome message when no file uploaded
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>📸 Step 1</h3>
            <p>Upload a clear photo of your receipt</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>🔍 Step 2</h3>
            <p>AI extracts items and prices</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>💡 Step 3</h3>
            <p>Get personalized financial insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👆 Please upload a receipt image to begin the analysis")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Powered by Tesseract OCR & Groq AI | © 2024 AI Receipt Analyzer</p>
</div>
""", unsafe_allow_html=True)
