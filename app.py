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
import plotly.express as px
import time

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
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
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        padding: 20px;
        background-color: white;
        color: black;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .category-box {
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f2f6;
        color:black;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    footer {
        text-align: center;
        padding: 20px;
        color: gray;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# IMAGE PREPROCESSING FUNCTION
# ----------------------------
def preprocess_image(image_path: str):
    """
    Preprocess receipt image for OCR
    """
    try:
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
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

# ----------------------------
# OCR EXTRACTION FUNCTION
# ----------------------------
def extract_text(image_path: str) -> str:
    """
    Extract text from receipt image
    """
    try:
        preprocessed = preprocess_image(image_path)
        if preprocessed is None:
            return ""
        
        # For deployment, use a simpler config
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(preprocessed, config=config)
        
        return text
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

# ----------------------------
# RECEIPT PARSING FUNCTION
# ----------------------------
def parse_receipt(text: str) -> pd.DataFrame:
    """
    Parse OCR text to extract items and prices
    """
    data = []
    
    if not text or len(text.strip()) < 10:
        return pd.DataFrame(columns=['Item', 'Price', 'Currency'])
    
    # Price patterns
    price_patterns = [
        r'[€$£]\s*(\d+(?:\.\d{2})?)',  # €60, $60, £60
        r'(\d+(?:\.\d{2})?)\s*[€$£]',  # 60€, 60$, 60£
        r'(\d+\.\d{2})',  # Decimal number
    ]
    
    # Skip header lines
    skip_patterns = [
        r'total', r'subtotal', r'tax', r'change', r'thank', 
        r'date', r'tel', r'cashier', r'receipt', r'store',
        r'materials', r'tools', r'price', r'list', r'bom',
        r'description', r'estimated', r'---', r'\|'
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
                        break
                except ValueError:
                    continue
        
        if price is None or price == 0:
            continue
        
        # Extract item name
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
    
    # If no data found, try BOM format
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
        # Look for patterns like "Item    Price"
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
            'camera', 'logitech', 'sensor', 'module', 'board',
            'processor', 'power', 'rgb', 'c920'
        ],
        'Components': [
            'resistor', 'capacitor', 'diode', 'transistor', 'ic', 
            'jumper', 'wire', 'cable', 'connector', 'breadboard'
        ],
        'Hardware': [
            'screw', 'nut', 'bolt', 'mount', 'enclosure',
            'box', 'case', 'multiplex', 'wood', 'acrylic'
        ],
        'Tools': [
            'soldering', 'iron', 'multimeter', 'pliers', 'cutter',
            'screwdriver', 'drill', 'glue', 'tape', 'adhesive', 'gorilla'
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

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 AI-Powered Receipt Analyzer</h1>
    <p style='font-size: 1.2em;'>Upload a receipt to automatically extract items, categorize expenses, and receive AI-driven financial insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/receipt.png", width=80)
    st.markdown("## 📋 About")
    st.markdown("""
    This AI-powered tool helps you:
    - 📸 Extract text from receipt images
    - 📊 Automatically categorize expenses
    - 💡 Get personalized financial advice
    
    **Supported Currencies:**
    - Euro (€)
    - Dollar ($)
    - Pound (£)
    """)
    
    st.divider()
    
    st.markdown("## 📝 Quick Start")
    st.markdown("""
    1. Click 'Browse files' to upload
    2. Wait for processing
    3. Review your analysis
    4. Download results
    """)
    
    st.divider()
    
    # Example BOM button
    if st.button("📋 Load Example BOM", use_container_width=True):
        example_data = [
            {'Item': 'Raspberry Pi 5', 'Price': 60.00, 'Currency': '€', 'Category': 'Electronics'},
            {'Item': 'RGB LED', 'Price': 2.00, 'Currency': '€', 'Category': 'Components'},
            {'Item': 'LCD 16x2 Display', 'Price': 5.00, 'Currency': '€', 'Category': 'Electronics'},
            {'Item': 'Logitech C920', 'Price': 70.00, 'Currency': '€', 'Category': 'Electronics'},
            {'Item': 'Jumper Wires', 'Price': 3.00, 'Currency': '€', 'Category': 'Components'},
            {'Item': 'Power Supply', 'Price': 10.00, 'Currency': '€', 'Category': 'Electronics'},
            {'Item': 'Multiplex 8mm Box', 'Price': 15.00, 'Currency': '€', 'Category': 'Hardware'},
            {'Item': 'Gorilla Glue', 'Price': 5.00, 'Currency': '€', 'Category': 'Tools'}
        ]
        st.session_state['df'] = pd.DataFrame(example_data)
        st.session_state['example_loaded'] = True
        st.rerun()

# Check for example data
if 'example_loaded' in st.session_state and st.session_state['example_loaded']:
    df = st.session_state['df']
    category_totals = df.groupby('Category')['Price'].sum()
    total_spend = category_totals.sum()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", len(df))
    with col2:
        st.metric("Total Amount", f"€{total_spend:.2f}")
    with col3:
        st.metric("Categories", len(df['Category'].unique()))
    with col4:
        st.metric("Avg Item Price", f"€{total_spend/len(df):.2f}")
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Itemized Breakdown")
        display_df = df[['Item', 'Price', 'Currency', 'Category']].copy()
        display_df['Price'] = display_df.apply(
            lambda x: f"{x['Currency']}{x['Price']:.2f}", axis=1
        )
        st.dataframe(
            display_df[['Item', 'Price', 'Category']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Expense Analysis")
        
        # Create pie chart
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title="Expense Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        for category, amount in category_totals.items():
            percentage = (amount/total_spend)*100
            st.markdown(f"""
            <div class="category-box">
                <strong>{category}:</strong> €{amount:.2f} ({percentage:.1f}%)
            </div>
            """, unsafe_allow_html=True)
    
    # Clear example button
    if st.button("Clear Example", use_container_width=True):
        del st.session_state['example_loaded']
        del st.session_state['df']
        st.rerun()

# Main upload section
else:
    uploaded_file = st.file_uploader(
        "📤 Upload Receipt Image", 
        type=["png", "jpg", "jpeg"],
        help="Ensure the receipt is clear and well-lit for optimal OCR accuracy"
    )

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Display image
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Receipt", width=400)
        
        with col2:
            st.info("🔄 Processing receipt...")
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            # Step 1: OCR
            status_text.text("📝 Extracting text from image...")
            progress_bar.progress(25)
            raw_text = extract_text(temp_path)
            
            with st.expander("🔍 View Extracted Text"):
                st.text(raw_text if raw_text else "No text extracted")

            # Step 2: Parse
            status_text.text("🔎 Parsing items and prices...")
            progress_bar.progress(50)
            df = parse_receipt(raw_text)

            if df.empty:
                st.error("❌ No items or prices detected. Try:")
                st.markdown("""
                - Uploading a clearer image
                - Using the Example BOM button
                - Checking if the receipt is readable
                """)
            else:
                # Step 3: Categorize
                status_text.text("📊 Categorizing expenses...")
                progress_bar.progress(75)
                df, category_totals = categorize(df)
                
                # Store in session
                st.session_state['df'] = df
                
                total_spend = category_totals.sum()
                
                # Show metrics
                st.success("✅ Processing complete!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Items Found", len(df))
                with col2:
                    st.metric("Total Amount", f"€{total_spend:.2f}")
                with col3:
                    st.metric("Categories", len(category_totals))

                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 Itemized Breakdown")
                    display_df = df[['Item', 'Price', 'Currency', 'Category']].copy()
                    display_df['Price'] = display_df.apply(
                        lambda x: f"{x['Currency']}{x['Price']:.2f}", axis=1
                    )
                    st.dataframe(
                        display_df[['Item', 'Price', 'Category']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.subheader("📊 Expense Analysis")
                    
                    if not category_totals.empty:
                        fig = px.bar(
                            x=category_totals.index,
                            y=category_totals.values,
                            title="Expenses by Category",
                            labels={'x': 'Category', 'y': 'Amount (€)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        for category, amount in category_totals.items():
                            percentage = (amount/total_spend)*100
                            st.markdown(f"""
                            <div class="category-box">
                                <strong>{category}:</strong> €{amount:.2f} ({percentage:.1f}%)
                            </div>
                            """, unsafe_allow_html=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis (CSV)",
                    data=csv,
                    file_name="receipt_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # AI Advice (optional)
                st.divider()
                st.subheader("🤖 AI Financial Insights")
                
                groq_api_key = os.getenv("GROQ_API_KEY")
                
                if groq_api_key:
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        analysis_text = "\n".join([
                            f"- {cat}: €{amt:.2f}" 
                            for cat, amt in category_totals.items()
                        ])
                        
                        prompt = f"""
                        Analyze this receipt:
                        
                        Total: €{total_spend:.2f}
                        Categories:
                        {analysis_text}
                        
                        Items:
                        {df[['Item', 'Price']].to_string(index=False)}
                        
                        Provide brief financial advice (max 100 words):
                        """
                        
                        with st.spinner("Generating insights..."):
                            response = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": "You are a helpful financial advisor."},
                                    {"role": "user", "content": prompt}
                                ],
                                model="llama-3.1-8b-instant",
                                temperature=0.7,
                                max_tokens=200
                            )
                            
                            advice = response.choices[0].message.content
                            st.markdown(f'<div class="success-box">{advice}</div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.info("AI insights temporarily unavailable")
                
                else:
                    st.info("💡 Set up GROQ_API_KEY for AI-powered insights")

        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        finally:
            # Cleanup
            progress_bar.progress(100)
            status_text.text("✅ Done!")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

# Footer
st.divider()
st.markdown("""
<footer>
    <p>Powered by Tesseract OCR & Groq AI | Made with Streamlit</p>
    <p>⭐ Star on GitHub | Report Issues | Request Features</p>
</footer>
""", unsafe_allow_html=True)
