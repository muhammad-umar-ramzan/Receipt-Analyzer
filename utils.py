import cv2
import pytesseract
import pandas as pd
import re
import numpy as np
from typing import Tuple

# Tesseract Configuration

def preprocess_image(image_path: str):
    """
    HACKATHON TASK 1: Image Preprocessing
    Applies Grayscale, Noise Reduction, Contrast Enhancement, and Thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Reduction using Median Blur
    denoised = cv2.medianBlur(gray, 3)
    
    # 3. Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 4. Thresholding (Adaptive Gaussian for better visibility of faint text)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 5. Dilation to make characters slightly thicker for OCR
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed_img = cv2.dilate(thresh, kernel, iterations=1)
    
    return processed_img

def extract_text(image_path: str) -> str:
    """Task 1.3: OCR Extraction using Tesseract."""
    preprocessed = preprocess_image(image_path)
    
    # PSM 6: Assume a single uniform block of text (best for receipts)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text

def parse_receipt(text: str) -> pd.DataFrame:
    """
    TASK 2: Data Parsing & Structuring
    Cleans data and handles OCR errors.
    """
    data = []
    # Pattern to find numbers (prices) at the end of lines
    price_pattern = re.compile(r'(\d+[\.,]\d{2})|(\d{2,})')

    for line in text.split('\n'):
        line = line.strip()
        
        # Data Cleaning: Skip empty or short noise lines
        if not line or len(line) < 3:
            continue

        # Ignore common receipt non-item words
        lower = line.lower()
        stopwords = ['total', 'subtotal', 'tax', 'change', 'thank', 'balance', 'date', 'tel', 'cashier']
        if any(word in lower for word in stopwords):
            continue

        # Find all number-like matches
        matches = list(price_pattern.finditer(line))
        if not matches:
            continue

        # Logic: The last number in a receipt line is usually the price
        last_match = matches[-1]
        price_str = last_match.group(0).replace(',', '')
        
        try:
            # Handle potential OCR typos like 1.OO instead of 1.00
            price_str = price_str.replace('O', '0').replace('o', '0')
            price = float(price_str)
        except ValueError:
            continue

        # Item name is everything before the price
        item = line[:last_match.start()].strip()
        
        # Clean item name from noise (special characters)
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        item = re.sub(r'\s+', ' ', item).strip()

        if item:
            data.append({'Item': item, 'Price': price})

    df = pd.DataFrame(data)
    # Task 2.2: Handle empty data
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Price'])
    
    return df

def categorize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    TASK 3: Expense Categorization
    Classifies items into Pakistani household categories.
    """
    if df.empty:
        return df, pd.Series(dtype=float)

    # Expanded dictionary for local Islamabad context
    category_keywords = {
        'Dairy & Eggs': ['milk', 'yogurt', 'cheese', 'butter', 'cream', 'paneer', 'egg', 'anda', 'olpers', 'milkpak'],
        'Bakery':       ['bread', 'bun', 'cake', 'biscuit', 'rusk', 'donut', 'paratha', 'nan'],
        'Snacks':       ['chip', 'choco', 'candy', 'kurkure', 'slims', 'lays', 'nimko', 'bounty'],
        'Meat & Fish':  ['chicken', 'beef', 'mutton', 'fish', 'prawn', 'kebab', 'nugget', 'gosht'],
        'Vegetables':   ['tomato', 'onion', 'potato', 'carrot', 'cabbage', 'ginger', 'garlic', 'adrak', 'sabzi'],
        'Beverages':    ['cola', 'juice', 'water', 'tea', 'coffee', 'soda', 'pepsi', 'coke', 'nestle', 'tapal'],
        'Household':    ['soap', 'detergent', 'tissue', 'surf', 'shampoo', 'harpic', 'pampers', 'cleaner'],
        'Pantry':       ['oil', 'flour', 'ata', 'rice', 'chawal', 'sugar', 'daal', 'pulse', 'spice', 'masala']
    }

    def get_category(item: str) -> str:
        item_lower = item.lower()
        for cat, keywords in category_keywords.items():
            if any(k in item_lower for k in keywords):
                return cat
        return 'Other'

    df['Category'] = df['Item'].apply(get_category)
    
    # Task 3.2: Calculate totals
    category_totals = df.groupby('Category')['Price'].sum().round(2)
    
    return df, category_totals
