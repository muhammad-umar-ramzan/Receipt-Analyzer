import cv2
import pytesseract
import pandas as pd
import re
from typing import Tuple
import numpy as np

# ----------------------------
# IMAGE PREPROCESSING FUNCTION
# ----------------------------
def preprocess_image(image_path: str):
    """
    Preprocess receipt image for OCR:
    - Grayscale
    - Resize for better OCR
    - Bilateral filter for noise reduction
    - Contrast enhancement (CLAHE)
    - Adaptive thresholding
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Resize to 2x for better OCR accuracy
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 2. Noise reduction with bilateral filter (preserves edges)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 5. Optional dilation to strengthen text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed = cv2.dilate(thresh, kernel, iterations=1)

    return processed

# ----------------------------
# OCR EXTRACTION FUNCTION
# ----------------------------
def extract_text(image_path: str) -> str:
    """
    Extract text from receipt image using Tesseract OCR.
    Uses a whitelist to include numbers, letters, %, /, and dots.
    """
    preprocessed = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,%/@'
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text

# ----------------------------
# PARSE RECEIPT FUNCTION
# ----------------------------
def parse_receipt(text: str) -> pd.DataFrame:
    """
    Parse OCR text to extract items and prices.
    Returns a DataFrame with ['Item', 'Price'] columns.
    """
    data = []
    price_pattern = re.compile(r'(\d+\.\d{2})|(\d{1,4})')  # price detection
    stopwords = ['total', 'subtotal', 'tax', 'change', 'thank', 'balance', 'date', 'tel', 'cashier']

    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 2:
            continue

        lower_line = line.lower()
        if any(word in lower_line for word in stopwords):
            continue

        matches = list(price_pattern.finditer(line))
        if not matches:
            continue

        last_match = matches[-1]
        price_str = last_match.group(0).replace(',', '')

        try:
            price_str = price_str.replace('O', '0').replace('o', '0')
            price = float(price_str)
        except ValueError:
            continue

        # item is everything before last price match
        item = line[:last_match.start()].strip()
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        item = re.sub(r'\s+', ' ', item).strip()

        if item:
            data.append({'Item': item, 'Price': price})

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Price'])
    return df

# ----------------------------
# CATEGORIZATION FUNCTION
# ----------------------------
def categorize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Categorize items into expense categories and compute totals.
    Returns updated DataFrame with 'Category' column and category totals.
    """
    if df.empty:
        return df, pd.Series(dtype=float)

    category_keywords = {
        'Dairy & Eggs': ['milk', 'yogurt', 'cheese', 'butter', 'cream', 'paneer', 'egg', 'olpers', 'milkpak'],
        'Bakery': ['bread', 'bun', 'cake', 'biscuit', 'rusk', 'donut', 'paratha', 'nan'],
        'Snacks': ['chip', 'choco', 'candy', 'kurkure', 'slims', 'lays', 'nimko', 'bounty'],
        'Meat & Fish': ['chicken', 'beef', 'mutton', 'fish', 'prawn', 'kebab', 'nugget', 'gosht'],
        'Vegetables': ['tomato', 'onion', 'potato', 'carrot', 'cabbage', 'ginger', 'garlic', 'adrak', 'sabzi'],
        'Beverages': ['cola', 'juice', 'water', 'tea', 'coffee', 'soda', 'pepsi', 'coke', 'nestle', 'tapal'],
        'Household': ['soap', 'detergent', 'tissue', 'surf', 'shampoo', 'harpic', 'pampers', 'cleaner'],
        'Pantry': ['oil', 'flour', 'ata', 'rice', 'chawal', 'sugar', 'daal', 'pulse', 'spice', 'masala']
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
# DEBUGGING HELP (OPTIONAL)
# ----------------------------
def ocr_debug(image_path: str):
    """
    Display preprocessed image and OCR text for debugging.
    """
    preprocessed = preprocess_image(image_path)
    text = extract_text(image_path)
    return preprocessed, text
