import cv2
import pytesseract
import pandas as pd
import re
from typing import Tuple

def preprocess_image(image_path: str):
    """
    Preprocess the image for OCR:
    1. Grayscale conversion
    2. Noise reduction (Median Blur)
    3. Contrast enhancement (CLAHE)
    4. Adaptive thresholding
    5. Dilation to strengthen text for OCR
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed_img = cv2.dilate(thresh, kernel, iterations=1)
    
    return processed_img

def extract_text(image_path: str) -> str:
    """
    Perform OCR using pytesseract.
    Automatically detects tesseract path (works on Linux & Streamlit Cloud).
    """
    preprocessed = preprocess_image(image_path)
    config = r'--oem 3 --psm 6'  # single uniform block of text
    return pytesseract.image_to_string(preprocessed, config=config)

def parse_receipt(text: str) -> pd.DataFrame:
    """
    Parse OCR text to extract items and prices.
    Returns a DataFrame with columns: ['Item', 'Price']
    """
    data = []
    price_pattern = re.compile(r'(\d+[\.,]\d{2})|(\d{2,})')

    stopwords = ['total', 'subtotal', 'tax', 'change', 'thank', 'balance', 'date', 'tel', 'cashier']

    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 3:
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

        item = line[:last_match.start()].strip()
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        item = re.sub(r'\s+', ' ', item).strip()

        if item:
            data.append({'Item': item, 'Price': price})

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Price'])
    
    return df

def categorize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Categorize items into expense categories and compute totals per category.
    Returns updated DataFrame with 'Category' column and category totals Series.
    """
    if df.empty:
        return df, pd.Series(dtype=float)

    category_keywords = {
        'Dairy & Eggs': ['milk', 'yogurt', 'cheese', 'butter', 'cream', 'paneer', 'egg', 'olpers', 'milkpak'],
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
            if any(keyword in item_lower for keyword in keywords):
                return cat
        return 'Other'

    df['Category'] = df['Item'].apply(get_category)
    category_totals = df.groupby('Category')['Price'].sum().round(2)
    
    return df, category_totals
