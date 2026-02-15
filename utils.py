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
    Preprocess receipt image for OCR with multiple approaches:
    - Grayscale conversion
    - Noise reduction
    - Contrast enhancement
    - Multiple thresholding methods
    - Deskewing
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get original dimensions
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Resize based on image quality - if image is too small, scale up
    if h < 800 or w < 600:
        scale_factor = max(2.0, 1500 / h)
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)

    # 2. Remove noise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 3. Deskew image
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)

    # 4. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Try multiple thresholding methods
    processed_images = []
    
    # Method 1: Adaptive Gaussian
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    processed_images.append(thresh1)
    
    # Method 2: Otsu's thresholding
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh2)
    
    # Method 3: Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    thresh3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    processed_images.append(thresh3)

    return processed_images

# ----------------------------
# OCR EXTRACTION FUNCTION
# ----------------------------
def extract_text(image_path: str) -> str:
    """
    Extract text from receipt image using Tesseract OCR.
    Tries multiple preprocessing methods and combines results.
    """
    processed_images = preprocess_image(image_path)
    
    # OCR configuration - more flexible character set
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,%/@$:- '
    
    all_text = []
    for processed in processed_images:
        text = pytesseract.image_to_string(processed, config=custom_config)
        all_text.append(text)
    
    # Combine results from different preprocessing methods
    # Keep the longest valid text (usually the best result)
    combined_text = max(all_text, key=lambda x: len(x.strip()))
    
    return combined_text

# ----------------------------
# ENHANCED PARSE RECEIPT FUNCTION
# ----------------------------
def parse_receipt(text: str) -> pd.DataFrame:
    """
    Enhanced receipt parser with better price detection and item extraction.
    """
    data = []
    
    # Improved price patterns
    price_patterns = [
        r'(\d+\.\d{2})$',  # Price at end of line: 12.34
        r'(\d+\.\d{2})\s*$',  # Price with trailing spaces
        r'(\d{1,3}(?:,\d{3})*\.\d{2})',  # Price with commas: 1,234.56
        r'(\d+\.\d{2})\s*(?=[A-Z]|$)',  # Price followed by letter or end
        r'(?:Rs\.?|PKR|Rs)\s*(\d+\.?\d*)',  # Price with currency prefix
        r'(\d+\.?\d*)\s*(?:Rs\.?|PKR)?$',  # Price with optional currency suffix
    ]
    
    # Stopwords to filter out non-item lines
    stopwords = [
        'total', 'subtotal', 'tax', 'change', 'thank', 'balance', 
        'date', 'tel', 'cashier', 'receipt', 'store', 'shop', 
        'bill', 'invoice', 'payment', 'cash', 'card', 'visa',
        'mastercard', 'amount', 'discount', 'saved', 'you pay'
    ]
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Skip lines containing stopwords
        lower_line = line.lower()
        if any(word in lower_line for word in stopwords):
            continue
        
        # Try to find price in the line
        price = None
        price_match_end = -1
        
        for pattern in price_patterns:
            matches = list(re.finditer(pattern, line))
            if matches:
                # Take the last match (likely the price at the end)
                last_match = matches[-1]
                price_str = last_match.group(1) if last_match.groups() else last_match.group(0)
                
                try:
                    # Clean up price string
                    price_str = price_str.replace(',', '').replace('Rs', '').replace('PKR', '').strip()
                    price_str = re.sub(r'[^\d.]', '', price_str)
                    price = float(price_str)
                    price_match_end = last_match.end()
                    break
                except ValueError:
                    continue
        
        if price is None or price == 0:
            continue
        
        # Extract item name (everything before the price)
        if price_match_end > 0:
            item = line[:price_match_end - len(price_str)].strip()
        else:
            # Fallback: remove price from the end
            item = re.sub(r'[\d\s.,]+$', '', line).strip()
        
        # Clean up item name
        item = re.sub(r'[^\w\s\-.]', '', item)  # Keep letters, numbers, spaces, hyphens, dots
        item = re.sub(r'\s+', ' ', item).strip()
        
        # Filter out items that are too short or look like codes
        if (len(item) >= 2 and 
            not item.isdigit() and 
            not re.match(r'^\d+$', item) and
            not any(word in item.lower() for word in ['total', 'subtotal', 'tax'])):
            
            # Additional validation: price should be reasonable
            if 0.01 <= price <= 10000:  # Adjust based on your typical prices
                data.append({'Item': item, 'Price': price})
    
    # Remove duplicates while preserving order
    seen = set()
    unique_data = []
    for item in data:
        key = (item['Item'].lower(), item['Price'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    df = pd.DataFrame(unique_data)
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Price'])
    
    return df

# ----------------------------
# CATEGORIZATION FUNCTION
# ----------------------------
def categorize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Enhanced categorization with more keywords and fuzzy matching.
    """
    if df.empty:
        return df, pd.Series(dtype=float)
    
    # Expanded category keywords
    category_keywords = {
        'Dairy & Eggs': [
            'milk', 'yogurt', 'yoghurt', 'cheese', 'butter', 'cream', 
            'paneer', 'egg', 'eggs', 'olpers', 'milkpak', 'dairy', 
            'curd', 'ghee', 'malai', 'lassi'
        ],
        'Bakery': [
            'bread', 'bun', 'buns', 'cake', 'cakes', 'biscuit', 
            'biscuits', 'rusk', 'donut', 'donuts', 'paratha', 'nan',
            'roti', 'chapati', 'pastry', 'cookie', 'cookies'
        ],
        'Snacks': [
            'chip', 'chips', 'choco', 'chocolate', 'candy', 'candies',
            'kurkure', 'slims', 'lays', 'nimko', 'bounty', 'snack',
            'snacks', 'namkeen', 'popcorn', 'crisps', 'wafers'
        ],
        'Meat & Fish': [
            'chicken', 'beef', 'mutton', 'fish', 'prawn', 'prawns',
            'kebab', 'kabab', 'nugget', 'nuggets', 'gosht', 'meat',
            'mince', 'qeema', 'steak', 'sausage', 'sausages'
        ],
        'Vegetables': [
            'tomato', 'tomatoes', 'onion', 'onions', 'potato', 'potatoes',
            'carrot', 'carrots', 'cabbage', 'ginger', 'garlic', 'adrak',
            'sabzi', 'vegetable', 'vegetables', 'lemon', 'lime', 'chili',
            'chilies', 'mirch', 'bhindi', 'tori', 'karela', 'pumpkin'
        ],
        'Fruits': [
            'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges',
            'grape', 'grapes', 'mango', 'mangoes', 'fruit', 'fruits',
            'strawberry', 'berries', 'kiwi', 'pineapple', 'melon'
        ],
        'Beverages': [
            'cola', 'juice', 'juices', 'water', 'mineral water', 'tea',
            'coffee', 'soda', 'pepsi', 'coke', 'nestle', 'tapal',
            'lipton', 'drink', 'drinks', 'soft drink', 'beverage',
            'soda water', 'squash', 'syrup'
        ],
        'Household': [
            'soap', 'soaps', 'detergent', 'tissue', 'tissues', 'surf',
            'shampoo', 'harpic', 'pampers', 'cleaner', 'cleaning',
            'bleach', 'sponge', 'brush', 'broom', 'mop', 'wash'
        ],
        'Pantry': [
            'oil', 'cooking oil', 'flour', 'ata', 'rice', 'chawal',
            'sugar', 'daal', 'dal', 'pulse', 'pulses', 'spice', 'spices',
            'masala', 'salt', 'pepper', 'vinegar', 'sauce', 'ketchup',
            'mayonnaise', 'pickle', 'achar'
        ],
        'Personal Care': [
            'toothpaste', 'toothbrush', 'soap', 'shampoo', 'conditioner',
            'cream', 'lotion', 'moisturizer', 'deodorant', 'perfume',
            'razor', 'blade', 'sanitary', 'diaper', 'wipes'
        ]
    }
    
    def get_category(item: str) -> str:
        item_lower = item.lower()
        # Remove common noise words
        item_lower = re.sub(r'\b(pkt|pack|kg|g|ml|ltr|each|box)\b', '', item_lower)
        
        # Check each category
        for cat, keywords in category_keywords.items():
            if any(k in item_lower for k in keywords):
                return cat
        
        # Additional checks based on price patterns
        return 'Other'
    
    df['Category'] = df['Item'].apply(get_category)
    category_totals = df.groupby('Category')['Price'].sum().round(2)
    
    return df, category_totals

# ----------------------------
# DEBUGGING AND VISUALIZATION
# ----------------------------
def ocr_debug(image_path: str):
    """
    Enhanced debugging function showing preprocessing steps and OCR results.
    """
    processed_images = preprocess_image(image_path)
    text = extract_text(image_path)
    
    # Display original image
    img = cv2.imread(image_path)
    cv2.imshow('Original', img)
    
    # Display processed images
    for i, proc in enumerate(processed_images):
        cv2.imshow(f'Processed {i+1}', proc)
    
    print("OCR Results:")
    print("-" * 50)
    print(text)
    print("-" * 50)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return processed_images, text

# ----------------------------
# MAIN PROCESSING FUNCTION
# ----------------------------
def process_receipt(image_path: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Complete receipt processing pipeline.
    """
    # Extract text
    text = extract_text(image_path)
    
    # Parse receipt
    df = parse_receipt(text)
    
    # Categorize items
    df, category_totals = categorize(df)
    
    return df, category_totals, text

# ----------------------------
# SAVE RESULTS FUNCTION
# ----------------------------
def save_results(df: pd.DataFrame, category_totals: pd.Series, output_file: str = 'receipt_results.xlsx'):
    """
    Save results to Excel file with multiple sheets.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Items', index=False)
        category_totals.to_excel(writer, sheet_name='Category Totals')
        
        # Add summary sheet
        summary = pd.DataFrame({
            'Total Items': [len(df)],
            'Total Amount': [df['Price'].sum()],
            'Number of Categories': [len(category_totals)]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

# ----------------------------
# USAGE EXAMPLE
# ----------------------------
if __name__ == "__main__":
    # Example usage
    image_path = "receipt.jpg"  # Replace with your image path
    
    try:
        # Process receipt
        df, category_totals, raw_text = process_receipt(image_path)
        
        # Print results
        print("\n" + "="*50)
        print("EXTRACTED ITEMS:")
        print("="*50)
        print(df.to_string())
        
        print("\n" + "="*50)
        print("CATEGORY TOTALS:")
        print("="*50)
        print(category_totals.to_string())
        
        print(f"\nTotal Amount: PKR {df['Price'].sum():.2f}")
        
        # Save results
        save_results(df, category_totals)
        print(f"\nResults saved to receipt_results.xlsx")
        
        # For debugging uncomment:
        # ocr_debug(image_path)
        
    except Exception as e:
        print(f"Error processing receipt: {str(e)}")
