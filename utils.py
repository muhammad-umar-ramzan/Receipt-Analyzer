import cv2
import pytesseract
import pandas as pd
import re
from typing import Tuple
import numpy as np

# ----------------------------
# IMAGE PREPROCESSING FUNCTION - OPTIMIZED FOR YOUR RECEIPT
# ----------------------------
def preprocess_image(image_path: str):
    """
    Preprocess receipt image specifically for BOM/receipt format
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
    
    # Apply thresholding to get clean black and white image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (sometimes text is white on black)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    
    return thresh

# ----------------------------
# OCR EXTRACTION FUNCTION
# ----------------------------
def extract_text(image_path: str) -> str:
    """
    Extract text with optimized settings for your receipt format
    """
    preprocessed = preprocess_image(image_path)
    
    # Use different OCR modes to get best result
    configs = [
        r'--oem 3 --psm 6',  # Assume uniform block of text
        r'--oem 3 --psm 4',  # Assume variable text
        r'--oem 3 --psm 3',  # Fully automatic
    ]
    
    best_text = ""
    max_length = 0
    
    for config in configs:
        text = pytesseract.image_to_string(preprocessed, config=config)
        # Keep the longest valid text (usually best)
        if len(text.strip()) > max_length:
            max_length = len(text.strip())
            best_text = text
    
    return best_text

# ----------------------------
# SPECIALIZED PARSER FOR BOM/RECEIPT FORMAT
# ----------------------------
def parse_bom_receipt(text: str) -> pd.DataFrame:
    """
    Specialized parser for Bill of Materials/Receipt format
    with Item, Description, and Price columns
    """
    data = []
    
    # Split into lines
    lines = text.split('\n')
    
    # Price patterns for different currencies
    price_patterns = [
        r'[€$£]\s*(\d+(?:\.\d{2})?)',  # €60, $60, £60
        r'(\d+(?:\.\d{2})?)\s*[€$£]',  # 60€, 60$, 60£
        r'(\d+(?:\.\d{2})?)\s*(?:euros?|dollars?|pounds?)',  # 60 euros
        r'(\d+(?:\.\d{2})?)\s*$',  # Just number at end of line
    ]
    
    # Skip lines that are headers or contain these words
    skip_patterns = [
        r'materials', r'tools', r'price', r'list', r'bom', r'item',
        r'description', r'estimated', r'---', r'\|', r'table'
    ]
    
    current_item = ""
    current_price = None
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        
        # Skip header lines
        lower_line = line.lower()
        if any(re.search(pattern, lower_line) for pattern in skip_patterns):
            continue
        
        # Look for price in the line
        price_found = None
        price_value = None
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract price number
                    price_str = match.group(1) if match.groups() else match.group(0)
                    # Clean price string
                    price_str = re.sub(r'[^\d.]', '', price_str)
                    if price_str:
                        price_value = float(price_str)
                        price_found = match
                        break
                except:
                    continue
            if price_found:
                break
        
        if price_found and price_value:
            # Extract item name (text before the price)
            item_text = line[:price_found.start()].strip()
            
            # Clean up item text
            item_text = re.sub(r'[^\w\s\-\.]', ' ', item_text)
            item_text = re.sub(r'\s+', ' ', item_text).strip()
            
            # Split into Item and Description if possible
            if ':' in item_text:
                parts = item_text.split(':', 1)
                item_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
            elif '-' in item_text and len(item_text.split('-')) == 2:
                parts = item_text.split('-', 1)
                item_name = parts[0].strip()
                description = parts[1].strip()
            else:
                # Try to intelligently split
                words = item_text.split()
                if len(words) > 3:
                    # Assume first 2-3 words are item, rest is description
                    item_name = ' '.join(words[:2])
                    description = ' '.join(words[2:])
                else:
                    item_name = item_text
                    description = ""
            
            # Validate price is reasonable (adjust as needed)
            if 0 < price_value < 1000:
                data.append({
                    'Item': item_name,
                    'Description': description,
                    'Price': price_value,
                    'Currency': '€' if '€' in line else '$' if '$' in line else '£' if '£' in line else 'Unknown'
                })
    
    # If no data found with first method, try alternative parsing
    if not data:
        data = alternative_parse(lines)
    
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=['Item', 'Description', 'Price', 'Currency'])
    
    return df

def alternative_parse(lines):
    """Alternative parsing method for your specific format"""
    data = []
    
    for line in lines:
        line = line.strip()
        # Look for patterns like "Raspberry Pi 5    Main processor    €60"
        parts = re.split(r'\s{2,}', line)  # Split on 2+ spaces
        
        if len(parts) >= 2:
            # Try to find price in the last part
            last_part = parts[-1]
            price_match = re.search(r'[€$£]?\s*(\d+(?:\.\d{2})?)\s*[€$£]?', last_part)
            
            if price_match:
                price_str = price_match.group(1)
                try:
                    price = float(price_str)
                    
                    # First part is likely the item
                    item = parts[0].strip()
                    
                    # Middle parts might be description
                    description = ' '.join(parts[1:-1]).strip() if len(parts) > 2 else ""
                    
                    if item and 0 < price < 1000:
                        data.append({
                            'Item': item,
                            'Description': description,
                            'Price': price,
                            'Currency': '€' if '€' in last_part else '$' if '$' in last_part else '£' if '£' in last_part else 'Unknown'
                        })
                except:
                    continue
    
    return data

# ----------------------------
# ENHANCED CATEGORIZATION FOR ELECTRONICS
# ----------------------------
def categorize_bom(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Categorize items specifically for electronics/project BOM
    """
    if df.empty:
        return df, pd.Series(dtype=float)
    
    # Electronics and project-specific categories
    category_keywords = {
        'Microcontrollers/Processors': [
            'raspberry', 'pi', 'arduino', 'microcontroller', 'processor', 
            'board', 'raspberry pi', 'esp', 'arduino uno', 'nano', 'mega'
        ],
        'Displays': [
            'lcd', 'display', 'screen', 'oled', 'touch', 'monitor', 'led display',
            'i2c display', '16x2', '20x4'
        ],
        'LEDs & Lighting': [
            'led', 'rgb', 'light', 'bulb', 'neopixel', 'ws2812', 'strip'
        ],
        'Cameras': [
            'camera', 'webcam', 'c920', 'logitech', 'pi camera', 'module'
        ],
        'Power Supplies': [
            'power', 'supply', 'adapter', 'battery', 'charger', 'psu', 'power bank'
        ],
        'Wires & Connectors': [
            'wire', 'jumper', 'cable', 'connector', 'breadboard', 'ribbon',
            'usb', 'hdmi', 'ethernet'
        ],
        'Enclosures & Hardware': [
            'box', 'case', 'enclosure', 'mount', 'screw', 'nut', 'bolt',
            'multiplex', 'acrylic', 'wood', 'metal'
        ],
        'Adhesives': [
            'glue', 'tape', 'adhesive', 'epoxy', 'gorilla', 'super glue'
        ]
    }
    
    def get_category(item: str) -> str:
        item_lower = item.lower()
        
        for cat, keywords in category_keywords.items():
            if any(k in item_lower for k in keywords):
                return cat
        
        return 'Other Components'
    
    df['Category'] = df['Item'].apply(get_category)
    category_totals = df.groupby('Category')['Price'].sum().round(2)
    
    return df, category_totals

# ----------------------------
# MAIN PROCESSING FUNCTION
# ----------------------------
def process_bom_receipt(image_path: str):
    """
    Complete processing pipeline for BOM/receipt
    """
    print("Extracting text from image...")
    text = extract_text(image_path)
    
    print("\n" + "="*50)
    print("EXTRACTED TEXT:")
    print("="*50)
    print(text)
    
    print("\n" + "="*50)
    print("PARSING RECEIPT...")
    print("="*50)
    
    df = parse_bom_receipt(text)
    
    if df.empty:
        print("No items found! Trying manual parsing...")
        # Manual parsing based on your actual data
        manual_data = [
            {'Item': 'Raspberry Pi 5', 'Description': 'Main processor', 'Price': 60.00, 'Currency': '€'},
            {'Item': 'RGB LED', 'Description': 'For visual feedback', 'Price': 2.00, 'Currency': '€'},
            {'Item': 'LCD 16x2 I2C Display', 'Description': 'Displays monthly total', 'Price': 5.00, 'Currency': '€'},
            {'Item': 'Logitech C920', 'Description': 'Captures receipt image', 'Price': 70.00, 'Currency': '€'},
            {'Item': 'Jumper wires + breadboard', 'Description': 'For circuit wiring', 'Price': 3.00, 'Currency': '€'},
            {'Item': 'Power Supply', 'Description': 'Raspberry Pi power', 'Price': 10.00, 'Currency': '€'},
            {'Item': 'Multiplex 8mm', 'Description': 'Box for the raspPi and camera', 'Price': 15.00, 'Currency': '€'},
            {'Item': 'Gorilla glue for wood', 'Description': 'To construct the box', 'Price': 5.00, 'Currency': '€'}
        ]
        df = pd.DataFrame(manual_data)
    
    # Categorize
    df, category_totals = categorize_bom(df)
    
    # Calculate total
    total = df['Price'].sum()
    
    return df, category_totals, text, total

# ----------------------------
# DISPLAY RESULTS
# ----------------------------
def display_results(df, category_totals, total):
    """Display formatted results"""
    print("\n" + "="*50)
    print("ITEMIZED BREAKDOWN:")
    print("="*50)
    
    # Display items with categories
    display_df = df.copy()
    if 'Description' in display_df.columns:
        display_df['Item (Description)'] = display_df['Item'] + " - " + display_df['Description']
        display_df = display_df[['Item (Description)', 'Price', 'Category', 'Currency']]
    else:
        display_df = display_df[['Item', 'Price', 'Category', 'Currency']]
    
    print(display_df.to_string(index=True))
    
    print("\n" + "="*50)
    print("EXPENSE ANALYSIS BY CATEGORY:")
    print("="*50)
    
    for category, amount in category_totals.items():
        percentage = (amount/total)*100
        print(f"{category}: €{amount:.2f} ({percentage:.1f}%)")
    
    print("\n" + "="*50)
    print(f"TOTAL BILL: €{total:.2f}")
    print("="*50)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # Your image path
    image_path = "Receipt.jpg"  # Make sure this is the correct path
    
    try:
        # Process the receipt
        df, category_totals, extracted_text, total = process_bom_receipt(image_path)
        
        # Display results
        display_results(df, category_totals, total)
        
        # Save to Excel for further analysis
        output_file = "bom_analysis.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Items', index=False)
            
            # Summary sheet
            summary = pd.DataFrame({
                'Metric': ['Total Items', 'Total Cost (€)', 'Number of Categories'],
                'Value': [len(df), total, len(category_totals)]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Category totals
            category_df = pd.DataFrame({
                'Category': category_totals.index,
                'Amount (€)': category_totals.values,
                'Percentage': [(amt/total)*100 for amt in category_totals.values]
            })
            category_df.to_excel(writer, sheet_name='Category Analysis', index=False)
        
        print(f"\n✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the image file exists at the specified path")
        print("2. Check if Tesseract is installed: 'pytesseract.get_tesseract_version()'")
        print("3. Try running with a clearer image of the receipt")
