from flask import Flask, render_template, request, jsonify, send_file, abort, Response
from flask_cors import CORS
import cv2
import requests
import base64
import re
import numpy as np
import pytesseract
import easyocr
import os
from datetime import datetime
import pathlib
import traceback
import logging
from io import BytesIO
from PIL import Image
import concurrent.futures
import time
from difflib import SequenceMatcher
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "m0mRV8ub1J3STxQ8uS1O")
OCR_API_KEY = os.getenv("OCR_API_KEY", "K81884915988957")
ROBOFLOW_MODEL_URL = f"https://detect.roboflow.com/number-plate-rkkxy/3?api_key={ROBOFLOW_API_KEY}"
# Alternative model URL for letter detection
ROBOFLOW_LETTER_MODEL_URL = f"https://detect.roboflow.com/numberplate-sezdu/2?api_key=sEAOdISEt7ONTnPSSjxq"

# Directory setup with absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories with proper permissions
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set Flask config
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max for videos
COMPRESSION_RATIO = 0.5  # 50% compression (3MB → 1.5MB)
ROBOFLOW_MAX_SIZE = 2 * 1024 * 1024  # 2MB max for Roboflow API
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Cache for OCR readers to avoid reinitialization
ocr_readers_cache = {}

# Global camera object for live feed
camera = None

# === INDIAN NUMBER PLATE REFERENCE DATA ===
class IndianPlateReference:
    """Complete reference for Indian number plate patterns and codes"""
    
    def __init__(self):
        # Valid Indian state/UT codes (first 2 letters)
        self.state_codes = {
            'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
            'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh',
            'DL': 'Delhi', 'DN': 'Dadra and Nagar Haveli', 'GA': 'Goa', 'GJ': 'Gujarat',
            'HP': 'Himachal Pradesh', 'HR': 'Haryana', 'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir',
            'KA': 'Karnataka', 'KL': 'Kerala', 'LA': 'Ladakh', 'LD': 'Lakshadweep',
            'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur', 'MP': 'Madhya Pradesh',
            'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
            'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TG': 'Telangana',
            'TN': 'Tamil Nadu', 'TR': 'Tripura', 'TS': 'Telangana', 'UK': 'Uttarakhand',
            'UP': 'Uttar Pradesh', 'WB': 'West Bengal'
        }
        
        # Valid district numbers (01-99)
        self.valid_district_numbers = [f"{i:02d}" for i in range(1, 100)]
        
        # Valid single digit district codes (1-9)
        self.valid_single_digits = [str(i) for i in range(1, 10)]
        
        # Valid series letters (middle letters - commonly used combinations)
        self.common_series_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM',
            'AN', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
            'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BJ', 'BK', 'BL', 'BM',
            'BN', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ',
            'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CJ', 'CK', 'CL', 'CM',
            'CN', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ'
        ]
        
        # Valid 4-digit numbers (0001-9999)
        self.valid_numbers = [f"{i:04d}" for i in range(1, 10000)]
        
        # Common OCR error patterns
        self.ocr_corrections = {
            '0': ['O', 'Q', 'D'],
            '1': ['I', 'L', 'l', '|'],
            '2': ['Z'],
            '5': ['S'],
            '6': ['G'],
            '8': ['B'],
            'O': ['0', 'Q', 'D'],
            'I': ['1', 'l', '|'],
            'S': ['5'],
            'G': ['6'],
            'B': ['8'],
            'Z': ['2']
        }
        
        # Indian plate patterns with regex
        self.plate_patterns = [
            # Standard new format: KA01AB1234
            {
                'pattern': r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$',
                'format': 'New Standard Format',
                'description': 'State(2) + District(2) + Series(1-2) + Number(4)'
            },
            # Old format with single district digit: KA1AB1234
            {
                'pattern': r'^([A-Z]{2})(\d{1})([A-Z]{1,2})(\d{4})$',
                'format': 'Old Single Digit District',
                'description': 'State(2) + District(1) + Series(1-2) + Number(4)'
            },
            # Very old format: KAR1234
            {
                'pattern': r'^([A-Z]{3})(\d{4})$',
                'format': 'Very Old Format',
                'description': 'State/City(3) + Number(4)'
            },
            # Commercial vehicle format: KA01C1234
            {
                'pattern': r'^([A-Z]{2})(\d{2})([A-Z]{1})(\d{4})$',
                'format': 'Commercial Format',
                'description': 'State(2) + District(2) + Category(1) + Number(4)'
            }
        ]

plate_ref = IndianPlateReference()

def get_easyocr_reader():
    """Get or create EasyOCR reader with caching"""
    if 'easyocr' not in ocr_readers_cache:
        try:
            ocr_readers_cache['easyocr'] = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            ocr_readers_cache['easyocr'] = None
    return ocr_readers_cache['easyocr']

def compress_to_percentage_fast(image_file, target_ratio=COMPRESSION_RATIO):
    """
    Fast compression with optimized settings
    """
    try:
        # Get original size
        image_file.seek(0, os.SEEK_END)
        original_size = image_file.tell()
        image_file.seek(0)
        
        target_size = int(original_size * target_ratio)
        logger.info(f"🗜️ Fast compressing: {original_size} bytes → {target_size} bytes")
        
        # Open image with PIL
        img = Image.open(image_file)
        original_format = img.format if img.format else 'JPEG'
        
        # Convert to RGB if necessary
        if original_format == 'JPEG' and img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Use faster compression with optimized settings
        buffer = BytesIO()
        save_format = 'JPEG' if original_format in ['JPEG', 'JPG'] else original_format
        
        # Single pass compression with optimized quality
        quality = 75  # Balanced quality/speed
        if save_format == 'JPEG':
            img.save(buffer, format=save_format, quality=quality, optimize=True)
        else:
            img.save(buffer, format=save_format, optimize=True)
        
        compressed_size = buffer.getbuffer().nbytes
        buffer.seek(0)
        
        logger.info(f"✅ Fast compression completed: {compressed_size} bytes")
        return buffer, compressed_size
        
    except Exception as e:
        logger.error(f"❌ Fast compression error: {e}")
        # Fallback to original
        image_file.seek(0)
        return BytesIO(image_file.read()), original_size

def compress_image_for_roboflow_fast(image_path, max_size_bytes=ROBOFLOW_MAX_SIZE):
    """
    Fast compression for Roboflow API
    """
    try:
        logger.info(f"🎯 Fast compressing for Roboflow API")
        
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Single pass compression with optimized settings
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]  # Balanced quality/speed
        success, encoded_img = cv2.imencode('.jpg', img, encode_param)
        
        if not success:
            return None
        
        img_size = len(encoded_img.tobytes())
        logger.info(f"🔧 Roboflow fast compression: {img_size} bytes")
        
        if img_size <= max_size_bytes:
            # Save compressed version
            temp_path = image_path.replace('.', f'_roboflow_temp.')
            temp_path = temp_path.rsplit('.', 1)[0] + '.jpg'
            
            with open(temp_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            
            logger.info(f"✅ Roboflow fast compression successful")
            return temp_path
        
        # If still too large, resize
        height, width = img.shape[:2]
        scale = 0.7  # Moderate resize
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        if new_width >= 200 and new_height >= 200:
            img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            success, encoded_img = cv2.imencode('.jpg', img_resized, encode_param)
            
            if success:
                img_size = len(encoded_img.tobytes())
                if img_size <= max_size_bytes:
                    temp_path = image_path.replace('.', f'_roboflow_temp.')
                    temp_path = temp_path.rsplit('.', 1)[0] + '.jpg'
                    
                    with open(temp_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    
                    logger.info(f"✅ Roboflow resize successful: {img_size} bytes")
                    return temp_path
        
        logger.error("❌ Could not compress image enough for Roboflow API")
        return None
        
    except Exception as e:
        logger.error(f"❌ Roboflow fast compression error: {e}")
        return None

class SmartPlateValidator:
    """Advanced plate validation and correction using reference data"""
    
    def __init__(self):
        self.plate_ref = plate_ref
    
    def similarity_score(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a, b).ratio()
    
    def correct_ocr_errors(self, text):
        """Apply common OCR error corrections"""
        corrected = text
        for correct_char, error_chars in self.plate_ref.ocr_corrections.items():
            for error_char in error_chars:
                corrected = corrected.replace(error_char, correct_char)
        return corrected
    
    def find_best_state_match(self, detected_state):
        """Find the best matching state code"""
        if detected_state in self.plate_ref.state_codes:
            return detected_state, 1.0
        
        best_match = detected_state
        best_score = 0.0
        
        for valid_state in self.plate_ref.state_codes.keys():
            score = self.similarity_score(detected_state, valid_state)
            if score > best_score and score > 0.5:  # At least 50% similarity
                best_match = valid_state
                best_score = score
        
        return best_match, best_score
    
    def find_best_district_match(self, detected_district):
        """Find the best matching district number"""
        # Try exact match first
        if detected_district in self.plate_ref.valid_district_numbers:
            return detected_district, 1.0
        
        if len(detected_district) == 1 and detected_district in self.plate_ref.valid_single_digits:
            return detected_district, 1.0
        
        # Try to correct common OCR errors
        corrected = self.correct_ocr_errors(detected_district)
        if corrected in self.plate_ref.valid_district_numbers:
            return corrected, 0.9
        
        # Find closest match
        best_match = detected_district
        best_score = 0.0
        
        valid_districts = self.plate_ref.valid_district_numbers + self.plate_ref.valid_single_digits
        for valid_district in valid_districts:
            score = self.similarity_score(detected_district, valid_district)
            if score > best_score and score > 0.6:
                best_match = valid_district
                best_score = score
        
        return best_match, best_score
    
    def find_best_series_match(self, detected_series):
        """Find the best matching series letters"""
        # Try exact match first
        if detected_series in self.plate_ref.common_series_letters:
            return detected_series, 1.0
        
        # Try to correct OCR errors
        corrected = self.correct_ocr_errors(detected_series)
        if corrected in self.plate_ref.common_series_letters:
            return corrected, 0.9
        
        # Find closest match
        best_match = detected_series
        best_score = 0.0
        
        for valid_series in self.plate_ref.common_series_letters:
            if len(detected_series) == len(valid_series):  # Same length
                score = self.similarity_score(detected_series, valid_series)
                if score > best_score and score > 0.5:
                    best_match = valid_series
                    best_score = score
        
        return best_match, best_score
    
    def validate_and_correct_number(self, detected_number):
        """Validate and correct 4-digit number"""
        # Ensure 4 digits
        if len(detected_number) < 4:
            detected_number = detected_number.zfill(4)
        elif len(detected_number) > 4:
            detected_number = detected_number[:4]
        
        # Apply OCR corrections
        corrected = self.correct_ocr_errors(detected_number)
        
        # Ensure all are digits
        corrected = ''.join([c if c.isdigit() else '0' for c in corrected])
        
        # Validate range
        try:
            num = int(corrected)
            if 1 <= num <= 9999:
                return f"{num:04d}", 1.0
            else:
                return "0001", 0.3  # Default fallback
        except:
            return "0001", 0.1
    
    def extract_and_validate_plate(self, raw_text):
        """Extract and validate plate components using reference data"""
        logger.info(f"🔍 Validating plate text: '{raw_text}'")
        
        if not raw_text or len(raw_text) < 7:
            return None
        
        best_result = None
        best_confidence = 0.0
        
        # Try each pattern
        for pattern_info in self.plate_ref.plate_patterns:
            pattern = pattern_info['pattern']
            format_name = pattern_info['format']
            
            match = re.match(pattern, raw_text)
            if match:
                logger.info(f"✅ Pattern matched: {format_name}")
                result = self.validate_pattern_match(match, pattern_info, raw_text)
                if result and result['confidence'] > best_confidence:
                    best_result = result
                    best_confidence = result['confidence']
        
        # If no exact pattern match, try fuzzy matching
        if not best_result or best_confidence < 0.7:
            logger.info("🔄 Trying fuzzy pattern matching...")
            fuzzy_result = self.fuzzy_pattern_match(raw_text)
            if fuzzy_result and fuzzy_result['confidence'] > best_confidence:
                best_result = fuzzy_result
                best_confidence = fuzzy_result['confidence']
        
        return best_result
    
    def validate_pattern_match(self, match, pattern_info, raw_text):
        """Validate a regex pattern match against reference data"""
        groups = match.groups()
        format_name = pattern_info['format']
        
        if format_name == 'New Standard Format' and len(groups) == 4:
            # KA01AB1234 format
            state, district, series, number = groups
            
            # Validate each component
            corrected_state, state_conf = self.find_best_state_match(state)
            corrected_district, district_conf = self.find_best_district_match(district)
            corrected_series, series_conf = self.find_best_series_match(series)
            corrected_number, number_conf = self.validate_and_correct_number(number)
            
            # Calculate overall confidence
            confidence = (state_conf + district_conf + series_conf + number_conf) / 4
            
            corrected_plate = f"{corrected_state}{corrected_district}{corrected_series}{corrected_number}"
            
            return {
                'original_text': raw_text,
                'corrected_plate': corrected_plate,
                'format': format_name,
                'confidence': confidence,
                'components': {
                    'state': {'original': state, 'corrected': corrected_state, 'confidence': state_conf, 
                             'name': self.plate_ref.state_codes.get(corrected_state, 'Unknown')},
                    'district': {'original': district, 'corrected': corrected_district, 'confidence': district_conf},
                    'series': {'original': series, 'corrected': corrected_series, 'confidence': series_conf},
                    'number': {'original': number, 'corrected': corrected_number, 'confidence': number_conf}
                },
                'is_valid': confidence > 0.8
            }
        
        elif format_name == 'Old Single Digit District' and len(groups) == 4:
            # KA1AB1234 format
            state, district, series, number = groups
            
            corrected_state, state_conf = self.find_best_state_match(state)
            corrected_district, district_conf = self.find_best_district_match(district)
            corrected_series, series_conf = self.find_best_series_match(series)
            corrected_number, number_conf = self.validate_and_correct_number(number)
            
            confidence = (state_conf + district_conf + series_conf + number_conf) / 4
            corrected_plate = f"{corrected_state}{corrected_district}{corrected_series}{corrected_number}"
            
            return {
                'original_text': raw_text,
                'corrected_plate': corrected_plate,
                'format': format_name,
                'confidence': confidence,
                'components': {
                    'state': {'original': state, 'corrected': corrected_state, 'confidence': state_conf, 
                             'name': self.plate_ref.state_codes.get(corrected_state, 'Unknown')},
                    'district': {'original': district, 'corrected': corrected_district, 'confidence': district_conf},
                    'series': {'original': series, 'corrected': corrected_series, 'confidence': series_conf},
                    'number': {'original': number, 'corrected': corrected_number, 'confidence': number_conf}
                },
                'is_valid': confidence > 0.8
            }
        
        elif format_name == 'Very Old Format' and len(groups) == 2:
            # KAR1234 format
            state_code, number = groups
            
            # For old format, state code is 3 letters
            corrected_number, number_conf = self.validate_and_correct_number(number)
            
            # State code validation is less strict for old format
            state_conf = 0.8 if len(state_code) == 3 and state_code.isalpha() else 0.5
            
            confidence = (state_conf + number_conf) / 2
            corrected_plate = f"{state_code}{corrected_number}"
            
            return {
                'original_text': raw_text,
                'corrected_plate': corrected_plate,
                'format': format_name,
                'confidence': confidence,
                'components': {
                    'state_code': {'original': state_code, 'corrected': state_code, 'confidence': state_conf},
                    'number': {'original': number, 'corrected': corrected_number, 'confidence': number_conf}
                },
                'is_valid': confidence > 0.7
            }
        
        return None
    
    def fuzzy_pattern_match(self, text):
        """Try to extract plate components using fuzzy matching"""
        logger.info(f"🔄 Fuzzy matching for: '{text}'")
        
        # Apply OCR corrections first
        corrected_text = self.correct_ocr_errors(text)
        
        # Try to identify components based on position and character type
        if len(corrected_text) >= 10:  # Minimum for new format
            # Assume new format: XX##XX####
            state = corrected_text[:2]
            district = corrected_text[2:4]
            series = corrected_text[4:6] if corrected_text[6:7].isdigit() else corrected_text[4:5]
            number_start = 6 if len(series) == 2 else 5
            number = corrected_text[number_start:number_start+4]
            
            # Validate components
            corrected_state, state_conf = self.find_best_state_match(state)
            corrected_district, district_conf = self.find_best_district_match(district)
            corrected_series, series_conf = self.find_best_series_match(series)
            corrected_number, number_conf = self.validate_and_correct_number(number)
            
            confidence = (state_conf + district_conf + series_conf + number_conf) / 4
            
            if confidence > 0.5:
                corrected_plate = f"{corrected_state}{corrected_district}{corrected_series}{corrected_number}"
                
                return {
                    'original_text': text,
                    'corrected_plate': corrected_plate,
                    'format': 'Fuzzy Match - New Format',
                    'confidence': confidence * 0.8,  # Reduce confidence for fuzzy match
                    'components': {
                        'state': {'original': state, 'corrected': corrected_state, 'confidence': state_conf,
                                 'name': self.plate_ref.state_codes.get(corrected_state, 'Unknown')},
                        'district': {'original': district, 'corrected': corrected_district, 'confidence': district_conf},
                        'series': {'original': series, 'corrected': corrected_series, 'confidence': series_conf},
                        'number': {'original': number, 'corrected': corrected_number, 'confidence': number_conf}
                    },
                    'is_valid': confidence > 0.6
                }
        
        elif len(corrected_text) >= 7:  # Try old format
            # Assume old format: XXX####
            state_code = corrected_text[:3]
            number = corrected_text[3:7]
            
            corrected_number, number_conf = self.validate_and_correct_number(number)
            state_conf = 0.7 if len(state_code) == 3 and state_code.isalpha() else 0.4
            
            confidence = (state_conf + number_conf) / 2
            
            if confidence > 0.5:
                corrected_plate = f"{state_code}{corrected_number}"
                
                return {
                    'original_text': text,
                    'corrected_plate': corrected_plate,
                    'format': 'Fuzzy Match - Old Format',
                    'confidence': confidence * 0.8,
                    'components': {
                        'state_code': {'original': state_code, 'corrected': state_code, 'confidence': state_conf},
                        'number': {'original': number, 'corrected': corrected_number, 'confidence': number_conf}
                    },
                    'is_valid': confidence > 0.5
                }
        
        return None

class EnhancedNumberPlateDetector:
    def __init__(self):
        # Lazy initialization of OCR readers
        self.easyocr_reader = None
        self.tesseract_available = self.check_tesseract()
        self.plate_validator = SmartPlateValidator()
    
    def check_tesseract(self):
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract is available")
            return True
        except Exception:
            logger.warning("Tesseract is not available, skipping Tesseract OCR")
            return False
    
    def get_easyocr_reader(self):
        """Lazy initialization of EasyOCR"""
        if self.easyocr_reader is None:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        return self.easyocr_reader
    
    def clean_text(self, text):
        """Enhanced text cleaning for number plates"""
        if not text:
            return ""
        # Enhanced cleaning for Indian number plates
        text = str(text).upper().strip()
        # Remove common OCR errors and keep only alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
    
    def validate_indian_plate_format(self, text):
        """Validate if text matches Indian number plate patterns"""
        if not text or len(text) < 8:
            return False
        
        # Common Indian plate patterns
        patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$',  # Standard format: KA01AB1234
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',      # New format: KA01AB1234
            r'^[A-Z]{3}[0-9]{4}$',                       # Old format: KAR1234
            r'^[A-Z]{2}[0-9]{1,2}[A-Z][0-9]{4}$'        # Variant: KA01A1234
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        return False
    
    # === OCR Functions from your script ===
    def ocr_space_image(self, img):
        """Enhanced OCR.Space implementation"""
        try:
            # Encode with higher quality for better text recognition
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success, encoded_img = cv2.imencode(".jpg", img, encode_param)
            if not success:
                return ""
            
            payload = {
                "apikey": OCR_API_KEY, 
                "language": "eng", 
                "OCREngine": "2",
                "isTable": "false",
                "scale": "true",
                "detectOrientation": "false"
            }
            files = {"file": ("plate.jpg", encoded_img.tobytes(), "image/jpeg")}
            
            response = requests.post(
                "https://api.ocr.space/parse/image", 
                files=files, 
                data=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if "ParsedResults" in result and len(result["ParsedResults"]) > 0:
                    text = result["ParsedResults"][0]["ParsedText"].strip()
                    return self.clean_text(text)
        except Exception as e:
            logger.error(f"OCR.Space error: {e}")
        return ""
    
    def ocr_tesseract(self, img):
        """Enhanced Tesseract OCR with multiple configurations"""
        if not self.tesseract_available:
            return ""
        try:
            # Multiple Tesseract configurations for number plates
            configs = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # single word mode
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single text line
                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Uniform text block
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Raw line
            ]
            
            best_text = ""
            best_score = 0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    cleaned_text = self.clean_text(text)
                    score = self.score_text(cleaned_text)
                    
                    if score > best_score:
                        best_text = cleaned_text
                        best_score = score
                        
                except Exception:
                    continue
            
            return best_text
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return ""
    
    def ocr_easyocr(self, img):
        """Enhanced EasyOCR implementation"""
        try:
            reader = self.get_easyocr_reader()
            if reader is None:
                return ""
            
            # EasyOCR with optimized parameters for license plates
            results = reader.readtext(
                img, 
                detail=1, 
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            # Combine all detected text with confidence filtering
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence detections
                    texts.append(text)
            
            combined_text = "".join(texts)
            return self.clean_text(combined_text)
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
        return ""
    
    # === Preprocessing from your script (enhanced) ===
    def preprocess_variations_enhanced(self, img):
        """Enhanced preprocessing variations combining both approaches"""
        variations = []
        try:
            # === Original preprocessing from your script ===
            # Resize for better OCR (2x upscaling)
            resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized.copy()
            
            # Enhance contrast
            contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            # Apply Gaussian blur for noise reduction
            blur = cv2.GaussianBlur(contrast, (3, 3), 0)
            
            # Adaptive thresholding
            thresh_adapt = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Sharpening kernel
            kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
            sharpened = cv2.filter2D(thresh_adapt, -1, kernel)
            
            # Morphological closing
            morph_kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, morph_kernel)
            
            # Inverted version
            inverted = cv2.bitwise_not(morph_close)
            
            # Add your original variations
            variations.extend([
                ("Resized Gray", gray),
                ("Contrast Enhanced", contrast),
                ("Gaussian Blur", blur),
                ("Adaptive Threshold", thresh_adapt),
                ("Sharpened", sharpened),
                ("Morph Close", morph_close),
                ("Inverted Morph", inverted)
            ])
            
            # === Additional advanced preprocessing ===
            # CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_enhanced = clahe.apply(gray)
            variations.append(("CLAHE Enhanced", clahe_enhanced))
            
            # Otsu's thresholding
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variations.append(("Otsu Threshold", otsu))
            
            # Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            variations.append(("Bilateral Filtered", bilateral))
            
            # Edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
            variations.append(("Edge Enhanced", edges_dilated))
            
            logger.info(f"Generated {len(variations)} preprocessing variations")
            
        except Exception as e:
            logger.error(f"Enhanced preprocessing error: {e}")
            # Fallback to original
            if len(img.shape) == 3:
                variations.append(("Original Gray", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            else:
                variations.append(("Original", img))
        
        return variations
    
    def run_all_ocr_engines(self, img):
        """Run all OCR engines on an image (from your script)"""
        engines_to_use = []
        if self.tesseract_available:
            engines_to_use.append(("Tesseract", self.ocr_tesseract))
        engines_to_use.extend([
            ("OCR.Space", self.ocr_space_image),
            ("EasyOCR", self.ocr_easyocr)
        ])
        
        results = {}
        # Use thread pool for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines_to_use)) as executor:
            # Submit all OCR tasks
            future_to_engine = {
                executor.submit(engine_func, img): engine_name 
                for engine_name, engine_func in engines_to_use
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    results[engine] = future.result()
                except Exception as e:
                    logger.error(f"{engine} error: {e}")
                    results[engine] = ""
        
        return results
    
    def score_text(self, text):
        """Enhanced scoring system combining both approaches"""
        if not text:
            return 0
        
        # Base score from your script
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        score = len(text) + (2 if has_letters and has_numbers else 0)
        
        # Enhanced scoring
        # Bonus for valid Indian plate format
        if self.validate_indian_plate_format(text):
            score += 10
        
        # Penalty for too short or too long
        if len(text) < 6:
            score -= 3
        elif len(text) > 12:
            score -= 2
        
        # Bonus for optimal length (8-10 characters)
        if 8 <= len(text) <= 10:
            score += 3
        
        return score
    
    def detect_with_dual_models(self, image_path):
        """Try both Roboflow models for better detection"""
        models_to_try = [
            ("Primary Model", ROBOFLOW_MODEL_URL),
            ("Letter Detection Model", ROBOFLOW_LETTER_MODEL_URL)
        ]
        
        for model_name, model_url in models_to_try:
            try:
                logger.info(f"🔍 Trying {model_name}...")
                
                # Convert to base64
                with open(image_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                
                response = requests.post(
                    model_url,
                    data=img_base64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=25
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "predictions" in result and len(result["predictions"]) > 0:
                        logger.info(f"✅ {model_name} detected plate successfully")
                        return result, model_name
                    else:
                        logger.info(f"⚠️ {model_name} - no predictions found")
                else:
                    logger.warning(f"⚠️ {model_name} HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ {model_name} error: {e}")
                continue
        
        return None, "No Model"
    
    def process_image(self, image_path):
        try:
            start_time = time.time()
            
            if not os.path.exists(image_path):
                return {"error": "Image file not found"}
            
            # Fast image loading
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to load image - invalid format"}
            
            image_copy = image.copy()
            logger.info(f"Processing image: {image_path}")
            
            # Check if image needs compression for Roboflow API
            roboflow_image_path = image_path
            file_size = os.path.getsize(image_path)
            
            # Base64 encoding increases size by ~33%, so check against smaller limit
            if file_size > ROBOFLOW_MAX_SIZE * 0.75:
                logger.info(f"🔄 Image too large for Roboflow ({file_size} bytes), compressing...")
                compressed_path = compress_image_for_roboflow_fast(image_path, ROBOFLOW_MAX_SIZE * 0.75)
                
                if compressed_path is None:
                    return {"error": "Image too large and could not be compressed for processing"}
                
                roboflow_image_path = compressed_path
                logger.info(f"✅ Using compressed image for Roboflow: {roboflow_image_path}")
            
            # === Step 1: Detect number plate with dual model approach ===
            try:
                result, model_used = self.detect_with_dual_models(roboflow_image_path)
                
                if result is None:
                    return {"error": "No number plate detected by any model"}
                
                pred = result["predictions"][0]
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                confidence = pred.get("confidence", 0)
                
                logger.info(f"📡 Detection successful with {model_used} (confidence: {confidence:.3f})")
                
            except Exception as e:
                return {"error": f"Detection error: {str(e)}"}
            finally:
                # Clean up temporary compressed file
                if roboflow_image_path != image_path and os.path.exists(roboflow_image_path):
                    try:
                        os.remove(roboflow_image_path)
                    except Exception as e:
                        logger.error(f"Failed to clean up temporary file: {e}")
            
            # === Step 2: Crop detected plate safely within image bounds ===
            x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
            x2, y2 = min(image.shape[1], int(x + w / 2)), min(image.shape[0], int(y + h / 2))
            
            if x2 <= x1 or y2 <= y1:
                return {"error": "Invalid bounding box dimensions"}
            
            cropped_plate = image[y1:y2, x1:x2]
            
            # Save cropped plate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cropped_filename = f"cropped_{timestamp}.jpg"
            cropped_path = os.path.join(RESULTS_DIR, cropped_filename)
            cv2.imwrite(cropped_path, cropped_plate)
            
            # === Step 3: Run OCR on original crop with all engines ===
            logger.info("🔄 Running OCR on original cropped plate...")
            combined_results = {}
            base_results = self.run_all_ocr_engines(cropped_plate)
            
            for engine, text in base_results.items():
                combined_results[f"Original - {engine}"] = text
                logger.info(f"📝 Original - {engine}: '{text}' (score: {self.score_text(text)})")
            
            # === Step 4: Check if any result is good enough ===
            best_base_engine = max(base_results, key=lambda k: self.score_text(base_results[k]))
            best_score = self.score_text(base_results[best_base_engine])
            best_text = base_results[best_base_engine]
            best_method = f"Original - {best_base_engine}"
            
            if best_score > 3:
                logger.info(f"✅ Good result from original image: '{best_text}' (score: {best_score})")
            else:
                # === Step 5: Try preprocessing variations if original OCR not good ===
                logger.info("⚠️ Raw plate unreadable, trying enhanced preprocessing...")
                variations = self.preprocess_variations_enhanced(cropped_plate)
                
                for name, var_img in variations:
                    try:
                        var_results = self.run_all_ocr_engines(var_img)
                        for engine, text in var_results.items():
                            key = f"{name} - {engine}"
                            combined_results[key] = text
                            score = self.score_text(text)
                            
                            if score > best_score:
                                best_text = text
                                best_method = key
                                best_score = score
                                
                            logger.info(f"📝 {key}: '{text}' (score: {score})")
                            
                    except Exception as e:
                        logger.error(f"Error processing variation {name}: {e}")
                        continue
            
            # === STEP 6: SMART VALIDATION AND CORRECTION ===
            logger.info("🧠 Running smart plate validation and correction...")
            validation_result = None
            if best_text:
                validation_result = self.plate_validator.extract_and_validate_plate(best_text)
            
            # Determine final plate text
            if validation_result and validation_result['is_valid']:
                final_plate = validation_result['corrected_plate']
                final_confidence = validation_result['confidence']
                validation_applied = True
                logger.info(f"✅ Smart validation successful: '{final_plate}' (confidence: {final_confidence:.3f})")
            else:
                final_plate = best_text
                final_confidence = best_score / 15.0  # Normalize to 0-1 range
                validation_applied = False
                logger.info(f"⚠️ Using raw OCR result: '{final_plate}'")
            
            # Create annotated image using original image
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add text annotation
            display_text = final_plate if final_plate else "No Text Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(display_text, font, font_scale, thickness)
            text_x = x1
            text_y = max(y1 - 10, text_size[1] + 10)
            
            # Background rectangle for text
            cv2.rectangle(image_copy, (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)
            cv2.putText(image_copy, display_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            
            # Save annotated image
            annotated_filename = f"annotated_{timestamp}.jpg"
            annotated_path = os.path.join(RESULTS_DIR, annotated_filename)
            cv2.imwrite(annotated_path, image_copy)
            
            processing_time = time.time() - start_time
            
            # Check if result is valid Indian plate format
            is_valid_format = self.validate_indian_plate_format(final_plate)
            
            logger.info(f"✅ Detection completed in {processing_time:.2f}s: '{final_plate}' "
                       f"(Valid format: {is_valid_format}, Final confidence: {final_confidence:.3f})")
            
            # === Final result message (like your script) ===
            final_message = ""
            if final_plate and final_confidence > 0.3:
                if validation_applied:
                    final_message = f"✅ Smart Validated Plate Number: {final_plate} (Confidence: {final_confidence:.1%})"
                else:
                    if "Original" in best_method:
                        final_message = f"✅ Detected Plate Number (Original Image, {best_method.split(' - ')[1]}): {final_plate}"
                    else:
                        final_message = f"✅ Best OCR Result: {final_plate} (from {best_method})"
            else:
                final_message = "❌ Could not read number plate even after preprocessing and smart validation."
            
            return {
                "success": True,
                "detected_text": final_plate,
                "raw_ocr_text": best_text,
                "detection_method": best_method,
                "model_used": model_used,
                "confidence": float(confidence),
                "text_score": best_score,
                "final_confidence": final_confidence,
                "valid_format": is_valid_format,
                "validation_applied": validation_applied,
                "validation_result": validation_result,
                "final_message": final_message,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "cropped_path": cropped_path,
                "annotated_path": annotated_path,
                "all_ocr_results": combined_results,
                "total_variations_tested": len([v for v in combined_results.keys() if "Original" not in v]) + len(base_results),
                "timestamp": timestamp,
                "processing_time": f"{processing_time:.2f}s"
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Processing error: {str(e)}"}
    
    def process_video_frame(self, frame):
        """Process a single video frame for number plate detection"""
        try:
            # Save frame temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_frame_path = os.path.join(UPLOAD_DIR, f"temp_frame_{timestamp}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Process the frame as an image
            result = self.process_image(temp_frame_path)
            
            # Clean up temp file
            try:
                os.remove(temp_frame_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            return {"error": f"Frame processing error: {str(e)}"}

# Initialize detector
detector = EnhancedNumberPlateDetector()

def get_camera():
    """Initialize camera for live feed"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Use default camera
        if not camera.isOpened():
            logger.error("Could not open camera")
            return None
    return camera

def generate_camera_frames():
    """Generate camera frames for live feed"""
    global camera
    camera = get_camera()
    if camera is None:
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_plate():
    """API endpoint for image detection with 50% compression"""
    start_time = time.time()
    logger.info("=== SMART PLATE VALIDATION API DETECT REQUEST STARTED ===")
    
    try:
        if 'image' not in request.files:
            error_msg = "No image file provided"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        file = request.files['image']
        logger.info(f"Processing file: {file.filename}")
        
        if file.filename == '' or file.filename is None:
            error_msg = "No file selected"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Validate file type - Image formats only for this endpoint
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: '{file_ext}'")
        
        if file_ext not in allowed_extensions:
            error_msg = f"Invalid file type '{file_ext}'. Supported: JPG, PNG, BMP, TIFF"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Check original file size
        file.seek(0, os.SEEK_END)
        original_size = file.tell()
        file.seek(0)
        
        logger.info(f"📁 Original file size: {original_size} bytes ({original_size / (1024*1024):.2f} MB)")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"upload_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # **ALWAYS COMPRESS TO 50%** - This is the main feature
        logger.info(f"🗜️ Applying 50% compression to all images...")
        
        try:
            compressed_buffer, compressed_size = compress_to_percentage_fast(file, COMPRESSION_RATIO)
            
            if compressed_buffer is None:
                error_msg = "Failed to compress image. Please try a different file."
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 400
            
            # Save compressed image
            with open(file_path, 'wb') as f:
                f.write(compressed_buffer.read())
            
            # Calculate actual compression ratio
            actual_ratio = compressed_size / original_size
            compression_percentage = (1 - actual_ratio) * 100
            
            logger.info(f"✅ Compression successful: {original_size} → {compressed_size} bytes "
                       f"({actual_ratio*100:.1f}% of original, {compression_percentage:.1f}% reduction)")
            
        except Exception as e:
            error_msg = f"Compression failed: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Verify file was saved
        if not os.path.exists(file_path):
            error_msg = "File was not saved properly"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        
        # Verify final file size
        saved_size = os.path.getsize(file_path)
        logger.info(f"💾 Final saved file size: {saved_size} bytes")
        
        # Process the image using thread pool for better responsiveness
        logger.info("🔄 Starting smart plate validation processing...")
        future = thread_pool.submit(detector.process_image, file_path)
        result = future.result()
        
        if "error" in result:
            logger.error(f"Processing error: {result['error']}")
            return jsonify(result), 400
        
        # Add compression info to result
        total_time = time.time() - start_time
        result["compression_applied"] = True
        result["original_size"] = original_size
        result["compressed_size"] = compressed_size
        result["final_size"] = saved_size
        result["compression_ratio"] = f"{actual_ratio*100:.1f}%"
        result["size_reduction"] = f"{compression_percentage:.1f}%"
        result["total_processing_time"] = f"{total_time:.2f}s"
        
        logger.info(f"=== ✅ SMART VALIDATION API COMPLETED IN {total_time:.2f}s ===")
        logger.info(f"🎯 Final Result: '{result.get('detected_text', 'No text')}' "
                   f"(Valid: {result.get('valid_format', False)}) "
                   f"(Validation Applied: {result.get('validation_applied', False)})")
        
        return jsonify(result)
    
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(f"Unexpected error: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/api/detect-video', methods=['POST'])
def detect_plate_video():
    """NEW: API endpoint for video processing"""
    start_time = time.time()
    logger.info("=== VIDEO PLATE DETECTION REQUEST STARTED ===")
    
    try:
        if 'video' not in request.files:
            error_msg = "No video file provided"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        file = request.files['video']
        logger.info(f"Processing video file: {file.filename}")
        
        if file.filename == '' or file.filename is None:
            error_msg = "No file selected"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Validate video file type
        allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.info(f"Video file extension: '{file_ext}'")
        
        if file_ext not in allowed_video_extensions:
            error_msg = f"Invalid video file type '{file_ext}'. Supported: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Check original file size
        file.seek(0, os.SEEK_END)
        original_size = file.tell()
        file.seek(0)
        
        logger.info(f"📁 Original video file size: {original_size} bytes ({original_size / (1024*1024):.2f} MB)")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"video_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save video file
        file.save(file_path)
        logger.info(f"✅ Video saved: {file_path}")
        
        # Process video frames
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 400
        
        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Process every Nth frame to avoid too many detections
        frame_skip = max(1, int(fps / 2))  # Process 2 frames per second
        frame_results = []
        processed_frames = 0
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only selected frames
            if frame_num % frame_skip == 0:
                logger.info(f"🔄 Processing frame {frame_num}/{total_frames}")
                
                # Process this frame
                result = detector.process_video_frame(frame)
                if result and "detected_text" in result and result["detected_text"]:
                    timestamp_sec = frame_num / fps if fps > 0 else 0
                    frame_results.append({
                        "frame_number": frame_num,
                        "timestamp": f"{timestamp_sec:.2f}s",
                        "detected_text": result["detected_text"],
                        "confidence": result.get("final_confidence", 0),
                        "valid_format": result.get("valid_format", False),
                        "validation_applied": result.get("validation_applied", False)
                    })
                    logger.info(f"✅ Frame {frame_num}: '{result['detected_text']}'")
                
                processed_frames += 1
            
            frame_num += 1
        
        cap.release()
        
        # Clean up video file
        try:
            os.remove(file_path)
        except:
            pass
        
        total_time = time.time() - start_time
        
        # Analyze results
        unique_plates = {}
        for result in frame_results:
            plate = result["detected_text"]
            if plate not in unique_plates:
                unique_plates[plate] = {
                    "plate": plate,
                    "count": 0,
                    "max_confidence": 0,
                    "first_seen": result["timestamp"],
                    "valid_format": result["valid_format"],
                    "validation_applied": result["validation_applied"]
                }
            unique_plates[plate]["count"] += 1
            unique_plates[plate]["max_confidence"] = max(
                unique_plates[plate]["max_confidence"], 
                result["confidence"]
            )
        
        # Sort by count and confidence
        sorted_plates = sorted(
            unique_plates.values(), 
            key=lambda x: (x["count"], x["max_confidence"]), 
            reverse=True
        )
        
        return jsonify({
            "success": True,
            "video_info": {
                "original_size": original_size,
                "total_frames": total_frames,
                "fps": fps,
                "duration": f"{duration:.2f}s",
                "processed_frames": processed_frames
            },
            "detections": {
                "total_detections": len(frame_results),
                "unique_plates": len(unique_plates),
                "frame_results": frame_results,
                "unique_plates_summary": sorted_plates
            },
            "best_detection": sorted_plates[0] if sorted_plates else None,
            "processing_time": f"{total_time:.2f}s",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Video processing error: {str(e)}"
        logger.error(f"Unexpected video error: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/api/camera-feed')
def camera_feed():
    """NEW: Live camera feed endpoint"""
    try:
        return Response(
            generate_camera_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Camera feed error: {e}")
        return jsonify({"error": "Camera not available"}), 500

@app.route('/api/camera-capture', methods=['POST'])
def camera_capture():
    """NEW: Capture and process current camera frame"""
    try:
        global camera
        camera = get_camera()
        if camera is None:
            return jsonify({"error": "Camera not available"}), 500
        
        # Capture frame
        success, frame = camera.read()
        if not success:
            return jsonify({"error": "Could not capture frame"}), 500
        
        # Process the captured frame
        result = detector.process_video_frame(frame)
        
        if "error" in result:
            return jsonify(result), 400
        
        result["source"] = "live_camera"
        result["capture_time"] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Camera capture error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/api/camera-start', methods=['POST'])
def camera_start():
    """NEW: Initialize camera"""
    try:
        camera = get_camera()
        if camera is None:
            return jsonify({"error": "Could not initialize camera"}), 500
        
        return jsonify({
            "success": True,
            "message": "Camera initialized successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Camera initialization failed: {str(e)}"}), 500

@app.route('/api/camera-stop', methods=['POST'])
def camera_stop():
    """NEW: Release camera"""
    try:
        global camera
        if camera is not None:
            camera.release()
            camera = None
        
        return jsonify({
            "success": True,
            "message": "Camera released successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Camera release failed: {str(e)}"}), 500

@app.route('/api/image/<path:filename>')
def get_image(filename):
    try:
        safe_path = pathlib.Path(filename).resolve()
        
        if not safe_path.exists() or not safe_path.is_file():
            logger.error(f"Image not found: {filename}")
            abort(404)
        
        allowed_dirs = [pathlib.Path(UPLOAD_DIR).resolve(), pathlib.Path(RESULTS_DIR).resolve()]
        if not any(str(safe_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
            logger.error(f"Access denied to file: {filename}")
            abort(403)
        
        return send_file(str(safe_path))
    
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}")
        abort(404)

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "upload_dir": UPLOAD_DIR,
        "results_dir": RESULTS_DIR,
        "max_upload_size": f"{MAX_CONTENT_LENGTH / (1024*1024):.0f}MB",
        "compression_ratio": f"{COMPRESSION_RATIO*100}%",
        "roboflow_max_size": f"{ROBOFLOW_MAX_SIZE / (1024*1024):.0f}MB",
        "enhanced_ocr": True,
        "dual_model_detection": True,
        "smart_validation": True,
        "video_processing": True,
        "live_camera": True,
        "indian_plate_patterns": len(plate_ref.plate_patterns),
        "state_codes_supported": len(plate_ref.state_codes),
        "tesseract_available": detector.tesseract_available
    })

@app.route('/api/validate-plate', methods=['POST'])
def validate_plate_text():
    """Standalone API endpoint to validate plate text"""
    try:
        data = request.get_json()
        if not data or 'plate_text' not in data:
            return jsonify({"error": "Missing plate_text parameter"}), 400
        
        plate_text = str(data['plate_text']).strip()
        if not plate_text:
            return jsonify({"error": "Empty plate_text"}), 400
        
        validator = SmartPlateValidator()
        result = validator.extract_and_validate_plate(plate_text)
        
        if result:
            return jsonify({
                "success": True,
                "validation_result": result,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Could not validate plate text",
                "original_text": plate_text,
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum upload size is {MAX_CONTENT_LENGTH / (1024*1024):.0f}MB."
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Smart Indian Number Plate Detection Server...")
    logger.info("🔧 Features: Image + Video + Live Camera + Advanced OCR + Dual Model + Smart Validation")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"📏 Max upload size: {MAX_CONTENT_LENGTH / (1024*1024):.0f}MB")
    logger.info(f"🗜️ Compression ratio: {COMPRESSION_RATIO*100}% (Images compressed to half size)")
    logger.info(f"🎯 Roboflow API limit: {ROBOFLOW_MAX_SIZE / (1024*1024):.0f}MB")
    logger.info(f"🤖 Dual Model Support: Primary + Letter Detection Models")
    logger.info(f"🧠 Smart Validation: {len(plate_ref.state_codes)} states, {len(plate_ref.plate_patterns)} patterns")
    logger.info(f"📹 Video Processing: Enabled")
    logger.info(f"📷 Live Camera Feed: Enabled")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
