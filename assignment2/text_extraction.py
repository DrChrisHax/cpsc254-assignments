#!/usr/bin/env python3
# text_extraction.py
"""
Receipt OCR extraction using Tesseract (pytesseract) with improved heuristics.

Usage:
    python text_extraction.py --zip receipts.zip --out shopping_summary.csv
    python text_extraction.py --folder ./receipts --conf 50 --max-item 150

Only uses pytesseract/Tesseract; tune --conf and --max-item for your data.
"""

import argparse
import os
import re
import csv
import zipfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pytesseract

# Tesseract executable path override (if needed)
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# ---------- constants & regex ----------
PRICE_RE = re.compile(r'\$?\s*\d{1,3}(?:[.,]\d{1,2})?$')
ANY_NUM_RE = re.compile(r'[\d\.,]+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')
LONG_INT_ONLY_RE = re.compile(r'^\d{6,}$')
SURVEY_FOOTERS = ['survey', 'visit', 'www', 'http', 'feedback', 'give us', 'thank you', 'receipt id', 'ref', 'approval', 'terminal', 'tc#', 'trans id', 'merchant', 'card', 'visa', 'mastercard', 'amex', 'transaction']
TOTAL_KEYWORDS = ['total', 'amount due', 'grand total', 'subtotal', 'balance', 'amount']
COMMON_STORES = ['walmart', 'trader', 'trader joe', 'whole foods', 'safeway', 'target', 'cvs', 'walgreens', 'aldi', 'sprouts']

# ---------- filesystem helpers ----------
def extract_zip_to_folder(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    return out_dir

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    files = [str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in exts]
    return sorted(files)

# ---------- image preprocessing ----------
def preprocess(img_bgr, target_w=1200):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w > 0 and w != target_w:
        scale = target_w / w
        new_h = int(h * scale)
        gray = cv2.resize(gray, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

# ---------- OCR wrapper (defined and used consistently) ----------
def run_tesseract_image_to_data(img_gray, psm=6, oem=3):
    pil_img = Image.fromarray(img_gray)
    cfg = f'--oem {oem} --psm {psm}'
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=cfg)
    return data

# ---------- utilities ----------
def normalize_price_token(tok):
    if tok is None:
        return None
    tok = str(tok).strip()
    tok = tok.replace('$', '').replace('€', '').replace('£', '').strip()
    tok = tok.replace('O', '0').replace('o', '0')
    tok = tok.replace('l', '1').replace('I', '1')
    if ',' in tok and '.' not in tok:
        parts = tok.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            tok = tok.replace(',', '.')
        else:
            tok = tok.replace(',', '')
    elif ',' in tok and '.' in tok:
        tok = tok.replace(',', '')
    tok = re.sub(r'[^0-9.]', '', tok)
    if not tok or tok == '.':
        return None
    try:
        return round(float(tok), 2)
    except ValueError:
        return None
    
def token_looks_like_price(token):
    token = str(token).strip()
    if not token:
        return False
    if '$' in token:
        return True
    if re.search(r'\d[.,]\d', token):
        return True
    cleaned = re.sub(r'[^0-9]', '', token)
    if cleaned and 1 <= len(cleaned) <= 5:
        return True
    return False

def is_footer_line(line_text):
    lower = line_text.lower()
    for kw in SURVEY_FOOTERS:
        if kw in lower:
            return True
    if PHONE_RE.search(line_text):
        return True
    return False

def pick_store_name(top_lines):
    ignore_words = {'receipt', 'invoice', 'order', 'date', 'time', 'cashier', 'register',
                    'store', 'phone', 'tel', 'fax', 'address'}
    for line in top_lines:
        lower = line.lower().strip()
        if not lower:
            continue
        for store in COMMON_STORES:
            if store in lower:
                return line.strip()
    for line in top_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in ignore_words:
            continue
        letters = sum(1 for c in stripped if c.isalpha())
        digits = sum(1 for c in stripped if c.isdigit())
        if letters > 2 and letters >= digits:
            return stripped
    for line in top_lines:
        if line.strip():
            return line.strip()
    return 'Unknown Store'

# ---------- token -> lines clustering ----------
def cluster_tokens_into_lines(data):
    tokens = []
    n = len(data.get('text', []))
    for i in range(n):
        text = str(data['text'][i]).strip()
        if not text:
            continue
        try:
            left = int(data['left'][i])
            top = int(data['top'][i])
            height = int(data['height'][i])
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            continue
        tokens.append({'text': text, 'left': left, 'top': top, 'height': height, 'conf': conf})

    if not tokens:
        return []

    tokens.sort(key=lambda t: (t['top'], t['left']))
    lines = []
    current_line_tokens = [tokens[0]]
    current_top = tokens[0]['top']
    current_height = tokens[0]['height']

    for tok in tokens[1:]:
        threshold = max(current_height * 0.6, 8)
        if abs(tok['top'] - current_top) <= threshold:
            current_line_tokens.append(tok)
        else:
            current_line_tokens.sort(key=lambda t: t['left'])
            line_text = ' '.join(t['text'] for t in current_line_tokens)
            lines.append({'tokens': current_line_tokens, 'line_text': line_text})
            current_line_tokens = [tok]
            current_top = tok['top']
            current_height = tok['height']

    if current_line_tokens:
        current_line_tokens.sort(key=lambda t: t['left'])
        line_text = ' '.join(t['text'] for t in current_line_tokens)
        lines.append({'tokens': current_line_tokens, 'line_text': line_text})

    return lines

# ---------- line parsing ----------
def parse_line_for_item(line_obj, conf_threshold, max_item_price):
    tokens = line_obj['tokens']
    line_text = line_obj['line_text']
    lower_text = line_text.lower()
    is_total = any(kw in lower_text for kw in TOTAL_KEYWORDS)

    if is_footer_line(line_text):
        return None, None, False

    qty_match = re.search(r'(\d+)\s*[@xX]\s*(\$?\s*[\d.,]+)', line_text)
    if qty_match:
        qty = int(qty_match.group(1))
        unit_price = normalize_price_token(qty_match.group(2))
        if unit_price is not None:
            total_price = round(qty * unit_price, 2)
            item_name = line_text[:qty_match.start()].strip()
            if not item_name:
                item_name = line_text
            item_name = re.sub(r'[\s]+', ' ', item_name).strip()
            return item_name if item_name else None, total_price, is_total

    candidates = []
    for i, tok in enumerate(tokens):
        if token_looks_like_price(tok['text']):
            cleaned = re.sub(r'[^0-9]', '', tok['text'])
            if LONG_INT_ONLY_RE.match(cleaned):
                continue
            if PHONE_RE.match(tok['text']):
                continue
            price = normalize_price_token(tok['text'])
            if price is not None and price > 0:
                candidates.append((i, tok, price))

    if not candidates:
        return None, None, is_total

    best = None
    for i, tok, price in reversed(candidates):
        if tok['conf'] >= conf_threshold:
            best = (i, tok, price)
            break
    if best is None:
        best = candidates[-1]

    idx, price_tok, price = best

    if not is_total and price > max_item_price:
        return None, None, is_total

    name_parts = []
    for t in tokens:
        if t['left'] < price_tok['left']:
            name_parts.append(t['text'])
        elif t is price_tok:
            break

    item_name = ' '.join(name_parts).strip()
    item_name = re.sub(r'^[\s\-:.*#]+', '', item_name)
    item_name = re.sub(r'[\s\-:.*#]+$', '', item_name)
    item_name = re.sub(r'\s+', ' ', item_name).strip()

    if not item_name:
        item_name = line_text.replace(price_tok['text'], '').strip()
        item_name = re.sub(r'\s+', ' ', item_name).strip()

    if not item_name:
        return None, None, is_total

    return item_name, price, is_total

# ---------- process a single file ----------
def process_image_file(path, args):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    processed = preprocess(img_bgr, target_w=args.target_width)
    data = run_tesseract_image_to_data(processed, psm=args.psm)
    lines = cluster_tokens_into_lines(data)

    if not lines:
        return 'Unknown Store', [], 0.0, None

    top_lines = [l['line_text'] for l in lines[:5]]
    store = pick_store_name(top_lines)

    items = []
    printed_total = None

    for line_obj in lines:
        item_name, price, is_total = parse_line_for_item(line_obj, args.conf, args.max_item)
        if item_name is None or price is None:
            continue
        if is_total:
            printed_total = price
        else:
            items.append((item_name, price))

    computed_total = round(sum(p for _, p in items), 2)
    return store, items, computed_total, printed_total

# ---------- CSV I/O ----------
def write_csv_rows(rows, out_path):
    """
    Write rows (list of dict with keys 'store','item','amount') to CSV file with header.
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['store','item','amount'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------- CLI entry ----------
def main():
    parser = argparse.ArgumentParser(description='Receipt OCR extractor (Tesseract only) with robust heuristics.')
    parser.add_argument('--zip', type=str, default=None, help='zip file with images')
    parser.add_argument('--folder', type=str, default='../datasets/receipts', help='folder with images')
    parser.add_argument('--out', type=str, default='shopping_summary.csv', help='output CSV')
    parser.add_argument('--psm', type=int, default=6, help='tesseract PSM (page segmentation mode)')
    parser.add_argument('--conf', type=float, default=45.0, help='preferred min token confidence (0-100)')
    parser.add_argument('--max-item', type=float, default=200.0, help='max plausible per-item price (non-total)')
    parser.add_argument('--target-width', type=int, default=1200, help='preprocess resize width for OCR (higher = slower)')
    args = parser.parse_args()

    if not args.zip and not args.folder:
        raise SystemExit('Provide --zip or --folder containing receipt images.')

    img_folder = None
    if args.zip:
        img_folder = 'extracted_receipts'
        extract_zip_to_folder(args.zip, img_folder)
    else:
        img_folder = args.folder

    files = list_images(img_folder)
    if not files:
        raise SystemExit('No images found in folder.')

    rows = []
    for f in sorted(files):
        try:
            store, items, computed_total, printed_total = process_image_file(f, args)
        except Exception as e:
            print(f"[ERROR] processing {f}: {e}")
            continue
        for name, price in items:
            rows.append({'store': store, 'item': name, 'amount': f'${price:.2f}'})
        rows.append({'store': store, 'item': 'Total', 'amount': f'${computed_total:.2f}'})
        if printed_total is not None:
            rows.append({'store': store, 'item': 'Receipt_Total', 'amount': f'${printed_total:.2f}'})

    write_csv_rows(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print('If you still see huge nonsense values: increase --conf (e.g. 55) and lower --max-item (e.g. 100).')

if __name__ == '__main__':
    main()
