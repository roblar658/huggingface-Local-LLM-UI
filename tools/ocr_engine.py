import os
import sys
import numpy as np
from PIL import Image, ImageGrab

# Tving UTF-8 standard output på Windows
sys.stdout.reconfigure(encoding='utf-8')

def grab_clipboard_to_files(dest_img_path, dest_thumb_raw):
    try:
        clip = ImageGrab.grabclipboard()
        img = None
        if isinstance(clip, Image.Image):
            img = clip
        elif isinstance(clip, list):
            # Filkopiering i Windows Explorer (f.eks. Ctrl+C på bildefil)
            for f in clip:
                if os.path.exists(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
                    img = Image.open(f)
                    break

        if img is None:
            sys.stderr.write("NO_IMAGE: Ingen bilde funnet i utklippstavlen.\n")
            sys.exit(2)

        # Lagre fullversjon som PNG
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            alpha = img.convert('RGBA')
            bg = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
            bg.paste(alpha, mask=alpha)
            rgb_img = bg.convert('RGB')
        else:
            rgb_img = img.convert('RGB')

        rgb_img.save(dest_img_path, "PNG")

        # Lag thumbnail for egui UI
        thumb = img.convert('RGBA')
        thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
        w, h = thumb.size

        with open(dest_thumb_raw, "wb") as f:
            header = f"{w},{h}\n".encode('ascii')
            f.write(header)
            f.write(thumb.tobytes())

        print(f"OK {w} {h}")
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)

def run_ocr_on_file(image_path):
    if not os.path.exists(image_path):
        sys.stderr.write(f"FILE_NOT_FOUND: {image_path}\n")
        sys.exit(2)

    try:
        img = Image.open(image_path)
    except Exception as e:
        sys.stderr.write(f"CANNOT_OPEN: {e}\n")
        sys.exit(2)

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA')
        bg = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
        bg.paste(alpha, mask=alpha)
        img = bg.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    img_np = np.array(img)
    result, _ = engine(img_np)

    if not result:
        # Ingen tekst gjenkjent
        return

    # Pakk ut polygon-koordinater og tekst
    boxes = []
    for item in result:
        bbox, text, score = item
        text = text.strip()
        if not text:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        top = min(ys)
        bottom = max(ys)
        left = min(xs)
        right = max(xs)
        height = max(1.0, bottom - top)
        boxes.append({
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'height': height,
            'text': text
        })

    if not boxes:
        return

    # =========================================================================
    # REKONSTRUKSJON AV VISUELLE LINJER, BOKSER OG MELLOMROM MELLOM TALL
    # =========================================================================
    # Grupper tekstbokser inn i samme horisontale linje dersom de overlapper vertikalt
    # med mer enn 40 % av høyden. Dette sikrer at nettside-bokser, kolonner og tall
    # på samme rad holdes sammen på samme linje!
    rows = []
    for b in sorted(boxes, key=lambda x: (x['top'], x['left'])):
        best_row = None
        best_overlap = 0.0

        for r in rows:
            r_top = min(item['top'] for item in r)
            r_bottom = max(item['bottom'] for item in r)
            r_height = r_bottom - r_top

            overlap_top = max(b['top'], r_top)
            overlap_bottom = min(b['bottom'], r_bottom)
            overlap = max(0.0, overlap_bottom - overlap_top)

            min_h = min(b['height'], r_height)
            if min_h > 0 and (overlap / min_h) > 0.4:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_row = r

        if best_row is not None:
            best_row.append(b)
        else:
            rows.append([b])

    # Sorter radene vertikalt ovenfra og ned
    rows.sort(key=lambda r: sum(b['top'] for b in r) / len(r))

    avg_height = sum(b['height'] for b in boxes) / len(boxes)
    formatted_rows = []
    last_row_bottom = None

    for r in rows:
        # Sorter elementer i samme rad fra venstre mot høyre
        r.sort(key=lambda b: b['left'])

        r_top = min(b['top'] for b in r)
        r_bottom = max(b['bottom'] for b in r)

        # Visuell avstand til forrige rad: dersom avstanden er markant (avsnitt / boks-skille)
        if last_row_bottom is not None:
            v_gap = r_top - last_row_bottom
            if v_gap > avg_height * 0.7:
                formatted_rows.append("")

        last_row_bottom = r_bottom

        # Formater raden og bevar mellomrom mellom ord og tall
        row_str = ""
        last_item = None

        for b in r:
            t = b['text'].strip()
            if not t:
                continue

            if last_item is None:
                row_str = t
            else:
                h_gap = max(0.0, b['left'] - last_item['right'])
                prev_text = last_item['text'].strip()

                prev_ends_digit = prev_text[-1].isdigit() if prev_text else False
                cur_starts_digit = t[0].isdigit() if t else False

                # Beregn naturlig mellomrom basert på pikselavstand
                char_w = max(6.0, last_item['height'] * 0.45)
                spaces_count = max(1, int(round(h_gap / char_w)))

                if spaces_count >= 4:
                    spacing = "    "
                elif spaces_count > 1:
                    spacing = " " * spaces_count
                else:
                    spacing = " "

                # Sørg for at to tall aldri limes sammen uten mellomrom
                if prev_ends_digit and cur_starts_digit:
                    if not spacing:
                        spacing = " "

                row_str += spacing + t

            last_item = b

        if row_str:
            formatted_rows.append(row_str)

    output = "\n".join(formatted_rows).strip()
    print(output)

if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == '--grab-clipboard':
        grab_clipboard_to_files(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        img_path = sys.argv[2] if sys.argv[1] == '--ocr' and len(sys.argv) >= 3 else sys.argv[1]
        run_ocr_on_file(img_path)
    else:
        sys.stderr.write("Bruk: ocr_engine.py --grab-clipboard <dest_img> <dest_thumb> | --ocr <image_path>\n")
        sys.exit(1)
