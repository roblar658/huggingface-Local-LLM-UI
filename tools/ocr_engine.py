import os
import re
import sys
import time

# Begrens OpenBLAS og OMP til 1 tråd for å forhindre allokeringsfeil under Windows minnepress
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Tving UTF-8 standard output på Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, ImageGrab, ImageOps


def grab_clipboard_to_files(dest_img_path, dest_thumb_raw):
    try:
        clip = None
        for _ in range(4):
            try:
                clip = ImageGrab.grabclipboard()
                if clip is not None:
                    break
            except Exception:
                pass
            time.sleep(0.05)

        img = None
        if isinstance(clip, Image.Image):
            img = clip
        elif isinstance(clip, list):
            for f in clip:
                if (
                    isinstance(f, str)
                    and os.path.exists(f)
                    and f.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
                    )
                ):
                    img = Image.open(f)
                    break

        if img is None:
            try:
                import subprocess

                ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
if ([System.Windows.Forms.Clipboard]::ContainsImage()) {{
    $img = [System.Windows.Forms.Clipboard]::GetImage()
    $img.Save("{dest_img_path}", [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output "OK_PS"
}}
'''
                ps_out = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        ps_script,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if "OK_PS" in ps_out.stdout and os.path.exists(dest_img_path):
                    img = Image.open(dest_img_path)
            except Exception:
                pass

        if img is None:
            sys.stderr.write("NO_IMAGE: Ingen bilde funnet i utklippstavlen.\n")
            sys.exit(2)

        if img.mode in ('RGBA', 'LA') or (
            img.mode == 'P' and 'transparency' in img.info
        ):
            alpha = img.convert('RGBA')
            bg = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
            bg.paste(alpha, mask=alpha)
            rgb_img = bg.convert('RGB')
        else:
            rgb_img = img.convert('RGB')

        rgb_img.save(dest_img_path, "PNG")

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


def _normalize_homoglyph_num(tok: str) -> str:
    out = []
    for ch in tok:
        if ch in ('i', 'I', 'l', '|'):
            out.append('1')
        elif ch in ('o', 'O'):
            out.append('0')
        else:
            out.append(ch)
    return "".join(out)


def fix_ocr_numeric_traps(text: str) -> str:
    """
    Korrigerer typiske OCR-feiltolkninger og font-feller (f.eks. Georgia old-style tall
    i programmeringskonkurranser som Kattis/Codeforces der '100' har x-høyde som 'ioo'/'1oo'/'i00',
    eller '1' tolkes som 'l'/'i'/'I').
    """
    # 1. Utbedre avkuttet "l be between" -> "will be between"
    text = re.sub(r'(?i)\bl\s+be\s+between\b', 'will be between', text)

    # 2. Tierpotenser og tall med old-style '1'/'0'-homoglyfer: ioo, i00, 1oo, l00, I00, iooo -> 100, 1000
    def _replace_pow10(m):
        return '1' + ('0' * len(m.group(2)))

    text = re.sub(r'\b([1iIl])([0oO]{2,10})\b', _replace_pow10, text)
    text = re.sub(r'\b([iIl])(0{1,10})\b', _replace_pow10, text)

    # 3. Sifre etterfulgt av 'o'/'O' (f.eks. 1o -> 10, 5o -> 50, 10o -> 100)
    text = re.sub(
        r'\b(\d+)([oO]+)\b', lambda m: m.group(1) + ('0' * len(m.group(2))), text
    )

    # 4. 'o'/'O' etterfulgt av sifre (f.eks. o5 -> 05)
    text = re.sub(
        r'\b([oO]+)(\d+)\b', lambda m: ('0' * len(m.group(1))) + m.group(2), text
    )

    # 5. Intervall-uttrykk: "between X and Y" / "from X to Y" / "mellom X og Y"
    def _fix_range(m):
        w1, a, w2, b = m.group(1), m.group(2), m.group(3), m.group(4)
        # Hvis minst ett av leddene ser ut som et tall eller homoglyf-tall, normaliser begge
        a_norm = _normalize_homoglyph_num(a)
        b_norm = _normalize_homoglyph_num(b)
        if a_norm.isdigit() and b_norm.isdigit():
            return f"{w1} {a_norm} {w2} {b_norm}"
        return m.group(0)

    text = re.sub(
        r'(?i)\b(between|from|mellom|fra)\s+([0-9iIl|oO]+)\s+(and|to|og|til)\s+([0-9iIl|oO]+)\b',
        _fix_range,
        text,
    )

    # 6. Ulikheter: "l <= C" eller "A <= ioo"
    text = re.sub(
        r'\b([1iIl][0oO]*)\b(?=\s*[≤≥<>])',
        lambda m: _normalize_homoglyph_num(m.group(1)),
        text,
    )
    text = re.sub(
        r'(?<=[≤≥<>])\s*\b([1iIl][0oO]+|l)\b',
        lambda m: " " + _normalize_homoglyph_num(m.group(1)),
        text,
    )

    return text


def extract_boxes_from_rapidocr(result, scale=1.0, border=0.0):
    boxes = []
    if not result:
        return boxes
    for item in result:
        bbox, text, score = item
        text = text.strip()
        if not text:
            continue
        ys = [(p[1] / scale) - border for p in bbox]
        xs = [(p[0] / scale) - border for p in bbox]
        top = min(ys)
        bottom = max(ys)
        left = min(xs)
        right = max(xs)
        height = max(1.0, bottom - top)
        boxes.append(
            {
                'top': top,
                'bottom': bottom,
                'left': left,
                'right': right,
                'height': height,
                'text': text,
            }
        )
    return boxes


def run_ocr_on_file(image_path):
    if not os.path.exists(image_path):
        sys.stderr.write(f"FILE_NOT_FOUND: {image_path}\n")
        sys.exit(2)

    try:
        img = Image.open(image_path)
    except Exception as e:
        sys.stderr.write(f"CANNOT_OPEN: {e}\n")
        sys.exit(2)

    if img.mode in ('RGBA', 'LA') or (
        img.mode == 'P' and 'transparency' in img.info
    ):
        alpha = img.convert('RGBA')
        bg = Image.new('RGBA', alpha.size, (255, 255, 255, 255))
        bg.paste(alpha, mask=alpha)
        img = bg.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    img_np = np.array(img)
    # Deaktiver 180-graders vinkelklassifiserer (use_cls=False) slik at horisontale tekstlinjer
    # med kursive variabler (A, B, C) aldri snus opp-ned ved en feil!
    result_1x, _ = engine(img_np, use_cls=False)
    boxes = extract_boxes_from_rapidocr(result_1x, scale=1.0, border=0.0)

    # Sjekk om det finnes mistenkelige vertikale hull (f.eks. der en linje med skråstilte
    # matematiske symboler som 'B' ble hoppet over av DBNet ved 1x oppløsning).
    # Ved behov kjører vi et 2x oppskalert pass med use_cls=False og fletter inn manglende linjer!
    need_2x_pass = not boxes
    if len(boxes) >= 2:
        sorted_b = sorted(boxes, key=lambda b: b['top'])
        avg_h = sum(b['height'] for b in sorted_b) / len(sorted_b)
        for i in range(len(sorted_b) - 1):
            gap = sorted_b[i + 1]['top'] - sorted_b[i]['bottom']
            if gap > avg_h * 0.85:
                need_2x_pass = True
                break

    if need_2x_pass:
        border_px = 24
        img_pad = ImageOps.expand(img, border=border_px, fill='white')
        img_2x = img_pad.resize(
            (img_pad.width * 2, img_pad.height * 2), Image.Resampling.BICUBIC
        )
        result_2x, _ = engine(
            np.array(img_2x),
            use_cls=False,
            box_thresh=0.3,
            unclip_ratio=1.8,
        )
        boxes_2x = extract_boxes_from_rapidocr(
            result_2x, scale=2.0, border=float(border_px)
        )
        # Flett inn bokser fra 2x-passet som ikke allerede overlapper med en boks i 1x-passet
        for b2 in boxes_2x:
            overlaps_existing = False
            for b1 in boxes:
                ov_top = max(b1['top'], b2['top'])
                ov_bot = min(b1['bottom'], b2['bottom'])
                ov_h = max(0.0, ov_bot - ov_top)
                min_h = min(b1['height'], b2['height'])
                if min_h > 0 and (ov_h / min_h) > 0.35:
                    overlaps_existing = True
                    break
            if not overlaps_existing:
                boxes.append(b2)

    if not boxes:
        return

    # =========================================================================
    # REKONSTRUKSJON AV VISUELLE LINJER, BOKSER OG MELLOMROM MELLOM TALL
    # =========================================================================
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

    rows.sort(key=lambda r: sum(b['top'] for b in r) / len(r))

    avg_height = sum(b['height'] for b in boxes) / len(boxes)
    formatted_rows = []
    last_row_bottom = None

    for r in rows:
        r.sort(key=lambda b: b['left'])

        r_top = min(b['top'] for b in r)
        r_bottom = max(b['bottom'] for b in r)

        if last_row_bottom is not None:
            v_gap = r_top - last_row_bottom
            if v_gap > avg_height * 0.7:
                formatted_rows.append("")

        last_row_bottom = r_bottom

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

                char_w = max(6.0, last_item['height'] * 0.45)
                spaces_count = max(1, int(round(h_gap / char_w)))

                if spaces_count >= 4:
                    spacing = "    "
                elif spaces_count > 1:
                    spacing = " " * spaces_count
                else:
                    spacing = " "

                if prev_ends_digit and cur_starts_digit:
                    if not spacing:
                        spacing = " "

                row_str += spacing + t

            last_item = b

        if row_str:
            formatted_rows.append(row_str)

    output = "\n".join(formatted_rows).strip()
    output = fix_ocr_numeric_traps(output)
    print(output)


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == '--grab-clipboard':
        grab_clipboard_to_files(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        img_path = (
            sys.argv[2]
            if sys.argv[1] == '--ocr' and len(sys.argv) >= 3
            else sys.argv[1]
        )
        run_ocr_on_file(img_path)
    else:
        sys.stderr.write(
            "Bruk: ocr_engine.py --grab-clipboard <dest_img> <dest_thumb> | --ocr <image_path>\n"
        )
        sys.exit(1)
