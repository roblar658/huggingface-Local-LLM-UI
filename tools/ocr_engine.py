import os
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

from PIL import Image, ImageGrab

def grab_clipboard_to_files(dest_img_path, dest_thumb_raw):
    try:
        # Prøv å hente bilde fra utklippstavlen (med opptil 4 korte forsøk i tilfelle Windows låser)
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
            # Filkopiering i Windows Explorer (f.eks. Ctrl+C på bildefil)
            for f in clip:
                if isinstance(f, str) and os.path.exists(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
                    img = Image.open(f)
                    break

        # Fallback via Windows Forms Clipboard dersom PIL ikke fanget formatet direkte
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
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if "OK_PS" in ps_out.stdout and os.path.exists(dest_img_path):
                    img = Image.open(dest_img_path)
            except Exception:
                pass

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

    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    img_np = np.array(img)
    result, _ = engine(img_np)

    if not result:
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
