import os
import sys
import numpy as np
from PIL import Image, ImageGrab

# Tving UTF-8 standard output på Windows
sys.stdout.reconfigure(encoding='utf-8')

def get_image():
    # 1. Sjekk argument (filbane)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        try:
            return Image.open(sys.argv[1])
        except Exception as e:
            sys.stderr.write(f"FEIL_FIL: Kunne ikke åpne bildefilen: {e}\n")
            sys.exit(2)

    # 2. Sjekk utklippstavle
    try:
        clip = ImageGrab.grabclipboard()
        if isinstance(clip, Image.Image):
            return clip
        elif isinstance(clip, list):
            # Filkopiering i Windows Explorer
            for f in clip:
                if os.path.exists(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
                    return Image.open(f)
    except Exception:
        pass

    sys.stderr.write("INGEN_BILDE: Fant ikke bilde i utklippstavlen. Ta et skjermbilde (f.eks. med Win+Shift+S) først.\n")
    sys.exit(2)

def run_ocr(img):
    # Konverter RGBA til RGB med hvit bakgrunn for optimal kontrast
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
        sys.stderr.write("TOM_TEKST: Bildet ble analysert, men ingen tekst ble gjenkjent.\n")
        sys.exit(4)

    # Bevar visuelle linjeskift og avsnitt basert på koordinater (bounding boxes)
    lines_with_boxes = []
    for item in result:
        bbox, text, score = item
        text = text.strip()
        if not text:
            continue
        # bbox er [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        top = min(ys)
        bottom = max(ys)
        left = min(xs)
        height = max(1.0, bottom - top)
        lines_with_boxes.append({
            'top': top,
            'bottom': bottom,
            'left': left,
            'height': height,
            'text': text
        })

    if not lines_with_boxes:
        sys.stderr.write("TOM_TEKST: Ingen gyldig tekst funnet.\n")
        sys.exit(4)

    # Sorter linjene ovenfra og ned
    lines_with_boxes.sort(key=lambda item: (item['top'], item['left']))

    # Beregn gjennomsnittlig linjehøyde for å skille vanlige linjeskift fra avsnitt
    avg_height = sum(l['height'] for l in lines_with_boxes) / len(lines_with_boxes)

    formatted_lines = []
    last_bottom = None

    for item in lines_with_boxes:
        cur_top = item['top']
        cur_bottom = item['bottom']
        text = item['text']

        if last_bottom is not None:
            gap = cur_top - last_bottom
            # Avsnittshopp dersom vertikalt gap er større enn 0.6x gjennomsnittlig linjehøyde
            if gap > avg_height * 0.6:
                formatted_lines.append("")

        formatted_lines.append(text)
        last_bottom = cur_bottom

    output_text = "\n".join(formatted_lines).strip()
    print(output_text)

if __name__ == '__main__':
    img = get_image()
    run_ocr(img)
