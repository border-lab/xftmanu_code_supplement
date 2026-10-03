#!/usr/bin/env python3
"""Apply the author's October 2 palette, panel-letter, and Methods-placement edits.

Build from a preserved local backup. Do not refresh fields or regenerate review text.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import hashlib
import io
import json
import math
import shutil
import sys

import fitz
import numpy as np
from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'new submission'
VERIFY = BASE / 'verification/figure_style_20261002'
BACKUP = ROOT / (VERIFY / 'backup_path.txt').read_text().strip()
STAGE = VERIFY / 'artwork'
FONT = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
SIZE = 7.0
MM = 72 / 25.4
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
FILES = ['manuscript_1.docx', 'manuscript_methods.docx',
         'manuscript_redline_vs_round5_edit.docx',
         'inventory_supporting_information.docx', 'supplementary_information.docx']
REPORT = {'target_panel_letter_pt': SIZE, 'font': 'Liberation Sans Bold',
          'palette': 'Orange–blue; original continuous limits retained',
          'pdf': {}, 'raster': {}, 'docx': {}, 'replacements': {}}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans']]


def panel_spans(page):
    return [s for s in spans(page) if s['text'] in list('abcdef') and 'Bold' in s['font']]


def drawing_signature(page):
    # Text redaction must never remove/repaint a plotted path or embedded diagram.
    return [{k: repr(v) for k, v in d.items() if k != 'seqno'} for d in page.get_drawings()]


def standardize_pdf(src, dest, count):
    doc = fitz.open(src)
    page = doc[0]
    labels = panel_spans(page)
    assert len(labels) == count, (src, labels)
    before_drawings = drawing_signature(page)
    before_text = Counter(''.join(s['text'] for s in spans(page)))
    for s in labels:
        page.add_redact_annot(fitz.Rect(s['bbox']), fill=None, cross_out=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    assert drawing_signature(page) == before_drawings, src
    remaining = Counter(''.join(s['text'] for s in spans(page)))
    assert remaining == before_text - Counter(''.join(s['text'] for s in labels)), src
    font = fitz.Font(fontfile=FONT)
    for s in labels:
        x, y = s['bbox'][:2]
        page.insert_text((x, max(0, y) + font.ascender * SIZE), s['text'],
                         fontsize=SIZE, fontname='PanelLabel', fontfile=FONT, color=(0, 0, 0))
    doc.subset_fonts()
    doc.save(dest, garbage=4, deflate=True)
    with fitz.open(dest) as check:
        got = panel_spans(check[0])
        assert len(got) == count and all(abs(s['size'] - SIZE) < 0.001 for s in got)
        assert Counter(''.join(s['text'] for s in spans(check[0]))) == before_text
        assert drawing_signature(check[0]) == before_drawings
        check[0].get_pixmap(dpi=300).save(dest.with_suffix('.png'))
    REPORT['pdf'][dest.name] = {
        'original_sizes_pt': [s['size'] for s in labels], 'final_size_pt': SIZE,
        'labels': [s['text'] for s in labels], 'original_boxes': [s['bbox'] for s in labels],
        'page_size_pt': list(page.rect)[2:], 'all_text_characters_preserved': True,
        'all_plotted_paths_preserved': True}
    return labels, page.rect


def placements():
    found = {}
    for name in FILES:
        with ZipFile(BACKUP / 'new submission' / name) as z:
            root = E.fromstring(z.read('word/document.xml'))
            rels = {x.get('Id'): x.get('Target') for x in E.fromstring(z.read('word/_rels/document.xml.rels'))}
            for d in root.findall('.//w:drawing', NS):
                b = d.find('.//a:blip', NS)
                extent = d.find('.//wp:extent', NS)
                if b is None or extent is None:
                    continue
                part = 'word/' + rels[b.get('{' + NS['r'] + '}embed')]
                digest = sha(z.read(part))
                found.setdefault(digest, []).append({'document': name, 'part': part,
                    'width_pt': int(extent.get('cx')) / 12700,
                    'height_pt': int(extent.get('cy')) / 12700})
    return found


def paint_raster_labels(src_bytes, dest, boxes, placement):
    """Change only isolated glyph rectangles; keep original pixels everywhere else."""
    im = Image.open(io.BytesIO(src_bytes)).convert('RGB')
    original = np.array(im)
    mask = np.zeros(original.shape[:2], dtype=bool)
    px_per_pt = im.width / placement['width_pt']
    scale = 8
    em_high = round(SIZE * px_per_pt * scale)
    font = ImageFont.truetype(FONT, em_high)
    actual_size = em_high / scale / px_per_pt
    details = []
    for ch, coords in boxes:
        x0, y0, x1, y1 = [int(v) for v in coords]
        crop = original[y0:y1, x0:x1]
        ys, xs = np.where(crop.max(axis=2) < 180)
        assert len(xs), (dest, ch, coords)
        left, top = x0 + int(xs.min()), y0 + int(ys.min())
        ImageDraw.Draw(im).rectangle((x0, y0, x1 - 1, y1 - 1), fill='white')
        mask[y0:y1, x0:x1] = True
        bbox = font.getbbox(ch)
        w = math.ceil((bbox[2] - bbox[0]) / scale) + 2
        h = math.ceil((bbox[3] - bbox[1]) / scale) + 2
        alpha = Image.new('L', (w * scale, h * scale))
        ImageDraw.Draw(alpha).text((-bbox[0], -bbox[1]), ch, font=font, fill=255)
        alpha = alpha.resize((w, h), Image.Resampling.LANCZOS)
        # Never paint over content just outside the original letter rectangle.
        target = original[top:top+h, left:left+w]
        outside = ~mask[top:top+h, left:left+w]
        assert np.all(target[outside] >= 245), (dest, ch, 'adjacent artwork')
        im.paste((0, 0, 0), (left, top), alpha)
        mask[top:top+h, left:left+w] = True
        details.append({'label': ch, 'old_box_px': list(coords), 'new_ink_origin_px': [left, top],
                        'effective_font_pt': actual_size})
    assert np.array_equal(original[~mask], np.array(im)[~mask])
    dpi = px_per_pt * 72
    im.save(dest, dpi=(dpi, dpi))
    REPORT['raster'][dest.name] = {'size_px': list(im.size), 'placement': placement,
        'labels': details, 'pixels_outside_label_regions_identical': True,
        'no_resampling_of_source_artwork': True, 'png_nominal_dpi': dpi}


def build():
    STAGE.mkdir(exist_ok=True)
    places = placements()
    replacements = {}
    for n, count in [(1, 2), (2, 4), (3, 3), (4, 4), (5, 4)]:
        src = BACKUP / f'new submission/figures/Fig_{n}.pdf'
        if n == 3:
            src = ROOT / 'figures_2_3_portrait/output/submission/Fig_2.pdf'
        dest = STAGE / f'Fig_{n}.pdf'
        standardize_pdf(src, dest, count)
        old = (BACKUP / f'new submission/figures/Fig_{n}.png').read_bytes()
        replacements[sha(old)] = dest.with_suffix('.png').name
        for where in places[sha(old)]:
            with fitz.open(dest) as pdf:
                realized = SIZE * where['width_pt'] / pdf[0].rect.width
            assert abs(realized - SIZE) < 0.01
    # Preserve the native PDF size for submission; separately size embedded letters
    # for their actual Word placement without changing those image dimensions.
    with ZipFile(BACKUP / 'new submission/inventory_supporting_information.docx') as inv:
        for n, count, media in [(3, 4, 3), (4, 2, 5), (5, 2, 4), (8, 2, 8)]:
            src = BACKUP / f'new submission/figures/Extended_Data_Fig_{n}.pdf'
            labels, rect = standardize_pdf(src, STAGE / src.name, count)
            old = inv.read(f'word/media/image{media}.png')
            im = Image.open(io.BytesIO(old))
            boxes = []
            for s in labels:
                x0, y0, x1, y1 = s['bbox']
                boxes.append((s['text'], [max(0, math.floor(x0 / rect.width * im.width) - 2),
                    max(0, math.floor(y0 / rect.height * im.height) - 2),
                    math.ceil(x1 / rect.width * im.width) + 2,
                    math.ceil(y1 / rect.height * im.height) + 2]))
            dest = STAGE / f'ED{n}_embedded.png'
            paint_raster_labels(old, dest, boxes, places[sha(old)][0])
            replacements[sha(old)] = dest.name
    si_boxes = {
        'S1': [('a',(6,11,22,29)),('b',(6,487,23,509)),('c',(6,971,22,989))],
        'S2': [('a',(6,11,22,29)),('b',(6,487,23,509)),('c',(6,971,22,989))],
        'S2_extra_image': [('a',(5,11,22,28)),('b',(6,472,23,494)),('c',(6,941,22,960))],
        'S14': [('a',(10,21,34,47)),('b',(1313,13,1337,46)),
                ('c',(11,821,33,847)),('d',(1311,813,1336,847)),
                ('e',(10,1621,34,1647)),('f',(1306,1613,1322,1647))],
        'S15': [('a',(17,31,51,68)),('b',(1226,19,1261,68))],
        'S16': [('a',(6,10,23,28)),('b',(722,5,740,28)),('c',(5,589,21,607))],
        'S17': [('a',(18,28,53,66)),('b',(20,1270,56,1321))],
    }
    for name, boxes in si_boxes.items():
        src = BACKUP / f'new submission/figures/supplementary/{name}.png'
        old = src.read_bytes()
        dest = STAGE / f'{name}.png'
        where = next(p for p in places[sha(old)] if p['document'] == 'supplementary_information.docx')
        paint_raster_labels(old, dest, boxes, where)
        replacements[sha(old)] = dest.name
        for p in places[sha(old)]:
            assert abs(p['width_pt'] - where['width_pt']) < .1
    for p in (ROOT / 'figures_2_3_portrait/output/submission').glob('Fig_2_panel_*.csv'):
        assert p.read_bytes() == (BACKUP / p.relative_to(ROOT)).read_bytes(), p
    REPORT['figure3_all_three_plotted_csvs_byte_identical'] = True
    REPORT['replacements'] = replacements
    (VERIFY / 'build_verification.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print('Built', len(REPORT['pdf']), 'PDFs and', len(REPORT['raster']), 'embedded/SI images.')


def ptext(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def install():
    report = json.loads((VERIFY / 'build_verification.json').read_text())
    assert not (VERIFY / 'installed.json').exists(), 'Already installed'
    sentences = ['UK Biobank data were accessed under application 33127.',
                 'Secondary analysis of Taiwan NHIRD data is under the review of the Taiwan National Health Research Institute.']
    for filename in FILES:
        path = BASE / filename
        assert path.read_bytes() == (BACKUP / 'new submission' / filename).read_bytes(), filename
        with ZipFile(path) as z:
            infos = z.infolist()
            parts = {i.filename: z.read(i.filename) for i in infos}
        changed = []
        for part, data in list(parts.items()):
            replacement = report['replacements'].get(sha(data)) if part.startswith('word/media/') else None
            if replacement:
                parts[part] = (STAGE / replacement).read_bytes()
                changed.append(part)
        moves = []
        if filename in FILES[:3]:
            root = E.fromstring(parts['word/document.xml'])
            body = root.find('w:body', NS)
            original_children = list(body)
            expected_children = list(original_children)
            for sentence in sentences:
                matches = [p for p in body.findall('w:p', NS) if ptext(p) == sentence]
                assert len(matches) == 1, (filename, sentence)
                p = matches[0]
                following = p.getnext()
                assert following is not None and len(ptext(following)) > 200
                # Each affected subsection contains this one substantive paragraph.
                next_heading = following.getnext()
                assert next_heading is not None and ptext(next_heading) in [
                    'Mate identification in the UK Biobank', 'Taiwan NHIRD cross-mate CCA procedure']
                idx = expected_children.index(p)
                expected_children[idx:idx+2] = [following, p]
                body.remove(p)
                following.addnext(p)
                moves.append(sentence)
            assert list(body) == expected_children
            # Nodes are moved intact: all paragraph text, styles, math and fields survive.
            assert Counter(E.tostring(p) for p in body) == Counter(E.tostring(p) for p in original_children)
            parts['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
        temp = path.with_suffix('.figure-style-tmp.docx')
        with ZipFile(temp, 'w') as z:
            for info in infos:
                z.writestr(info, parts[info.filename])
        with ZipFile(path) as before, ZipFile(temp) as after:
            assert after.testzip() is None and before.namelist() == after.namelist()
            allowed = changed + (['word/document.xml'] if moves else [])
            for name in before.namelist():
                if name not in allowed:
                    assert before.read(name) == after.read(name), (filename, name)
        temp.replace(path)
        report['docx'][filename] = {'changed_image_parts': changed, 'sentences_moved_intact': moves,
            'all_other_package_parts_byte_identical': True, 'paragraph_xml_preserved': True,
            'sha256': sha(path.read_bytes())}
    for p in STAGE.glob('*.pdf'):
        shutil.copy2(p, BASE / 'figures' / p.name)
        shutil.copy2(p.with_suffix('.png'), BASE / 'figures' / p.with_suffix('.png').name)
    for p in STAGE.glob('S*.png'):
        shutil.copy2(p, BASE / 'figures/supplementary' / p.name)
    report['status'] = 'Installed; final rendered-page review pending'
    (VERIFY / 'installed.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Installed artwork and moved the two original Methods sentences in all three relevant copies.')


if __name__ == '__main__':
    {'build': build, 'install': install}[sys.argv[1]]()
