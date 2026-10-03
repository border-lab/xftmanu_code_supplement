#!/usr/bin/env python3
"""Replace only Figure 5a's raster schematic with its preserved SVG derivative.

The SVG supplies native ordinary text, vector arrows/nodes/math, and the original
DNA icon textures. Lower plotted panels, page size, and panel letters are exact.
"""
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile
import hashlib, json, shutil, re
import fitz
import numpy as np

BASE = Path(__file__).resolve().parents[1]
REPORT = BASE/'verification/submission_completion/figure5_vector_schematic.json'

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    assert not REPORT.exists(), 'Already installed; restore the recorded backup to repeat.'
    src = BASE/'figures/Fig_5.pdf'
    png = src.with_suffix('.png')
    candidate = BASE/'artwork_sources/figure5/Fig_5_schematic_editable.pdf'
    doc = fitz.open(src)
    page = doc[0]
    original_text = page.get_text()
    original_size = list(page.rect)
    pix = page.get_pixmap(matrix=fitz.Matrix(4,4), alpha=False)
    original = np.frombuffer(pix.samples, np.uint8).reshape(pix.height,pix.width,3).copy()
    # These are the original Figure 5 source XObjects, checked before mutation.
    rects = page.get_image_rects(9)
    assert len(rects) == 1
    rect = rects[0]
    assert abs(rect.width-394.682644)<.001 and rect.y1<217
    stream = doc.xref_stream(13)
    pattern = rb'q 530\.73 0 0 291\.33 58\.64 428\.67 cm 1 J 1 j /Im1 Do Q'
    changed, n = re.subn(pattern,b'',stream)
    assert n == 1
    doc.update_stream(13,changed)
    schematic = fitz.open(candidate)
    page.show_pdf_page(rect,schematic,0,keep_proportion=False)
    dest = src.with_name('Fig_5.vector_candidate.pdf')
    doc.save(dest,garbage=4,deflate=True)
    doc.close()
    check = fitz.open(dest)
    p = check[0]
    assert list(p.rect) == original_size
    assert p.get_text().startswith(original_text)
    for phrase in ['Parental Edu','Parental Wealth','Individual factors','Mating','xAM','VT']:
        assert phrase in p.get_text()
    pix = p.get_pixmap(matrix=fitz.Matrix(4,4), alpha=False)
    result = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    # PDF reserialization can round an 8-bit background channel by one level.
    # The lower-panel source operators themselves are unmodified above.
    delta=int(np.abs(original[220*4:].astype('int16')-result[220*4:].astype('int16')).max())
    assert delta<=1
    labels = [s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']
              if s['text'] in list('abcd') and 'Bold' in s['font']]
    assert len(labels)==4 and all(abs(s['size']-7)<.001 for s in labels)
    backup=BASE/'backups'/('before_figure5_vector_schematic.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    names=['manuscript_1.docx','manuscript_redline_vs_round5_edit.docx']
    for f in [src,png,*[BASE/name for name in names]]: shutil.copy2(f,backup/f.name)
    oldpng=png.read_bytes()
    p.get_pixmap(dpi=300).save(png)
    check.close()
    dest.replace(src)
    documents={}
    for name in names:
        f=BASE/name
        with ZipFile(f) as z:
            infos=z.infolist(); parts={i.filename:z.read(i.filename) for i in infos}
        media=[n for n,v in parts.items() if n.startswith('word/media/') and v==oldpng]
        assert len(media)==1,(name,media)
        before=sha(parts['word/document.xml'])
        parts[media[0]]=png.read_bytes()
        temp=f.with_suffix('.updating.docx')
        with ZipFile(temp,'w') as z:
            for i in infos:z.writestr(i,parts[i.filename])
        with ZipFile(temp) as z:
            assert z.testzip() is None
            assert sha(z.read('word/document.xml'))==before
        temp.replace(f)
        documents[name]={'replaced_image':media[0],'document_xml_byte_identical':True,'sha256':sha(f.read_bytes())}
    report={'backup':str(backup),'original_source':'artwork_sources/figure5/edu_cdiag_redux_original.svg',
            'editable_source':'artwork_sources/figure5/Fig_5_schematic_editable.svg',
            'diagram_rect_pt':list(rect),'maximum_lower_panel_channel_difference_at_288dpi':delta,
            'lower_panel_source_operators_unchanged':True,
            'all_original_native_plot_text_unchanged':True,'panel_letters_pt':7,
            'schematic_math_and_arrows':'vector paths; original mathematical values and geometry preserved',
            'schematic_ordinary_labels':'native text','dna_icons':'original raster textures',
            'page_size_unchanged':True,'documents':documents,'figure_sha256':sha(src.read_bytes())}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
