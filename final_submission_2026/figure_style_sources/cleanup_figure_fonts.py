#!/usr/bin/env python3
"""Resize figure lettering only, preserving plotted paths and all characters.

Build review candidates first. Installation is a separate operation. Full local
fonts supply the same font families as the original subset fonts; Helvetica uses
its metric-compatible Nimbus Sans equivalent. No plot is regenerated from data.
"""
from collections import Counter
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
from zipfile import ZipFile
import hashlib, json, math, shutil, sys
import fitz
import numpy as np

BASE = Path(__file__).resolve().parents[1]
VERIFY = BASE / 'verification/figure_font_cleanup'
STAGE = VERIFY / 'artwork'
MINIMUM, MAXIMUM = 5.0, 7.0

def digest(data): return hashlib.sha256(data).hexdigest()
def chars(page):
    return Counter(c['c'] for b in page.get_text('rawdict')['blocks']
                   for l in b.get('lines', []) for s in l['spans']
                   for c in s['chars'] if not c['c'].isspace())
def paths(page):
    return [{k:repr(v) for k,v in p.items() if k != 'seqno'} for p in page.get_drawings()]

def font_path(span):
    name = span['font']
    if name.startswith('NimbusSans') or name.startswith('Helvetica'):
        style = 'BoldItalic' if ('Bold' in name and ('Italic' in name or 'Oblique' in name)) else 'Bold' if 'Bold' in name else 'Italic' if ('Italic' in name or 'Oblique' in name) else 'Regular'
        return Path('/usr/share/fonts/opentype/urw-base35') / f'NimbusSans-{style}.otf'
    if name.startswith('NotoSansMath') or name == 'Symbol':
        return Path('/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf')
    if name.startswith('NotoSans'):
        return Path('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')
    if name.startswith('DejaVu'):
        return Path('/home/rsb/miniforge3/fonts/DejaVuSans.ttf')
    if name.startswith('Liberation'):
        return Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf' if 'Bold' in name else '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf')
    raise ValueError(name)

def build_one(src, dest):
    doc = fitz.open(src); p = doc[0]
    before_paths = paths(p); before_chars = chars(p)
    expected_chars=before_chars.copy(); restored_beta=0
    original_png = p.get_pixmap(matrix=fitz.Matrix(3,3), alpha=False)
    original_pixels = np.frombuffer(original_png.samples,np.uint8).reshape(original_png.height,original_png.width,3).copy()
    spans=[]; changed_lines=[]
    rawlines=[l for b in p.get_text('rawdict')['blocks'] for l in b.get('lines',[])]
    parents=list(range(len(rawlines)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def small_math_piece(line):
        letters=''.join(c['c'] for s in line['spans'] for c in s['chars'] if not c['c'].isspace())
        return len(letters)<=3 and not any('Bold' in s['font'] for s in line['spans'])
    # PDF extraction sometimes splits a superscript/subscript onto its own line.
    # Move those together with the touching parent label, preserving notation.
    for i,a in enumerate(rawlines):
        for j in range(i):
            b=rawlines[j]
            if a['dir']==b['dir'] and (small_math_piece(a) or small_math_piece(b)) and fitz.Rect(a['bbox']).intersects(fitz.Rect(b['bbox'])):
                parents[find(i)]=find(j)
    groups={}
    for i,line in enumerate(rawlines):groups.setdefault(find(i),[]).append(line)
    for group in groups.values():
        group_spans=[s for l in group for s in l['spans'] if any(not c['c'].isspace() for c in s['chars'])]
        group_needs=any(s['size']<MINIMUM-.01 or s['size']>MAXIMUM+.01 for s in group_spans)
        largest=max([s['size'] for s in group_spans],default=0)
        factor=min(1.,MAXIMUM/largest) if group_needs and largest else 1.
        group_box=fitz.Rect(group[0]['bbox'])
        for line in group[1:]:group_box|=fitz.Rect(line['bbox'])
        anchor=(group_box.tl+group_box.br)*.5
        for line in group:
            nonblank = [s for s in line['spans'] if any(not c['c'].isspace() for c in s['chars'])]
            needs = group_needs
            box = fitz.Rect(line['bbox'])
            for span in line['spans']:
                span=deepcopy(span)
                span['direction']=line['dir']; span['selected']=needs
                span['line_factor']=factor; span['anchor']=tuple(anchor)
                span['new_size']=min(MAXIMUM,max(MINIMUM,span['size']*factor)) if needs else span['size']
                span['changed']=needs and (abs(span['new_size']-span['size'])>.001 or abs(factor-1)>.0001)
                spans.append(span)
            if needs:
                changed_lines.append({'text': ''.join(c['c'] for s in line['spans'] for c in s['chars']), 'before_sizes':[s['size'] for s in line['spans']], 'bbox':list(box), 'factor':factor})
    if not changed_lines:
        return None
    # Redaction works at character-box resolution. Include any neighbouring span
    # that touches a selected span so no overlapping hat/subscript is lost.
    while True:
        selected = [s for s in spans if s['selected']]
        newly=[]
        for s in spans:
            if not s['selected'] and any(fitz.Rect(s['bbox']).intersects(fitz.Rect(t['bbox'])) for t in selected):
                newly.append(s)
        if not newly: break
        for s in newly:s['selected']=True
    selected=[s for s in spans if s['selected']]
    remove_chars=Counter(c['c'] for s in selected for c in s['chars'] if not c['c'].isspace())
    boxes=[]
    for s in selected:
        box=fitz.Rect(s['bbox']);boxes.append(box)
        p.add_redact_annot(box,fill=None,cross_out=False)
    p.apply_redactions(images=0,graphics=0,text=0)
    assert paths(p)==before_paths, (src.name,'paths changed during text removal')
    assert chars(p)==before_chars-remove_chars,(src.name,'unexpected removed characters',chars(p)-(before_chars-remove_chars),(before_chars-remove_chars)-chars(p))
    shape=p.new_shape();fonts={};new_boxes=[];placements=[];used_glyphs={}
    for s in selected:
        fp=font_path(s)
        if fp not in fonts:
            assert fp.is_file(),fp
            fonts[fp]=(f'CleanupFont{len(fonts)}',fitz.Font(fontfile=str(fp)))
        fn,font=fonts[fp]
        dx,dy=s['direction']
        rotation = int(round(math.degrees(math.atan2(-dy,dx))))%360
        assert rotation in [0,90,180,270], (src.name,rotation)
        f=s['line_factor'];oldorigin=fitz.Point(s['origin']);anchor=fitz.Point(s['anchor'])
        origin=anchor+(oldorigin-anchor)*f
        intrafactor=s['new_size']/s['size']
        for c in s['chars']:
            ch=c['c']
            if ch.isspace(): continue
            pt=origin+(fitz.Point(c['origin'])-oldorigin)*intrafactor
            char_fn,char_font,char_fp=fn,font,fp
            # The historical PDF substituted a dot for mathematical italic beta
            # in Figure5d. The executed source's cell36 explicitly specifies
            # expression(italic(r[hat('𝛽')])). Restore that exact notation.
            source_beta=(src.name=='Fig_5.pdf' and s['font']=='Helvetica-Oblique'
                         and abs(s['origin'][0]-252.52463)<.001
                         and abs(s['origin'][1]-444.33472)<.001)
            if source_beta:
                nbpath=BASE.parent.parent/'round4/submission/notebook_regen/mFigEdu_run_executed.ipynb'
                nb=json.loads(nbpath.read_text())
                assert "ylab(expression(italic(r[hat('𝛽')])))" in ''.join(nb['cells'][36]['source'])
                math_fp=Path('/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf')
                if math_fp not in fonts:fonts[math_fp]=(f'CleanupFont{len(fonts)}',fitz.Font(fontfile=str(math_fp)))
                math_fn,math_font=fonts[math_fp]
                if ch=='.':
                    ch='𝛽';char_fn,char_font,char_fp=math_fn,math_font,math_fp
                    expected_chars['.']-=1;expected_chars['𝛽']+=1;restored_beta+=1
                elif ch=='^':
                    # Place the hat above beta in the rotated label, using the
                    # actual glyph widths; the original dot lacked beta metrics.
                    bw=math_font.text_length('𝛽',fontsize=s['new_size'])
                    hw=font.text_length('^',fontsize=s['new_size'])
                    pt=origin+fitz.Point(-.7*s['new_size'],(hw-bw)/2)
            assert char_font.has_glyph(ord(ch)),(src.name,str(char_fp),ch,ord(ch))
            used_glyphs.setdefault(char_fn,{})[char_font.has_glyph(ord(ch))]=ch
            col=s['color'];colour=((col>>16 &255)/255,(col>>8 &255)/255,(col&255)/255)
            shape.insert_text(pt,ch,fontname=char_fn,fontfile=str(char_fp),fontsize=s['new_size'],rotate=rotation,color=colour)
            placements.append({'char':ch,'point':list(pt),'font':char_font.name,'size':s['new_size']})
    shape.commit()
    # This MuPDF release writes non-BMP code points as five-digit hex values in
    # generated ToUnicode maps. Supply proper UTF-16BE, preserving mathematical
    # italic alpha/beta as their original Unicode characters and visible glyphs.
    for item in p.get_fonts(full=True):
        if item[4] not in used_glyphs: continue
        mapping=used_glyphs[item[4]]
        lines=['/CIDInit /ProcSet findresource begin','12 dict begin','begincmap',
               '/CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def',
               '/CMapName /Cleanup-UCS def','/CMapType 2 def',
               '1 begincodespacerange','<0000> <FFFF>','endcodespacerange']
        pairs=sorted(mapping.items())
        for start in range(0,len(pairs),100):
            chunk=pairs[start:start+100];lines.append(f'{len(chunk)} beginbfchar')
            lines.extend(f'<{gid:04x}> <{ch.encode("utf-16-be").hex()}>' for gid,ch in chunk)
            lines.append('endbfchar')
        lines.extend(['endcmap','CMapName currentdict /CMap defineresource pop','end','end'])
        key=doc.xref_get_key(item[0],'ToUnicode');assert key[0]=='xref'
        doc.update_stream(int(key[1].split()[0]),'\n'.join(lines).encode())
    doc.subset_fonts()
    doc.save(dest,garbage=4,deflate=True)
    doc.close()
    check=fitz.open(dest);p=check[0]
    assert paths(p)==before_paths,(src.name,'plotted paths changed')
    assert chars(p)==+expected_chars,(src.name,'character mismatch',chars(p)-expected_chars,expected_chars-chars(p))
    assert restored_beta==(1 if src.name=='Fig_5.pdf' else 0)
    allspans=[s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    outside=[(s['text'],s['size']) for s in allspans if s['size']<MINIMUM-.01 or s['size']>MAXIMUM+.01]
    assert not outside,(src.name,outside)
    labels=[s for s in allspans if s['text'] in list('abcdef') and 'Bold' in s['font']]
    assert all(abs(s['size']-7)<.001 for s in labels),(src.name,'panel letter size')
    # Differences are confined to text rectangles; this also protects raster
    # components and plot elements that are not exposed as vector paths.
    pix=p.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False)
    result=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    assert result.shape==original_pixels.shape
    mask=np.zeros(result.shape[:2],bool)
    # Include the old boxes and all final text boxes. Pixel-level checks protect
    # everything outside lettering, while path equality protects under-label art.
    for box in boxes+[fitz.Rect(s['bbox']) for s in allspans]:
        box=box+(-1,-1,1,1)
        x0,y0=max(0,int(math.floor(box.x0*3))),max(0,int(math.floor(box.y0*3)))
        x1,y1=min(mask.shape[1],int(math.ceil(box.x1*3))),min(mask.shape[0],int(math.ceil(box.y1*3)))
        mask[y0:y1,x0:x1]=True
    delta=np.abs(original_pixels.astype('int16')-result.astype('int16'))
    max_delta=int(delta[~mask].max())
    assert max_delta<=1,(src.name,'non-lettering pixels changed',max_delta)
    p.get_pixmap(dpi=300).save(dest.with_suffix('.png'))
    return {'before_sha256':digest(src.read_bytes()),'after_sha256':digest(dest.read_bytes()),
            'changed_lines':changed_lines,'redrawn_spans':len(selected),'font_size_range_pt':[min(s['size'] for s in allspans),max(s['size'] for s in allspans)],
            'all_nonspace_characters_preserved_except_source_verified_beta':True,
            'source_verified_beta_restorations':restored_beta,
            'all_plotted_paths_exactly_preserved':True,
            'maximum_pixel_difference_outside_text':max_delta,'panel_letters_7pt':True,'page_size_pt':list(p.rect),
            'font_files':[str(p) for p in fonts]}

def build():
    STAGE.mkdir(parents=True,exist_ok=True)
    report={'minimum_pt':MINIMUM,'maximum_pt':MAXIMUM,'figures':{}}
    for src in sorted((BASE/'figures').glob('*.pdf')):
        print('Checking',src.name,flush=True)
        r=build_one(src,STAGE/src.name)
        if r:report['figures'][src.name]=r
        (VERIFY/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built',len(report['figures']),'font-only candidates.',flush=True)

def install():
    assert not (VERIFY/'installed.json').exists(),'Already installed; do not repeat.'
    report=json.loads((VERIFY/'build.json').read_text())
    backup=BASE/'backups'/('before_figure_font_cleanup.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir()
    replacement={}
    for name,r in report['figures'].items():
        src=BASE/'figures'/name;candidate=STAGE/name
        assert digest(src.read_bytes())==r['before_sha256'],name
        assert digest(candidate.read_bytes())==r['after_sha256'],name
        for path in [src,src.with_suffix('.png')]:
            target=backup/'figures'/path.name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target)
        replacement[digest(src.with_suffix('.png').read_bytes())]=candidate.with_suffix('.png')
    documents={}
    for name in ['manuscript_1.docx','manuscript_redline_vs_round5_edit.docx']:
        path=BASE/name;before=path.read_bytes();shutil.copy2(path,backup/name)
        with ZipFile(path) as z:
            infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos}
        changed=[]
        for part,data in parts.items():
            if part.startswith('word/media/') and digest(data) in replacement:
                parts[part]=replacement[digest(data)].read_bytes();changed.append(part)
        assert len(changed)==4,(name,changed)
        temp=path.with_suffix('.font-cleanup.docx')
        with ZipFile(temp,'w') as z:
            for info in infos:z.writestr(info,parts[info.filename])
        with ZipFile(temp) as new,ZipFile(path) as old:
            assert new.testzip() is None and new.namelist()==old.namelist()
            assert all(new.read(p)==old.read(p) for p in old.namelist() if p not in changed)
        assert path.read_bytes()==before
        temp.replace(path)
        documents[name]={'changed_image_parts':changed,'all_text_math_live_fields_and_other_package_parts_byte_identical':True,'sha256':digest(path.read_bytes())}
    for name in report['figures']:
        shutil.copy2(STAGE/name,BASE/'figures'/name)
        png=Path(name).with_suffix('.png')
        shutil.copy2(STAGE/png,BASE/'figures'/png)
    # Existing caption proof tools protect source identity with these hashes.
    path=BASE/'extended_data_artwork_verification.json'
    shutil.copy2(path,backup/path.name)
    ed=json.loads(path.read_text())
    for n,record in ed['figures'].items():
        if record['final_artwork'] in report['figures']:
            record['final_artwork_sha256']=digest((BASE/'figures'/record['final_artwork']).read_bytes())
    ed['font_cleanup_verification']='verification/figure_font_cleanup/installed.json'
    path.write_text(json.dumps(ed,ensure_ascii=False,indent=2)+'\n')
    report.update(backup=str(backup),documents=documents,
        notes='Figure1 already met the text-size range. ED9 retains the author-specific permitted native PNG. Supplementary figures and Methods text were not modified. Figure5d missing beta restored from executed plotting code. Native render checks follow installation.')
    (VERIFY/'installed.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Installed 13 PDFs and PNGs; updated four image parts in each working/review manuscript. Live Word XML unchanged.',flush=True)

if __name__=='__main__':
    {'build':build,'install':install}[sys.argv[1] if len(sys.argv)>1 else 'build']()
