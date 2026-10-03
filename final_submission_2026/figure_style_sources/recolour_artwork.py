#!/usr/bin/env python3
"""Reversible colour-only migration of the verified submission artwork.

PDFs: replace RGB colour operands only; all other stream bytes are invariant.
Raster-only figures: remap palette colours and their antialiased neutral blends;
never resample, move pixels, or generate observations. Preserve all neutral pixels.
Figure 3 alone is rebuilt by its local R script, with byte-identical plotted CSVs.
"""
from pathlib import Path
from zipfile import ZipFile
import csv
import hashlib
import io
import json
import re
import shutil

import fitz
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'new submission'
VERIFY = BASE / 'verification/colour_accessibility'
BACKUP = ROOT / (VERIFY / 'backup_path.txt').read_text().strip()
STAGE = VERIFY / 'artwork'
STAGE.mkdir(exist_ok=True)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def rgb(h):
    return tuple(bytes.fromhex(h.lstrip('#')))

SET1 = dict(zip(map(rgb, ['E41A1C','377EB8','4DAF4A','984EA3','FF7F00','A65628']),
                map(rgb, ['D55E00','0072B2','666666','CC79A7','E69F00','009E73'])))
ED1 = dict(zip(map(rgb, ['F8766D','A3A500','00BF7D','00B0F6','E76BF3']),
               map(rgb, ['666666','D55E00','E69F00','0072B2','CC79A7'])))
SCHEMATIC = dict(zip(map(rgb, ['008573','E81313','1071E5']),
                    map(rgb, ['666666','D55E00','0072B2'])))
LUT = list(csv.DictReader((VERIFY/'spectral_cividis_lut.csv').open()))
OLD = np.array([rgb(r['old']) for r in LUT], dtype=float)
NEW = np.array([rgb(r['new']) for r in LUT], dtype=float)
unique, inverse = np.unique(OLD, axis=0, return_inverse=True)
# A quantized original colour can represent a narrow interval of scale values.
# Use its midpoint, preserving the original quantization rather than inventing data.
mean_new = np.array([NEW[inverse==i].mean(0) for i in range(len(unique))])
OLD, NEW = unique, mean_new
GG8 = list(map(rgb,['F8766D','CD9600','7CAE00','00BE67','00BFC4','00A9FF','C77CFF','FF61CC']))
# Eight ordered sample sizes: a single sequential scale, plus existing line types.
HCL8 = dict(zip(GG8,[rgb(LUT[int(t*(len(LUT)-1))]['new'])
                         for t in np.linspace(.88,0,8)]))
report = {'palette': {''.join(f'{v:02X}' for v in k): ''.join(f'{v:02X}' for v in v2)
                      for k,v2 in SET1.items()}, 'pdf':{}, 'raster':{}, 'docx':{}}

def remap_raster(im, mapping=None, continuous=False, region=None):
    """Conservative palette remapping; neutral artwork and all coordinates retained."""
    original = np.array(im.convert('RGBA'))
    arr = original.copy()
    h,w = arr.shape[:2]
    mask = np.ptp(arr[:,:,:3].astype(int),axis=2)>2
    if region is not None:
        x0,y0,x1,y1 = region
        clip = np.zeros_like(mask);clip[y0:y1,x0:x1]=True;mask &= clip
    vals, inv = np.unique(arr[:,:,:3][mask],axis=0,return_inverse=True)
    old = np.array(list(mapping),dtype=float) if mapping else OLD.copy()
    new = np.array(list(mapping.values()),dtype=float) if mapping else NEW.copy()
    if continuous:
        # Source-code fixed reference / fitted lines are outside the continuous scale.
        old=np.vstack([old, rgb('2297E6'),rgb('3366FF'),rgb('DF536B')])
        new=np.vstack([new, rgb('2297E6'),rgb('3366FF'),rgb('666666')])
        # Distinguish palette colours from antialias blends with neutral backgrounds.
        # Coarse tree search followed by continuous least-squares opacity recovery.
        alphas=np.linspace(1/32,1,32)
        templates=[];bases=[];backgrounds=[]
        for bg in (255.,0.):
            t=bg+alphas[:,None,None]*(old[None,:,:]-bg)
            templates.append(t.reshape(-1,3))
            bases.append(np.tile(np.arange(len(old)),len(alphas)))
            backgrounds.append(np.full(len(old)*len(alphas),bg))
        templates=np.vstack(templates); bases=np.concatenate(bases); backgrounds=np.concatenate(backgrounds)
        _, nearest=cKDTree(templates).query(vals,k=8)
        candidate=bases[nearest];bg=backgrounds[nearest,None]
        direction=old[candidate]-bg
        a=np.sum((vals[:,None,:]-bg)*direction,axis=2)/np.sum(direction**2,axis=2)
        a=np.clip(a,0,1)
        error=np.linalg.norm(vals[:,None,:]-(bg+a[:,:,None]*direction),axis=2)
        choice=error.argmin(1);idx=candidate[np.arange(len(vals)),choice]
        opacity=a[np.arange(len(vals)),choice]
        residual=error[np.arange(len(vals)),choice]
    else:
        # Removing the neutral component gives a direction independent of opacity.
        chroma=vals-vals.mean(1)[:,None]
        direction=old-old.mean(1)[:,None]
        norms=np.linalg.norm(direction,axis=1)
        _,idx=cKDTree(direction/norms[:,None]).query(chroma/np.linalg.norm(chroma,axis=1)[:,None])
        opacity=np.sum(chroma*direction[idx],axis=1)/(norms[idx]**2)
        residual=np.linalg.norm(chroma-opacity[:,None]*direction[idx],axis=1)
    accepted=(residual<=3.0)&(opacity<=1.04)&(opacity>=0)
    mapped=vals.astype(float).copy()
    mapped[accepted]+=opacity[accepted,None]*(new[idx[accepted]]-old[idx[accepted]])
    mapped=np.clip(np.rint(mapped),0,255).astype(np.uint8)
    arr[:,:,:3][mask]=mapped[inv]
    changed=np.any(arr!=original,axis=2)
    neutral=np.ptp(original[:,:,:3].astype(int),axis=2)==0
    assert np.array_equal(arr[neutral],original[neutral])
    assert np.array_equal(arr[:,:,3],original[:,:,3])
    result=Image.fromarray(arr).convert(im.mode if im.mode in ('RGB','RGBA') else 'RGB')
    stats={'size':list(im.size),'changed_pixels':int(changed.sum()),
           'neutral_pixels_identical':True,'alpha_identical':True,'no_resampling':True,
           'chromatic_pixels_left_unmatched':int((~accepted)[inv].sum())}
    return result,stats

NUMBER=rb'[+-]?(?:\d*\.\d+|\d+\.?\d*)'
COLOUR=re.compile(rb'(?<![\w.])('+NUMBER+rb')\s+('+NUMBER+rb')\s+('+NUMBER+rb')\s+(rg|RG|scn|SCN)(?!\w)')

def repair_schematic_keys(doc):
    """Replace only four header key samples; plotted paths are untouched.

    The source plotting code uses R line types 2,3,4,5 and shapes 16,17,15,3.
    The historical schematic samples used different styles and omitted shapes.
    """
    page=doc[0]
    keys={'RM':('666666','[2.2 2.2] 0','circle'),
          'RM + VT':('CC79A7','[0.55 1.65] 0','triangle'),
          '5xAM':('D55E00','[0.55 1.65 2.2 1.65] 0','square'),
          '5xAM + VT':('0072B2','[3.85 1.65] 0','plus')}
    count=0
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                if span['text'] not in keys:continue
                colour,dashes,shape=keys[span['text']]
                box=fitz.Rect(span['bbox']);cx=(box.x0+box.x1)/2+1.55;cy=box.y1+1.55
                c=tuple(v/255 for v in rgb(colour))
                page.draw_rect(fitz.Rect(cx-21,box.y1+.08,cx+21,box.y1+3.5),
                    color=None,fill=tuple(v/255 for v in (242,243,245)),overlay=True)
                page.draw_line((cx-20,cy),(cx+20,cy),color=c,width=.55,dashes=dashes,overlay=True)
                if shape=='circle':page.draw_circle((cx,cy),.85,color=c,fill=c,overlay=True)
                elif shape=='triangle':page.draw_polyline([(cx,cy-1.05),(cx-.95,cy+.8),(cx+.95,cy+.8)],
                    color=c,fill=c,closePath=True,overlay=True)
                elif shape=='square':page.draw_rect(fitz.Rect(cx-.85,cy-.85,cx+.85,cy+.85),color=c,fill=c,overlay=True)
                else:
                    page.draw_line((cx-1.1,cy),(cx+1.1,cy),color=c,width=.55,overlay=True)
                    page.draw_line((cx,cy-1.1),(cx,cy+1.1),color=c,width=.55,overlay=True)
                count+=1
    assert count==4

def recolour_pdf(src,dest,mapping):
    doc=fitz.open(src); changes=0
    text_before=[p.get_text() for p in doc]
    for x in range(1,doc.xref_length()):
        if not doc.xref_is_stream(x) or doc.xref_get_key(x,'Subtype')[1]=='/Image':continue
        raw=doc.xref_stream(x)
        def replace(m):
            nonlocal changes
            c=tuple(round(float(m[i])*255) for i in (1,2,3))
            if c not in mapping:return m[0]
            changes+=1
            return (' '.join(f'{v/255:.6f}' for v in mapping[c])+' ').encode()+m[4]
        updated=COLOUR.sub(replace,raw)
        if updated!=raw:
            # Stronger than comparing only extracted paths: every non-colour byte stays.
            assert COLOUR.sub(b'COLOUR',raw)==COLOUR.sub(b'COLOUR',updated)
            doc.update_stream(x,updated)
    image_changes=[]
    for x in sorted({i[0] for p in doc for i in p.get_images(full=True)}):
        pm=fitz.Pixmap(doc,x)
        if pm.n!=3:continue
        im=Image.frombytes('RGB',(pm.width,pm.height),pm.samples)
        # Figure 5 includes the original schematic as a bitmap. Its palette is explicit.
        if src.name=='Fig_5.pdf':
            im2,stats=remap_raster(im,SCHEMATIC)
            if stats['changed_pixels']:
                doc.update_stream(x,im2.tobytes())
                image_changes.append({'xref':x,**stats})
    if src.name=='Fig_4.pdf':repair_schematic_keys(doc)
    assert text_before==[p.get_text() for p in doc]
    doc.save(dest,garbage=4,deflate=True)
    with fitz.open(src) as orig, fitz.open(dest) as out:
        assert [tuple(p.rect) for p in orig]==[tuple(p.rect) for p in out]
        assert [p.get_text() for p in orig]==[p.get_text() for p in out]
        out[0].get_pixmap(dpi=300).save(dest.with_suffix('.png'))
    report['pdf'][src.name]={'colour_operators_changed':changes,'noncolour_stream_bytes_identical':True,
       'all_text_identical':True,'page_geometry_identical':True,'embedded_images':image_changes,
       'before_sha256':sha(src.read_bytes()),'after_sha256':sha(dest.read_bytes())}
    if src.name=='Fig_4.pdf':
        report['pdf'][src.name]['key_repair']='Four header samples corrected to plot line types and panel-d shapes; overlay confined to header samples.'

def build():
    # Original source baselines are never overwritten; migration always uses the backup.
    figures=BACKUP/'new submission/figures'
    for i in (2,4,5):recolour_pdf(figures/f'Fig_{i}.pdf',STAGE/f'Fig_{i}.pdf',SET1)
    for i in (1,2,5,6,7,8,10):
        recolour_pdf(figures/f'Extended_Data_Fig_{i}.pdf',STAGE/f'Extended_Data_Fig_{i}.pdf',ED1 if i==1 else SET1)
    for ext in ('pdf','png'):
        shutil.copy2(ROOT/f'figures_2_3_portrait/output/submission/Fig_2.{ext}',STAGE/f'Fig_3.{ext}')
    for f in (ROOT/'figures_2_3_portrait/output/submission').glob('Fig_2_panel_*.csv'):
        assert f.read_bytes()==(BACKUP/f.relative_to(ROOT)).read_bytes()
    report['figure3']={'all_three_plotted_data_csvs_byte_identical':True,
       'source':'figures_2_3_portrait/scripts/plot_fig2_submission.R','scale':'cividis',
       'panel_sizes_and_figure_size_unchanged':True}
    # Original ED PNG pixels retain all original text and geometry in the Inventory.
    for i in (1,2,5,6,7,8,9,10):
        src=BASE/f'artwork_sources/extended_data/ED{i}_manuscript.png'
        im=Image.open(src);im2,stats=remap_raster(im,ED1 if i==1 else SET1)
        im2.save(STAGE/f'ED{i}_embedded.png',dpi=(300,300))
        report['raster'][f'ED{i}']=stats
        if i==9:im2.save(STAGE/'Extended_Data_Fig_9.png',dpi=(300,300))
    for media in (6,7,8,9,10,11,12,13,14,15,16,17,19):
        im=Image.open(VERIFY/f'si_original/image{media}.png')
        mapping={k:v for k,v in SET1.items()}
        if media==7:mapping[rgb('FFA500')]=rgb('666666')
        if media==9:mapping=HCL8
        if media==15:
            # Only lower-right panel f uses the environmental continuous scale.
            # The other panels use Set1 red carrier markers and R blue reference lines.
            im2,s1=remap_raster(im,mapping,region=(0,0,im.width,2*im.height//3))
            im2,s2=remap_raster(im2,mapping,region=(0,2*im.height//3,im.width//2,im.height))
            im2,s3=remap_raster(im2,continuous=True,region=(im.width//2,2*im.height//3,im.width,im.height))
            stats={'regions':[s1,s2,s3]}
        elif media==16:
            im2,stats=remap_raster(im,continuous=True)
        else:im2,stats=remap_raster(im,mapping)
        im2.save(STAGE/f'SI_media_{media}.png',dpi=im.info.get('dpi',(300,300)))
        report['raster'][f'SI_media_{media}']=stats
    (VERIFY/'artwork_build_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built colour revisions in',STAGE)

def install():
    report=json.loads((VERIFY/'artwork_build_verification.json').read_text())
    if (VERIFY/'installed.json').exists():raise RuntimeError('Already installed; review before rerunning.')
    for filename in ('manuscript_1.docx','inventory_supporting_information.docx','supplementary_information.docx'):
        path=BASE/filename
        assert path.read_bytes()==(BACKUP/'new submission'/filename).read_bytes(),f'Concurrent edit: {filename}'
    media_replacements={}
    for n in (1,2,5,6,7,8,9,10):
        old=(BASE/f'artwork_sources/extended_data/ED{n}_manuscript.png').read_bytes()
        media_replacements[sha(old)]=(STAGE/f'ED{n}_embedded.png').read_bytes()
    for n in (6,7,8,9,10,11,12,13,14,15,16,17,19):
        old=(VERIFY/f'si_original/image{n}.png').read_bytes()
        media_replacements[sha(old)]=(STAGE/f'SI_media_{n}.png').read_bytes()
    for n in (2,3,4,5):
        old=(BACKUP/f'new submission/figures/Fig_{n}.png').read_bytes()
        media_replacements[sha(old)]=(STAGE/f'Fig_{n}.png').read_bytes()
    for filename in ('manuscript_1.docx','inventory_supporting_information.docx','supplementary_information.docx'):
        path=BASE/filename;changes=[];temp=path.with_suffix('.colour-tmp.docx')
        with ZipFile(path) as before,ZipFile(temp,'w') as after:
            for info in before.infolist():
                b=before.read(info.filename);replacement=media_replacements.get(sha(b)) if info.filename.startswith('word/media/') else None
                after.writestr(info,replacement if replacement else b)
                if replacement:changes.append(info.filename)
        with ZipFile(path) as before,ZipFile(temp) as after:
            assert after.testzip() is None
            assert before.namelist()==after.namelist()
            for name in before.namelist():
                if name not in changes:assert before.read(name)==after.read(name),name
        report['docx'][filename]={'replaced_image_parts':changes,
             'all_nonimage_parts_byte_identical':True,'math_citations_prose_layout_preserved':True,
             'before_sha256':sha(path.read_bytes()),'after_sha256':sha(temp.read_bytes())}
        temp.replace(path)
    for f in STAGE.iterdir():
        if f.name.startswith(('Fig_','Extended_Data_Fig_')):shutil.copy2(f,BASE/'figures'/f.name)
    si_dir=BASE/'figures/supplementary';si_dir.mkdir(exist_ok=True)
    for i in range(1,20):
        label='S1' if i==1 else ('S2' if i==2 else ('S2_extra_image' if i==3 else f'S{i-1}'))
        f=STAGE/f'SI_media_{i}.png'
        shutil.copy2(f if f.exists() else VERIFY/f'si_original/image{i}.png',si_dir/f'{label}.png')
    # In-place local plotting output copies remain available under historical numbers.
    for ext in ('pdf','png'):
        shutil.copy2(STAGE/f'Fig_4.{ext}',ROOT/f'figures_2_3_portrait/output/submission/Fig_3.{ext}')
    ed_path=BASE/'extended_data_artwork_verification.json';ed=json.loads(ed_path.read_text())
    for record in ed['figures'].values():
        f=BASE/'figures'/record['final_artwork']
        record['final_artwork_sha256']=sha(f.read_bytes())
    ed['colour_revision']='See verification/colour_accessibility/installed.json; original source provenance retained.'
    ed_path.write_text(json.dumps(ed,indent=2,ensure_ascii=False)+'\n')
    (VERIFY/'installed.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Installed artwork, retaining all DOCX non-image parts exactly.')

if __name__=='__main__':
    import sys
    install() if '--install' in sys.argv else build()
