#!/usr/bin/env python3
"""Remove an empty 8-point band above Figure 3a, preserving plotted scale."""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile
from lxml import etree as E
import fitz,numpy as np,hashlib,shutil,json
BASE=Path(__file__).resolve().parents[1];src=BASE/'figures/Fig_3.pdf';png=src.with_suffix('.png')
backup=BASE/'backups'/('before_figure3_whitespace_trim.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));backup.mkdir()
for f in [src,png,BASE/'manuscript_1.docx',BASE/'manuscript_redline_vs_round5_edit.docx']:shutil.copy2(f,backup/f.name)
oldpng=png.read_bytes();doc=fitz.open(src);p=doc[0];assert abs(p.rect.height-510)<.01
pm=p.get_pixmap(matrix=fitz.Matrix(4,4),alpha=False);a=np.frombuffer(pm.samples,np.uint8).reshape(pm.height,pm.width,3)
assert np.all(a[24*4:32*4]==255),'Trim region is not entirely blank'
out=fitz.open();q=out.new_page(width=510,height=502)
q.show_pdf_page(fitz.Rect(0,0,510,24),doc,0,clip=fitz.Rect(0,0,510,24),keep_proportion=False)
q.show_pdf_page(fitz.Rect(0,24,510,502),doc,0,clip=fitz.Rect(0,32,510,510),keep_proportion=False)
temp=src.with_name('Fig_3.trimmed.pdf');out.save(temp,garbage=4,deflate=True);out.close();doc.close()
check=fitz.open(temp);pm2=check[0].get_pixmap(matrix=fitz.Matrix(4,4),alpha=False);b=np.frombuffer(pm2.samples,np.uint8).reshape(pm2.height,pm2.width,3)
expected=np.concatenate([a[:24*4],a[32*4:]])
max_pixel_delta=int(np.max(np.abs(expected.astype('int16')-b.astype('int16'))))
assert max_pixel_delta<=1,'Rendered contents changed outside the blank band'
check[0].get_pixmap(dpi=300).save(png);check.close();temp.replace(src)
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
changes={}
for name in ['manuscript_1.docx','manuscript_redline_vs_round5_edit.docx']:
    f=BASE/name
    with ZipFile(f) as z:infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos}
    names=[n for n,v in parts.items() if n.startswith('word/media/') and v==oldpng];assert len(names)==1,(name,names)
    media=names[0];parts[media]=png.read_bytes();root=E.fromstring(parts['word/document.xml']);rels={x.get('Id'):x.get('Target') for x in E.fromstring(parts['word/_rels/document.xml.rels'])}
    count=0
    for drawing in root.findall('.//w:drawing',ns):
        blip=drawing.find('.//a:blip',ns)
        if blip is not None and 'word/'+rels[blip.get('{'+ns['r']+'}embed')]==media:
            for ext in drawing.findall('.//wp:extent',ns)+drawing.findall('.//a:ext',ns):
                if ext.get('cy') is not None:ext.set('cy',str(round(int(ext.get('cy'))*502/510)))
            count+=1
    assert count==1
    parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    temp=f.with_suffix('.updating.docx')
    with ZipFile(temp,'w') as z:
        for i in infos:z.writestr(i,parts[i.filename])
    temp.replace(f);changes[name]={'media':media,'placements':count,'caption_and_equations_unchanged':True}
report={'backup':str(backup),'removed_white_band_pt':[24,32],'old_dimensions_pt':[510,510],'new_dimensions_pt':[510,502],'maximum_rendered_channel_difference_at_288dpi':max_pixel_delta,'pixel_comparison':'Original rendered image with blank band removed agrees within one 8-bit intensity level.','no_scaling_of_text_or_graphics':True,'docx':changes}
(BASE/'verification/submission_completion/figure3_whitespace_trim.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
