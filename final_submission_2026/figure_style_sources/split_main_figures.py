#!/usr/bin/env python3
"""Apply the approved Figure 1 split once, without round-tripping DOCX formatting."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, json, re, shutil
from lxml import etree as E
import fitz

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'new submission'
FIG = WORK / 'figures'
SRC = WORK / 'figure_split_sources'
REPORT = WORK / 'figure_split_verification.json'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w':W[1:-1], 'm':M[1:-1],
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
MM = 72/25.4
TITLES = {1:'Empirical mating patterns span multiple independent dimensions.',
          2:'Multivariate assortative mating amplifies bias in genetic analyses.'}
PAT = re.compile(r'\bFigures?\s+([1-4])([a-f])?(?:\s*[-–/]\s*([1-4])?([a-f]))?')
def sha(data): return hashlib.sha256(data).hexdigest()
def txt(p): return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def xml(e): return E.tostring(e, encoding='UTF-8', xml_declaration=True, standalone=True)
def inner(e):
    c=deepcopy(e); E.cleanup_namespaces(c); return E.tostring(c,method='c14n')
def span_replace(p,start,end,value):
    nodes=p.findall('.//'+W+'t'); off=0; spans=[]
    for n in nodes:
        s=n.text or ''; spans.append((n,off,off+len(s)));off+=len(s)
    affected=[(n,a,b) for n,a,b in spans if a<end and b>start]
    assert affected
    for i,(n,a,b) in enumerate(affected):
        old=n.text or ''; lo=max(0,start-a);hi=min(len(old),end-a)
        n.text=old[:lo]+(value if i==0 else '')+old[hi:]
        if n.text.startswith(' ') or n.text.endswith(' '):n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
def mapped(num,panel,context):
    if num!='1': return str(int(num)+1),panel
    if panel in ('a','b'):return '1',panel
    if panel in ('c','d','e','f'):return '2',chr(ord(panel)-2)
    assert 'simulations presented in Figure 1' in context or 'results presented in Figure 1' in context,context
    return '2',None

def renumber(p,audit):
    s=txt(p); changes=[]
    for m in PAT.finditer(s):
        if re.search(r'(?:Extended\s+Data|Supplementary)\s*$',s[:m.start()]): continue
        n,panel=mapped(m[1],m[2],s)
        if n!=m[1]:changes.append((*m.span(1),n))
        if panel!=m[2]:changes.append((*m.span(2),panel))
        if m[4]:
            nn,pp=mapped(m[3] or m[1],m[4],s)
            if m[3] and nn!=m[3]:changes.append((*m.span(3),nn))
            assert m[3] or nn==n
            if pp!=m[4]:changes.append((*m.span(4),pp))
    for a,b,v in sorted(changes,reverse=True):span_replace(p,a,b,v)
    if changes:audit.append({'before':s,'after':txt(p)})

def property(p,name,attrs=None):
    pr=p.find(W+'pPr')
    if pr is None:pr=E.Element(W+'pPr');p.insert(0,pr)
    item=pr.find(W+name)
    if item is None:item=E.SubElement(pr,W+name)
    if attrs:
        for k,v in attrs.items():item.set(W+k,str(v))
    return item

def add_title(p,title):
    # Existing caption number, punctuation and their bookmarks are retained.
    marker=next(c for c in p if '(a)' in txt(c))
    run=E.Element(W+'r');rp=E.SubElement(run,W+'rPr');E.SubElement(rp,W+'b');E.SubElement(rp,W+'bCs')
    t=E.SubElement(run,W+'t');t.text=title+' ';t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    p.insert(p.index(marker),run)

def assets():
    report={}
    source=fitz.open(SRC/'Original_Fig_1.pdf');page=source[0]
    labels={s['text']:s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'] in list('cdef') and s['size']==20}
    assert set(labels)==set('cdef')
    # Strip only panel-label text; preserve all paths, plotted values and other text.
    for s in labels.values():page.add_redact_annot(fitz.Rect(s['bbox']),fill=False)
    page.apply_redactions(images=0,graphics=0)
    for old,s in labels.items():page.insert_text(s['origin'],chr(ord(old)-2),fontsize=20,fontname='hebo')
    for n,clip in [(1,fitz.Rect(0,0,907,217)),(2,fitz.Rect(0,217,907,712))]:
        doc=fitz.open();dest=doc.new_page(width=180*MM,height=180*MM*clip.height/clip.width)
        # New Figure 1 is copied from the pristine source; the lower block was relabelled.
        orig=fitz.open(SRC/'Original_Fig_1.pdf') if n==1 else source
        dest.show_pdf_page(dest.rect,orig,0,clip=clip)
        doc.save(FIG/f'Fig_{n}.pdf',garbage=4,deflate=True)
        dest.get_pixmap(dpi=300).save(FIG/f'Fig_{n}.png')
        report[str(n)]={'source':'figure_split_sources/Original_Fig_1.pdf','original_panels':'ab' if n==1 else 'cdef','current_panels':'ab' if n==1 else 'abcd','clip_pt':list(clip),'artwork_mm':[180,180*clip.height/clip.width],'vector_preserved':True}
    for old,new in [(2,3),(3,4)]:
        for ext in ('pdf','png'):
            src=ROOT/'figures_2_3_portrait/output/submission'/f'Fig_{old}.{ext}'
            shutil.copy2(src,FIG/f'Fig_{new}.{ext}')
            assert sha(src.read_bytes())==sha((FIG/f'Fig_{new}.{ext}').read_bytes())
        report[str(new)]={'source':f'figures_2_3_portrait/output/submission/Fig_{old}.pdf','byte_identical_to_approved_layout':True}
    src=fitz.open(SRC/'Original_Fig_4.pdf');doc=fitz.open();p=doc.new_page(width=170*MM,height=170*MM*720/648)
    p.show_pdf_page(p.rect,src,0);doc.save(FIG/'Fig_5.pdf',garbage=4,deflate=True);p.get_pixmap(dpi=300).save(FIG/'Fig_5.png')
    report['5']={'source':'figure_split_sources/Original_Fig_4.pdf','artwork_mm':[170,170*720/648],'content_unchanged':True,'aspect_ratio_preserved':True}
    for n in report:
        for ext in ('pdf','png'):report[n][ext+'_sha256']=sha((FIG/f'Fig_{n}.{ext}').read_bytes())
    return report

def main():
    assert not REPORT.exists(),'Already applied; use the backup for a deliberate rebuild.'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=WORK/'backups'/('before_figure_split.'+stamp);backup.mkdir()
    names=['manuscript_1.docx','manuscript_methods.docx','inventory_supporting_information.docx','supplementary_information.docx']
    for name in names+['checklist_review.json','checklist_review.md','figure_title_proposals.json','author_checklist_working.docx']:shutil.copy2(WORK/name,backup/name)
    report={'mapping':{'old_1ab':'Figure 1ab','old_1cdef':'Figure 2abcd','old_2':'Figure 3','old_3':'Figure 4','old_4':'Figure 5'},'backup_directory':str(backup.relative_to(ROOT)),'documents':{}}
    # Verify that the selected source PNGs are exactly the manuscript artwork.
    with ZipFile(WORK/'manuscript_1.docx') as z:
        assert z.read('word/media/image1.png')==(SRC/'Original_Fig_1.png').read_bytes()
        assert z.read('word/media/image4.png')==(SRC/'Original_Fig_4.png').read_bytes()
    report['assets']=assets()
    for name in names:
        path=WORK/name
        with ZipFile(path) as z: infos=z.infolist();parts={i.filename:z.read(i.filename) for i in infos}
        original=E.fromstring(parts['word/document.xml']);root=deepcopy(original);audit=[];newparts={}
        caps={}
        if name=='manuscript_1.docx':
            body=root.find(W+'body')
            caps={int(m[1]):p for p in body if (m:=re.match(r'^Figure ([1-4])\.',txt(p)))}
            assert set(caps)=={1,2,3,4}
        for p in root.iter(W+'p'):
            if p not in caps.values():renumber(p,audit)
        if caps:
            images={n:body[body.index(p)-1] for n,p in caps.items()}
            assert all(len(p.xpath('.//a:blip',namespaces=NS))==1 for p in images.values())
            p=caps[1];cut=next(i for i,c in enumerate(p) if txt(c)=='(c)')
            q=E.Element(W+'p');q.append(deepcopy(p.find(W+'pPr')))
            # New caption prefix; all old caption content is moved as original XML.
            rr=E.SubElement(q,W+'r');rp=E.SubElement(rr,W+'rPr');E.SubElement(rp,W+'b');E.SubElement(rp,W+'bCs')
            t=E.SubElement(rr,W+'t');t.text='Figure 2. ';t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
            for c in list(p)[cut:]:q.append(c)
            s=txt(q)
            for m in reversed(list(re.finditer(r'\(([c-f])\)',s))):span_replace(q,*m.span(1),chr(ord(m[1])-2))
            add_title(p,TITLES[1]);add_title(q,TITLES[2])
            newimg=deepcopy(images[1])
            for node in newimg.iter():
                for attr in list(node.attrib):
                    if E.QName(attr).localname in ('paraId','textId','anchorId','editId'):del node.attrib[attr]
            newimg.xpath('.//wp:docPr',namespaces=NS)[0].set('id','5')
            newimg.xpath('.//pic:cNvPr',namespaces=NS)[0].set('id','5')
            at=body.index(p)+1;body.insert(at,newimg);body.insert(at+1,q)
            for old in (2,3,4):
                m=re.match(r'Figure (\d)',txt(caps[old]));span_replace(caps[old],*m.span(1),str(old+1))
            currentcaps={1:p,2:q,3:caps[2],4:caps[3],5:caps[4]}
            currentimages={1:images[1],2:newimg,3:images[2],4:images[3],5:images[4]}
            # Preserve the main text section; give the five figures an A4 portrait section.
            original_sect=deepcopy(p.find(W+'pPr').find(W+'sectPr'))
            heading=next(c for c in body if txt(c)=='Figures')
            before=body[body.index(heading)-1];property(before,'sectPr')
            before.find(W+'pPr').replace(before.find(W+'pPr').find(W+'sectPr'),original_sect)
            start=body.index(heading);end=body.index(currentcaps[5])
            for elem in list(body)[start:end+1]:
                for sect in elem.findall('.//'+W+'sectPr'):sect.getparent().remove(sect)
            # Empty former section-break paragraph has no remaining purpose.
            for elem in list(body)[start:end+1]:
                if elem.tag==W+'p' and not txt(elem) and not elem.xpath('.//w:drawing',namespaces=NS):body.remove(elem)
            sep=E.Element(W+'p');sp=property(sep,'sectPr')
            E.SubElement(sp,W+'type').set(W+'val','nextPage')
            pg=E.SubElement(sp,W+'pgSz');pg.set(W+'w','11906');pg.set(W+'h','16838')
            mar=E.SubElement(sp,W+'pgMar')
            for k,v in {'top':1417,'bottom':1417,'left':850,'right':850,'header':709,'footer':709,'gutter':0}.items():mar.set(W+k,str(v))
            E.SubElement(sp,W+'cols').set(W+'space','720')
            # Do not duplicate the original first-section page-number restart.
            body.insert(body.index(currentcaps[5])+1,sep)
            rels=E.fromstring(parts['word/_rels/document.xml.rels'])
            rns=E.QName(rels).namespace;rid='rId18';assert not any(e.get('Id')==rid for e in rels)
            rel=E.SubElement(rels,'{'+rns+'}Relationship',Id=rid,Type=NS['r']+'/image',Target='media/figure2_split.png')
            for n,img in currentimages.items():
                if n==2:img.xpath('.//a:blip',namespaces=NS)[0].set('{'+NS['r']+'}embed',rid)
                imgrid=img.xpath('.//a:blip',namespaces=NS)[0].get('{'+NS['r']+'}embed')
                target=next(e.get('Target') for e in rels if e.get('Id')==imgrid)
                newparts['word/'+target]=(FIG/f'Fig_{n}.png').read_bytes()
                pdf=fitz.open(FIG/f'Fig_{n}.pdf');cx=round(pdf[0].rect.width/MM*36000);cy=round(pdf[0].rect.height/MM*36000)
                for ext in img.xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=NS):ext.set('cx',str(cx));ext.set('cy',str(cy))
                dp=img.xpath('.//wp:docPr',namespaces=NS)[0];dp.set('name',f'Figure {n}');dp.set('descr',TITLES.get(n,f'Main Figure {n}'))
                property(img,'jc',{'val':'center'});property(img,'keepNext');property(img,'spacing',{'before':0,'after':120})
                if n>1:property(img,'pageBreakBefore')
                property(currentcaps[n],'keepLines')
            newparts['word/_rels/document.xml.rels']=xml(rels)
            assert list(currentcaps)==[1,2,3,4,5]
            assert len(root.xpath('.//wp:docPr',namespaces=NS))==5
            report['caption_word_counts']={str(n):len(''.join(c.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS)).split()) for n,c in currentcaps.items()}
        # Exact preservation, including order, of math, live citation instructions and field controls.
        counts={}
        for tag in (W+'instrText',W+'fldChar',M+'oMath'):
            before=[inner(e) for e in original.iter(tag)];after=[inner(e) for e in root.iter(tag)]
            assert before==after,(name,tag,'changed')
            counts[E.QName(tag).localname]=len(before)
        newparts['word/document.xml']=xml(root)
        temp=path.with_suffix('.split.tmp.docx')
        with ZipFile(temp,'w',ZIP_DEFLATED) as z:
            for info in infos:z.writestr(info,newparts.pop(info.filename,parts[info.filename]))
            for key,data in newparts.items():z.writestr(key,data)
        with ZipFile(temp) as z:
            assert z.testzip() is None
            changed=[k for k,v in parts.items() if z.read(k)!=v]
            assert all(k in ['word/document.xml','word/_rels/document.xml.rels'] or k.startswith('word/media/') for k in changed)
        from docx import Document
        Document(temp)
        temp.replace(path)
        report['documents'][name]={'reference_paragraphs_changed':len(audit),'reference_changes':audit,'preserved_XML_element_counts':counts,'equations_and_live_fields_identical':True,'styles_fonts_numbering_and_other_parts_identical':True,'changed_existing_package_members':changed,'sha256':sha(path.read_bytes())}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    (SRC/'provenance.json').write_text(json.dumps({'source_directory_read_only':'/home/rsb/Dropbox/ftsim/round4/submission/figures','source_selection':'Original PNGs are byte-identical to working manuscript image1.png and image4.png. Paired original PDFs supply vector artwork.','files':{p.name:sha(p.read_bytes()) for p in SRC.iterdir() if p.suffix in ('.pdf','.png')}},indent=2)+'\n')
    print(json.dumps({'backup':report['backup_directory'],'captions':report['caption_word_counts'],'documents':{n:{k:v for k,v in d.items() if k!='reference_changes'} for n,d in report['documents'].items()}},indent=2))
if __name__=='__main__':main()
