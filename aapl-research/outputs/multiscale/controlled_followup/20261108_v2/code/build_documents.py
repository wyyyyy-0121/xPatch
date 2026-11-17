from common import *
import subprocess
from format_documents import format_doc

def main():
    folder=BASE/'manuscript';config=BASE/'audit/fontconfig.xml'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>/mnt/c/Windows/Fonts</dir></fontconfig>')
    env=dict(os.environ,FONTCONFIG_FILE=str(config),OPENBLAS_NUM_THREADS='1')
    font=subprocess.check_output(['fc-match','Times New Roman'],env=env,text=True).strip()
    assert 'times' in font.lower(),font
    results=[]
    for name,two in [('MAIN_PAPER',True),('SUPPLEMENT',False)]:
        source=folder/f'{name}_EN.md';raw=folder/f'{name}_RAW.docx';out=folder/f'{name}_EN.docx'
        subprocess.run(['pandoc','--from=markdown+tex_math_single_backslash+autolink_bare_uris','--to=docx',
            '--standalone',f'--resource-path={folder}',f'--output={raw}',str(source)],check=True)
        format_doc(raw,out,two)
        profile=BASE/'audit'/f'lo_profile_{name.lower()}'
        subprocess.run(['libreoffice',f'-env:UserInstallation={profile.as_uri()}','--headless','--convert-to','pdf',
            '--outdir',str(folder),str(out)],env=env,check=True)
        pdf=out.with_suffix('.pdf');assert pdf.exists()
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(x.split(':',1)[1] for x in info.splitlines() if x.startswith('Pages:')))
        subprocess.run(['pdftotext','-layout',str(pdf),str(folder/f'{name}_PDF_TEXT.txt')],check=True)
        preview=folder/'page_previews'/name.lower();preview.mkdir(parents=True,exist_ok=True)
        for p in preview.glob('page-*.png'):p.unlink()
        subprocess.run(['pdftoppm','-scale-to','1400','-png',str(pdf),str(preview/'page')],check=True)
        fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        (folder/f'{name}_PDF_FONTS.txt').write_text(fonts)
        results.append({'document':name,'pages':pages,'pdf_sha256':sha(pdf),'docx_sha256':sha(out),
            'source_sha256':sha(source),'preview_pages':len(list(preview.glob('page-*.png'))),
            'font_match':font,'times_embedded':'TimesNewRoman' in fonts,'preservation':read(str(out)+'.preservation.json')})
    atomic(BASE/'audit/DOCUMENT_BUILD.json',{'status':'built_requires_manual_visual_review','documents':results,
        'main_page_target_4_to_6':4<=results[0]['pages']<=6,'visual_review_complete':False})
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
