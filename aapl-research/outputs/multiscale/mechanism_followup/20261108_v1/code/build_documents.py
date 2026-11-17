from common import *
import subprocess
from format_documents import format_doc


def main():
    folder=BASE/'manuscript'
    config=BASE/'audit/fontconfig.xml'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>/mnt/c/Windows/Fonts</dir></fontconfig>')
    env=dict(os.environ,FONTCONFIG_FILE=str(config))
    font=subprocess.check_output(['fc-match','Times New Roman'],env=env,text=True).strip()
    assert 'times' in font.lower(),font
    generated=[]
    for name,two in [('MAIN_PAPER',True),('SUPPLEMENT',False)]:
        source=folder/f'{name}_EN.md'; raw=folder/f'{name}_RAW.docx'; doc=folder/f'{name}_EN.docx'
        subprocess.run(['pandoc','--from=markdown+tex_math_single_backslash+autolink_bare_uris',
            '--to=docx','--standalone',f'--resource-path={folder}',f'--output={raw}',str(source)],check=True)
        format_doc(raw,doc,two)
        profile=BASE/'audit'/f'lo_profile_{name.lower()}'
        subprocess.run(['libreoffice',f'-env:UserInstallation={profile.as_uri()}',
            '--headless','--convert-to','pdf','--outdir',str(folder),str(doc)],env=env,check=True)
        pdf=doc.with_suffix('.pdf'); assert pdf.exists()
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(x.split(':',1)[1] for x in info.splitlines() if x.startswith('Pages:')))
        subprocess.run(['pdftotext','-layout',str(pdf),str(folder/f'{name}_PDF_TEXT.txt')],check=True)
        rendered=folder/'page_previews'/name.lower();rendered.mkdir(parents=True,exist_ok=True)
        for stale in rendered.glob('page-*.png'):stale.unlink()
        subprocess.run(['pdftoppm','-scale-to','1400','-png',str(pdf),str(rendered/'page')],check=True)
        font_report=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        (folder/f'{name}_PDF_FONTS.txt').write_text(font_report)
        generated.append({'name':name,'docx_sha256':sha(doc),'pdf_sha256':sha(pdf),'pages':pages,
            'previews':len(list(rendered.glob('page-*.png'))),'font_match':font,
            'pdf_times_font_embedded':'TimesNewRoman' in font_report or 'TimesNewRomanPS' in font_report,
            'preservation':read(Path(str(doc)+'.preservation.json'))})
    atomic(BASE/'audit/DOCUMENT_BUILD.json',{'status':'built_requires_visual_review','documents':generated,
        'main_page_target_4_to_6':4<=generated[0]['pages']<=6,'visual_review_complete':False})


if __name__=='__main__': main()
