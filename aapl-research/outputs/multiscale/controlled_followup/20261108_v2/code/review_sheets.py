"""Prepare page pairs for actual visual inspection; never assert acceptance."""
from common import *
from PIL import Image, ImageDraw, ImageFont


def main():
    build = read(BASE / 'audit/DOCUMENT_BUILD.json')
    out = BASE / 'manuscript/review_contact_sheets'
    out.mkdir(exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 20)
    sheets = []
    for document in build['documents']:
        name = document['document']
        folder = BASE / 'manuscript/page_previews' / name.lower()
        pages = sorted(folder.glob('page-*.png'), key=lambda p: int(p.stem.split('-')[-1]))
        assert len(pages) == document['pages']
        for start in range(0, len(pages), 2):
            selected = pages[start:start + 2]
            images = [Image.open(p).convert('RGB') for p in selected]
            width = sum(image.width for image in images) + 20 * (len(images) - 1)
            height = max(image.height for image in images) + 40
            canvas = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(canvas)
            offset = 0
            for index, image in enumerate(images):
                draw.text((offset + 10, 8), f'{name} page {start + index + 1}', fill='black', font=font)
                canvas.paste(image, (offset, 40))
                offset += image.width + 20
                image.close()
            target = out / f'{name.lower()}_{start + 1:02d}_{start + len(selected):02d}.png'
            canvas.save(target)
            sheets.append({'document': name, 'pages': list(range(start + 1, start + len(selected) + 1)),
                           'path': str(target), 'sha256': sha(target),
                           'source_pdf_sha256': document['pdf_sha256']})
    atomic(BASE / 'audit/VISUAL_CONTACT_SHEETS.json', {'status': 'prepared_for_human_or_agent_review',
        'acceptance_asserted': False, 'sheets': sheets})
    print(json.dumps({'sheets': len(sheets), 'pages': sum(len(s['pages']) for s in sheets)}))


if __name__ == '__main__':
    main()
