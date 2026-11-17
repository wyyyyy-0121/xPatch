"""Draw a non-data schematic of the two label-completion boundaries."""
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "chronological_split_schematic.pdf"
WIDTH, HEIGHT = 648, 286
LEFT, RIGHT = 138, 628
BLUE = HexColor("#285F81")
GOLD = HexColor("#A0622A")
GREEN = HexColor("#39735F")
TEXT = HexColor("#25313D")
MUTED = HexColor("#66737F")
GRID = Color(0.79, 0.83, 0.86)


def x(position):
    return LEFT + (RIGHT - LEFT) * position / 12


def path(c, start, end, y, colour):
    c.setStrokeColor(colour)
    c.setLineWidth(3.2)
    c.line(x(start), y, x(end), y)
    c.setFillColor(colour)
    c.circle(x(start), y, 3.8, stroke=0, fill=1)
    c.rect(x(end) - 3.4, y - 3.4, 6.8, 6.8, stroke=0, fill=1)


def main():
    c = canvas.Canvas(str(OUTPUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    c.setTitle("Chronological split and label-completion boundaries")
    c.setFillColor(white)
    c.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(18, 261, "Chronological evaluation boundaries")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 12)
    c.drawRightString(WIDTH - 18, 261, "Schematic; not to scale")

    for position, title, detail in ((5.2, "First inner-stop", "decision"),
                                    (9.0, "First outer-evaluation", "decision")):
        xpos = x(position)
        c.setStrokeColor(MUTED)
        c.setLineWidth(0.8)
        c.setDash(3, 3)
        c.line(xpos, 61, xpos, 225)
        c.setDash()
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(xpos, 236, title)
        c.setFont("Helvetica", 12)
        c.drawCentredString(xpos, 222, detail)

    lanes = (("Inner fit", 184, BLUE), ("Inner stop", 128, GOLD), ("Outer evaluation", 72, GREEN))
    for label, y, colour in lanes:
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(18, y + 2, label)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 11.5)
        c.drawString(18, y - 13, "example origins")
        c.setStrokeColor(GRID)
        c.setLineWidth(0.5)
        c.line(LEFT, y - 18, RIGHT, y - 18)

    path(c, 0.8, 3.2, 192, BLUE)
    path(c, 1.8, 4.2, 176, BLUE)
    path(c, 5.2, 7.6, 136, GOLD)
    path(c, 6.0, 8.4, 120, GOLD)
    path(c, 9.0, 11.4, 72, GREEN)

    c.setFillColor(TEXT)
    c.setFont("Helvetica", 12)
    c.drawString(18, 22, "Target path")
    c.setFillColor(BLUE)
    c.circle(150, 26, 3.5, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.drawString(160, 22, "origin / decision")
    c.setFillColor(BLUE)
    c.rect(301, 22.5, 6.5, 6.5, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.drawString(314, 22, "label completion")
    c.setStrokeColor(MUTED)
    c.setLineWidth(0.8)
    c.line(522, 26, 626, 26)
    c.line(626, 26, 620, 29)
    c.line(626, 26, 620, 23)
    c.drawRightString(626, 38, "time")
    c.showPage()
    c.save()
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
