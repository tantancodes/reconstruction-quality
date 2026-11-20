"""Design tokens shared by the capture app and the web viewer, plus the Tk drawing they need.

The values live once, in viewer/tokens.css (one `--name: value;` per line); the viewer links that
file and the app reads it here. Tk has no rounded corners, so cards, panels, pill buttons and tags
are 9-slice images drawn with Pillow and registered as ttk elements.
"""
import re
from pathlib import Path

TOKENS_CSS = Path(__file__).resolve().parents[2] / 'viewer' / 'tokens.css'
_LINE = re.compile(r'^\s*--([\w-]+)\s*:\s*([^;]+);')


def tokens(path=TOKENS_CSS):
    """{'page': '#f5f4f0', 'radius': '16px', ...} from the CSS custom properties."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


T = tokens()
PAGE, CARD, LINE, PANEL = T['page'], T['card'], T['line'], T['panel']
INK, TEXT, TEXT_2 = T['ink'], T['text'], T['text-2']
CAMERA, MESH = T['camera'], T['mesh']
RED, RED_SOFT, GREEN, GREEN_SOFT = T['red'], T['red-soft'], T['green'], T['green-soft']
AMBER, AMBER_SOFT, NEUTRAL_SOFT = T['amber'], T['amber-soft'], T['neutral-soft']
RADIUS = int(T['radius'].rstrip('px'))

# Text buttons: label and focus ring only (no border box), so they line up with plain text.
TEXT_LAYOUT = [('Button.focus', {'sticky': 'nsew', 'children': [
    ('Button.padding', {'sticky': 'nsew', 'children': [('Button.label', {'sticky': 'nsew'})]})]})]

TAG_KINDS = {'good': (GREEN_SOFT, GREEN), 'bad': (RED_SOFT, RED), 'caution': (AMBER_SOFT, AMBER),
             '': (NEUTRAL_SOFT, TEXT_2)}


def rounded(size, radius, fill, border=None, behind=None, scale=4):
    """A Tk PhotoImage of a rounded rectangle, antialiased by drawing large and shrinking."""
    from PIL import Image, ImageDraw, ImageTk
    w, h = size
    img = Image.new('RGBA', (w * scale, h * scale), behind or (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * scale - 1, h * scale - 1], radius * scale, fill=fill,
                        outline=border, width=scale if border else 0)
    return ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))


def install(style, root):
    """Register the rounded elements and their styles on a ttk.Style (after theme_use('clam')).

    Styles: Card.TFrame (white, 1px border, 16px radius), Panel.TFrame (grey visual area),
    Primary.TButton (black pill), Text.TButton / Card.Text.TButton (plain text), and
    Tag.<kind>.TLabel for kind in good, bad, caution, neutral."""
    keep = root.__dict__.setdefault('_theme_images', [])

    def element(name, img, border, padding=None, **images_by_state):
        # padding defaults to the border in Tk, which would pad every label inside a pill or tag
        keep.append(img)
        keep.extend(images_by_state.values())
        states = [(s, i) for s, i in images_by_state.items()]
        style.element_create(name, 'image', img, *states, border=border, sticky='nsew',
                             padding=border if padding is None else padding)

    # Each image is larger than twice its border, so the 9-slice has a middle to stretch.
    s = 2 * RADIUS + 12
    element('Card.bg', rounded((s, s), RADIUS, CARD, LINE, PAGE), RADIUS + 1)
    style.layout('Card.TFrame', [('Card.bg', {'sticky': 'nsew'})])     # give padding= on the widget

    p = 2 * 12 + 12
    element('Panel.bg', rounded((p, p), 12, PANEL, None, CARD), 13)
    style.layout('Panel.TFrame', [('Panel.bg', {'sticky': 'nsew'})])

    for host, bg in (('', PAGE), ('Card.', CARD)):
        h, w = 44, 64
        # vertical border 2px short of half the height: a thin middle row to stretch, ends stay round
        element(f'{host}Pill.bg', rounded((w, h), h // 2, INK, None, bg), (h // 2, h // 2 - 2), 0,
                **{'disabled': rounded((w, h), h // 2, '#b9b7b1', None, bg),
                   'pressed': rounded((w, h), h // 2, '#3a3a38', None, bg),
                   'active': rounded((w, h), h // 2, '#2a2a28', None, bg)})
        style.layout(f'{host}Primary.TButton', [(f'{host}Pill.bg', {'sticky': 'nsew', 'children': [
            ('Button.padding', {'sticky': 'nsew', 'children': [('Button.label', {'sticky': 'nsew'})]})]})])
        style.configure(f'{host}Primary.TButton', foreground='#ffffff', padding=(24, 10), anchor='center', width=0)
        style.map(f'{host}Primary.TButton', foreground=[('disabled', '#ffffff')])

        style.layout(f'{host}Text.TButton', TEXT_LAYOUT)
        style.configure(f'{host}Text.TButton', background=bg, foreground=INK, bordercolor=bg, lightcolor=bg,
                        darkcolor=bg, padding=(0, 6), relief='flat', width=0, anchor='w')
        style.map(f'{host}Text.TButton', background=[('active', bg), ('pressed', bg)],
                  foreground=[('disabled', '#b9b7b1'), ('active', TEXT_2)])

        for kind, (soft, fg) in TAG_KINDS.items():
            name = kind or 'neutral'
            t = 22
            element(f'{host}Tag.{name}.bg', rounded((t + 12, t), t // 2, soft, None, bg), (t // 2, t // 2 - 2), 0)
            style.layout(f'{host}Tag.{name}.TLabel', [(f'{host}Tag.{name}.bg', {'sticky': 'nsew', 'children': [
                ('Label.padding', {'sticky': 'nsew', 'children': [('Label.label', {'sticky': 'nsew'})]})]})])
            style.configure(f'{host}Tag.{name}.TLabel', foreground=fg, padding=(10, 3))
