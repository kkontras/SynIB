from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
TQA_ROOT = Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA")
TRAIN_JSON = TQA_ROOT / "train" / "tqa_v1_train.json"
OUT_DIR = REPO_ROOT / "artifacts" / "reports" / "tqa_examples" / "lesson_cards_synergy_summary"


@dataclass(frozen=True)
class LessonSpec:
    lesson_id: str
    question_id: str
    synergy_cue: str


SPECS = [
    LessonSpec("L_0003", "DQ_000013", "You must map the erosion-process description onto the numbered callouts in the coastal diagram."),
    LessonSpec("L_0018", "DQ_000306", "The answer requires using the tide explanation to interpret what the labeled region in the diagram represents."),
    LessonSpec("L_0057", "DQ_001060", "You need the planet-order text and the labeled solar-system graphic at the same time."),
    LessonSpec("L_0148", "DQ_002552", "The key step is distinguishing penumbra from other eclipse regions, then locating that zone in the figure."),
    LessonSpec("L_0301", "DQ_003022", "You have to combine the equinox definition with the Earth-Sun geometry shown in the diagram."),
]


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SUBTITLE = load_font(22, bold=True)
FONT_BODY = load_font(20)
FONT_TINY = load_font(14)


def text_width(draw, text, font):
    return draw.textsize(text, font=font)[0]


def wrap(draw, text, font, width):
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if text_width(draw, candidate, font) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def crop_to_box(img, box_w, box_h):
    img = img.convert("RGB")
    scale = max(box_w / img.width, box_h / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    left = max(0, (resized.width - box_w) // 2)
    top = max(0, (resized.height - box_h) // 2)
    return resized.crop((left, top, left + box_w, top + box_h))


def contain_to_box(img, box_w, box_h, bg="#F3F4F6"):
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), bg)
    left = max(0, (box_w - resized.width) // 2)
    top = max(0, (box_h - resized.height) // 2)
    canvas.paste(resized, (left, top))
    return canvas


def lesson_text_blocks(lesson):
    blocks = []
    for bucket_name in ("topics", "adjunctTopics"):
        bucket = lesson.get(bucket_name) or {}
        for _, topic in bucket.items():
            text = (topic.get("content") or {}).get("text", "")
            text = " ".join(text.split())
            if text:
                blocks.append(text)
    return blocks


def summarize_lesson(lesson):
    bullets = []
    for block in lesson_text_blocks(lesson):
        for sent in block.split(". "):
            sent = sent.strip().replace("\n", " ")
            if len(sent) < 55:
                continue
            sent = sent.rstrip(".") + "."
            bullets.append(sent)
        if len(bullets) >= 10:
            break
    keywords = ["diagram", "figure", "label", "eclipse", "planet", "water", "tide", "equinox", "shadow", "erosion"]
    ranked = []
    for bullet in bullets:
        score = sum(k in bullet.lower() for k in keywords) * 10 + min(len(bullet), 140)
        ranked.append((score, bullet))
    ranked.sort(reverse=True)
    picked, seen = [], set()
    for _, bullet in ranked:
        low = bullet.lower()
        if low in seen:
            continue
        seen.add(low)
        picked.append(bullet)
        if len(picked) == 3:
            break
    return picked or bullets[:3]


def collect_figures(lesson):
    figures = []
    for bucket_name in ("topics", "adjunctTopics"):
        bucket = lesson.get(bucket_name) or {}
        for _, topic in bucket.items():
            for fig in (topic.get("content") or {}).get("figures", []) or []:
                rel = fig.get("imagePath")
                if not rel:
                    continue
                path = TQA_ROOT / "train" / rel
                if path.exists():
                    caption = " ".join((fig.get("caption") or "").split())
                    figures.append((caption, path))
    return figures


def question_image(question):
    rel = question.get("imagePath")
    if not rel:
        return None
    path = TQA_ROOT / "train" / rel
    if path.exists():
        return path
    return None


def find_question(lesson, qid):
    questions = (lesson.get("questions") or {}).get("diagramQuestions", {}) or {}
    if qid in questions:
        return questions[qid]
    for _, q in questions.items():
        if q.get("globalID") == qid:
            return q
    raise KeyError(qid)


def panel(draw, xy, fill, outline="#D0D7DE"):
    draw.rectangle(xy, fill=fill, outline=outline)


def render_card(lesson, question, spec, out_path):
    width, height = 1800, 1180
    img = Image.new("RGB", (width, height), "#F6F1E7")
    draw = ImageDraw.Draw(img)

    margin = 34
    draw.rectangle((16, 16, width - 16, height - 16), fill="#FBF8F2", outline="#CBBFA8")

    title = f"{lesson['globalID']}  {lesson['lessonName'].title()}"
    draw.text((margin, margin), title, font=FONT_TITLE, fill="#17212B")

    q_text = (question.get('beingAsked') or {}).get('rawText') or (question.get('beingAsked') or {}).get('processedText') or str(question.get('beingAsked'))
    q_box = (margin, 90, width - margin, 182)
    panel(draw, q_box, "#E8F1EE", "#9AB7AB")
    q_lines = wrap(draw, f"Question: {q_text}", FONT_SUBTITLE, q_box[2] - q_box[0] - 28)
    y = q_box[1] + 14
    for line in q_lines[:2]:
        draw.text((q_box[0] + 14, y), line, font=FONT_SUBTITLE, fill="#163328")
        y += 30

    collage = (margin, 212, 1230, height - margin)
    side = (1260, 212, width - margin, height - margin)
    panel(draw, collage, "#FFFFFF")
    panel(draw, side, "#FFFFFF")

    draw.text((collage[0] + 16, collage[1] + 14), "Figures", font=FONT_SUBTITLE, fill="#17212B")
    q_path = question_image(question)
    lesson_figs = collect_figures(lesson)
    support_figs = []
    for caption, path in lesson_figs:
        if q_path and path == q_path:
            continue
        support_figs.append((caption, path))
        if len(support_figs) == 2:
            break

    main_box = (collage[0] + 16, collage[1] + 56, 860, 560)
    small_a = (collage[0] + 16, collage[1] + 640, 422, 250)
    small_b = (collage[0] + 454, collage[1] + 640, 422, 250)

    if q_path:
        x, y0, bw, bh = main_box
        draw.rectangle((x, y0, x + bw, y0 + bh), fill="#F3F4F6", outline="#D0D7DE")
        with Image.open(q_path) as im:
            fitted = contain_to_box(im, bw - 10, bh - 42)
        img.paste(fitted, (x + 5, y0 + 5))
        draw.rectangle((x + 5, y0 + bh - 36, x + bw - 5, y0 + bh - 5), fill="#F9FAFB")
        draw.text((x + 10, y0 + bh - 31), "Question diagram", font=FONT_TINY, fill="#374151")

    for box, item in zip((small_a, small_b), support_figs):
        caption, path = item
        x, y0, bw, bh = box
        draw.rectangle((x, y0, x + bw, y0 + bh), fill="#F3F4F6", outline="#D0D7DE")
        with Image.open(path) as im:
            fitted = contain_to_box(im, bw - 10, bh - 42)
        img.paste(fitted, (x + 5, y0 + 5))
        draw.rectangle((x + 5, y0 + bh - 36, x + bw - 5, y0 + bh - 5), fill="#F9FAFB")
        cap = caption or path.name
        cap_lines = wrap(draw, cap, FONT_TINY, bw - 16)
        yy = y0 + bh - 31
        for line in cap_lines[:1]:
            draw.text((x + 10, yy), line, font=FONT_TINY, fill="#374151")
            yy += 14

    draw.text((side[0] + 16, side[1] + 14), "Short Context Summary", font=FONT_SUBTITLE, fill="#17212B")
    y = side[1] + 58
    text_w = side[2] - side[0] - 34
    special_bridge = None
    if lesson.get("globalID") == "L_0057" and question.get("globalID") == "DQ_001060":
        special_bridge = [
            "The planets in order from the Sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "Mars is the fourth planet from the Sun.",
            "In the diagram, identify Earth first, then move one orbit outward.",
            "That next planet is labeled K, so K is Mars.",
        ]
    summary_lines = special_bridge if special_bridge else summarize_lesson(lesson)
    for bullet in summary_lines:
        lines = wrap(draw, bullet, FONT_BODY, text_w - 18)
        draw.text((side[0] + 16, y), "-", font=FONT_SUBTITLE, fill="#8A4B08")
        yy = y
        for line in lines[:4]:
            draw.text((side[0] + 36, yy), line, font=FONT_BODY, fill="#25313C")
            yy += 24
        y = yy + 12

    y += 10
    draw.text((side[0] + 16, y), "Answer Choices", font=FONT_SUBTITLE, fill="#17212B")
    choice_y = y + 38
    choices = question.get("answerChoices", {}) or {}
    for key in sorted(choices.keys()):
        choice = choices[key]
        value = choice.get('rawText') or choice.get('processedText') or str(choice)
        label = f"{key}. {value}"
        for line in wrap(draw, label, FONT_BODY, text_w):
            draw.text((side[0] + 18, choice_y), line, font=FONT_BODY, fill="#25313C")
            choice_y += 23
        choice_y += 8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main():
    with TRAIN_JSON.open() as f:
        data = json.load(f)
    lesson_map = {lesson["globalID"]: lesson for lesson in data}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, spec in enumerate(SPECS, start=1):
        lesson = lesson_map[spec.lesson_id]
        question = find_question(lesson, spec.question_id)
        out_path = OUT_DIR / f"{i:02d}_{spec.lesson_id}_{spec.question_id}_summary.png"
        render_card(lesson, question, spec, out_path)
        q_text = (question.get('beingAsked') or {}).get('rawText') or (question.get('beingAsked') or {}).get('processedText') or str(question.get('beingAsked'))
        ans = (question.get('correctAnswer') or {}).get('processedText') or (question.get('correctAnswer') or {}).get('rawText') or str(question.get('correctAnswer'))
        lines.append("\t".join([spec.lesson_id, spec.question_id, q_text, f"answer={ans}", str(out_path)]))
    (OUT_DIR / "index.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
