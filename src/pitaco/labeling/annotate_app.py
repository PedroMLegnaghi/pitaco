"""Local Gradio app for blind gold labeling (`pitaco annotate --split dev`).

Blind by construction: it reads only the review text — never the star rating, never teacher output.
Bound to 127.0.0.1 with no share link, so review text never leaves the machine.
"""

from pathlib import Path

import gradio as gr
import pandas as pd

from pitaco.labeling.annotations import (
    CHOICES,
    annotation_order,
    append_annotation,
    build_annotation,
    choices_from_annotation,
    load_annotations,
)
from pitaco.schema import ASPECTS

ASPECT_HELP = {
    "entrega": "prazo, atraso, não recebi, faltou item do pedido, Correios",
    "produto": "qualidade, defeito, dano, item errado, cor/tamanho, peça faltando",
    "embalagem": "bem/mal embalado, caixa aberta ou amassada",
    "atendimento": "loja/vendedor, resposta, troca, devolução, reembolso",
    "preco": "preço, custo-benefício, frete",
    "outro": "site, pagamento, nota fiscal; genérico só se nada mais",
}


def build_app(reviews_path: Path, annotations_path: Path, seed: int, annotator: str) -> gr.Blocks:
    reviews = pd.read_json(reviews_path, lines=True).set_index("review_id")
    order = annotation_order(reviews.index.tolist(), seed)

    def first_unlabeled() -> int:
        done = load_annotations(annotations_path)
        return next((i for i, rid in enumerate(order) if rid not in done), len(order) - 1)

    def render(index: int):
        done = load_annotations(annotations_path)
        rid = order[index]
        ann = done.get(rid)
        choices = choices_from_annotation(ann)
        status = "✓ rotulado" if ann else "não rotulado"
        progress = f"**{len(done)} / {len(order)}** rotulados · review {index + 1} · {status}"
        text = f"### {reviews.loc[rid, 'text']}"
        return (
            index,
            progress,
            text,
            *[choices[a] for a in ASPECTS],
            ann.difficult if ann else False,
            ann.note if ann else "",
        )

    def save_and_next(index: int, *values):
        *radios, difficult, note = values
        ann = build_annotation(
            order[index],
            dict(zip(ASPECTS, radios, strict=True)),
            difficult=difficult,
            note=note,
            annotator=annotator,
        )
        append_annotation(annotations_path, ann)
        return render(min(index + 1, len(order) - 1))

    with gr.Blocks(title="Pitaco · anotação") as app:
        index = gr.State(0)
        progress = gr.Markdown()
        text = gr.Markdown()
        radios = [
            gr.Radio(list(CHOICES), value=CHOICES[0], label=f"{a} — {ASPECT_HELP[a]}")
            for a in ASPECTS
        ]
        with gr.Row():
            difficult = gr.Checkbox(label="difícil / ambíguo")
            note = gr.Textbox(label="nota (opcional)", lines=1)
        with gr.Row():
            prev_btn = gr.Button("◀ anterior")
            next_btn = gr.Button("pular ▶")
            save_btn = gr.Button("salvar e próximo ✓", variant="primary")

        outputs = [index, progress, text, *radios, difficult, note]
        # Resume at the first unlabeled review on every page load, not at build time.
        app.load(lambda: render(first_unlabeled()), outputs=outputs)
        save_btn.click(save_and_next, inputs=[index, *radios, difficult, note], outputs=outputs)
        prev_btn.click(lambda i: render(max(i - 1, 0)), inputs=index, outputs=outputs)
        next_btn.click(lambda i: render(min(i + 1, len(order) - 1)), inputs=index, outputs=outputs)
    return app
