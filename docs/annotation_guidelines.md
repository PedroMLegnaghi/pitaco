# Annotation guidelines — v1.0

These rules define the gold labels. The teacher prompt (M3) is derived from them, so a rule that is
ambiguous here becomes noise in both the gold and the silver data. When a review does not fit,
label your best guess, tick **difícil**, and write one line in **nota** — we revise the rules after
the dev round.

## The task

For each review, mark every aspect the author **evaluates**, with one sentiment per aspect.

- Label only what the text says. The star rating is hidden on purpose: the models never see it.
- An aspect counts only if the author judges it or reports an outcome about it. "Comprei um
  notebook com uma capa" alone is a fact, not an evaluation → no aspect.
- At most **one entry per aspect**. If the author says several things about the same aspect,
  pick the **dominant** sentiment (rules below).
- Evidence spans and summaries are not labeled by hand (they are scored automatically).

## Aspects

| Aspect | Covers | Does not cover |
|---|---|---|
| `entrega` | Whether, when and how the order arrived: on time, early, late, not yet received, **missing items of the order** ("recebi 1 de 2"), courier/Correios, having to pick it up, tracking | The condition of what arrived |
| `produto` | What arrived: quality, works or not, matches the ad/photo (color, size, model), **wrong item sent**, defects, **physical damage**, **missing part of one item** (screw, manual, accessory), authenticity | How it was packed |
| `embalagem` | The packaging itself: well/poorly packed, box opened or crushed, no original box, protection | Damage to the product (→ `produto`), unless the text blames the packaging; then label both |
| `atendimento` | The seller/store as a counterpart: communication, answers or **lack of answer**, after-sales, exchanges, returns, refunds, courtesy, praise or criticism of the store/seller ("loja de parabéns", "ótimo vendedor", "não compro mais nessa loja") | Waiting for an answer with no complaint yet ("aguardo resposta") → no label |
| `preco` | Price, value for money (custo-benefício), discounts, shipping cost (frete caro/grátis), price changes | Getting money back (→ `atendimento`) |
| `outro` | Website/app, payment (boleto, cartão), invoice (nota fiscal), and **generic statements with no target** ("Tudo certo", "Recomendo", "Adorei a compra") | — |

**`outro` for generic statements only when nothing else is labeled.** "Chegou no prazo, recomendo"
→ `entrega: positivo` only; the "recomendo" is already covered.

## Sentiment

| Label | Use when |
|---|---|
| `positivo` | Praise or expectation met: "chegou no prazo", "conforme anunciado", "bem embalado" |
| `negativo` | Complaint, problem, or unmet expectation: "atrasou", "ainda não recebi", "veio errado", "sem resposta" |
| `neutro` | Explicitly lukewarm ("razoável", "ok", "normal", "nada demais", "dentro do esperado") or a genuinely balanced mix |

Dominance rules, in order:
1. **A specific statement beats a generic one.** "Produto ok, porém a cor é bem mais clara que na
   foto" → `produto: negativo`.
2. **The more severe or more emphasized point wins.** "Veio um pouco amassadinha, mas gostei da
   pigmentação" → `produto: positivo`.
3. **Truly balanced → `neutro`.** Use sparingly.

Irony is labeled by intended meaning ("Parabéns, 2 meses e nada" → `entrega: negativo`).

## Worked examples (from the silver pool, never from gold)

| Review | Labels |
|---|---|
| Gostei do produto, porém entregaram apenas 1 unidade, faltou uma unidade do produto | `produto: positivo`, `entrega: negativo` |
| recebi no prazo porem a Cor veio diferente ao do anuncio | `entrega: positivo`, `produto: negativo` |
| meu produto chegou no sábado e veio faltando uma peça fundamental e agora nao consigo montar a cadeira, como faço para trocar? | `produto: negativo` (missing part of one item; the question is not yet a service complaint) |
| o produto veio com a caixa aberta. | `embalagem: negativo` |
| Recebi o produto quebrado, e até o momento não recebi nenhuma posição dos fornecedores. | `produto: negativo`, `atendimento: negativo` |
| efetuei a compra de dois toner e recebi somente 01. Já comuniquei a empresa, estou aguardando a resposta. | `entrega: negativo` (waiting is not a complaint yet) |
| Produto entregue com duas semanas antes do prazo. Ótimo custo benefício :) | `entrega: positivo`, `preco: positivo` |
| produto razoável. preço baixo, mas vai atender. | `produto: neutro`, `preco: positivo` |
| A loja está de parabéns. Obrigado pela atenção e atendimento. Produto chegou! Mas fico triste pela demora | `atendimento: positivo`, `entrega: negativo` |
| No quesito entrega vocês estão de parabéns. Chegou muito antes do previsto. Pena que o tablet trava muito | `entrega: positivo`, `produto: negativo` |
| Não posso avaliar o que não recebi | `entrega: negativo` |
| TUDO OK CONFORME ANUNCIADO. | `produto: positivo` ("conforme anunciado" = matches the ad) |
| Não recomendo e nem compraria de novo nesse site, pois quando coloquei na cesta tava um preço e fui fechar a compra o mesmo estava mais caro | `preco: negativo`, `outro: negativo` (site) |
| Tudo certo, recomendo! | `outro: positivo` (generic, nothing else labeled) |

## Process

1. Label **blind**: never look at teacher output before labeling.
2. Dev set (50) first. Then we compare notes, revise these rules to v1.1, and only then label the
   test set (250) — without changing the rules again until the test set is done.
3. Short sessions (≤ 1 h). Fatigue makes labels drift.
