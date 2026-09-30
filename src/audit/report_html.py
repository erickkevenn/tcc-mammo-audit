"""Relatorio de auditoria em HTML: alerta + trecho do laudo + caixa na imagem.

Rastreabilidade e o que diferencia 'auditoria' de 'mais uma caixa-preta'.
Este arquivo tambem e a figura da sua apresentacao.
"""
from __future__ import annotations
import base64
import html
from pathlib import Path

SEV_COLOR = {1: "#7f8c8d", 2: "#2d7dd2", 3: "#e0a100", 4: "#e2571e", 5: "#c0392b"}
SEV_LABEL = {1: "sem erro", 2: "nao acionavel", 3: "acionavel nao urgente",
             4: "urgente", 5: "emergente"}


def _img_tag(path: str | Path | None) -> str:
    if not path or not Path(path).exists():
        return "<em>imagem nao disponivel</em>"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lstrip(".") or "png"
    return f'<img src="data:image/{suffix};base64,{data}" style="max-width:420px;border-radius:6px">'


def render(exam_id: str, alerts: list, report_text: str,
           image_path: str | Path | None = None, out: str | Path = "audit.html") -> Path:
    rows = []
    for a in alerts:
        d = a.to_dict() if hasattr(a, "to_dict") else dict(a)
        sev = int(d.get("severity", 3))
        rows.append(
            f'<div style="border-left:5px solid {SEV_COLOR.get(sev,"#888")};'
            f'padding:10px 14px;margin:10px 0;background:#fafafa;border-radius:4px">'
            f'<b>{html.escape(d["code"])}</b> '
            f'<span style="color:{SEV_COLOR.get(sev,"#888")}">'
            f'severidade {sev} ({SEV_LABEL.get(sev,"?")})</span>'
            f'<span style="float:right;color:#666">mama {html.escape(str(d.get("laterality","?")))}</span>'
            f'<div style="margin-top:6px">{html.escape(str(d.get("message","")))}</div>'
            f'<pre style="font-size:12px;color:#555;white-space:pre-wrap;margin:6px 0 0">'
            f'{html.escape(str(d.get("evidence", {})))}</pre></div>'
        )
    body = "".join(rows) or "<p>Nenhum alerta. Laudo consistente com a imagem.</p>"
    doc = f"""<!doctype html><meta charset="utf-8">
<title>Auditoria {html.escape(exam_id)}</title>
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
            max-width:960px;margin:32px auto;line-height:1.5;color:#222">
<h1 style="margin-bottom:4px">Auditoria de laudo — {html.escape(exam_id)}</h1>
<p style="color:#666;margin-top:0">Alertas: {len(alerts)}. Este relatorio auxilia a
revisao; nao constitui diagnostico.</p>
<h2 style="font-size:17px">Alertas</h2>{body}
<h2 style="font-size:17px">Laudo</h2>
<pre style="white-space:pre-wrap;background:#f6f6f6;padding:14px;border-radius:6px;
            font-size:13px">{html.escape(report_text)}</pre>
<h2 style="font-size:17px">Evidencia visual</h2>{_img_tag(image_path)}
</div>"""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p
