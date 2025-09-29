from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import re
import requests
from utils.pdf import pdf_bytes

BASE = Path(__file__).parent
WEB = BASE / "web"

app = FastAPI(title="Email AI")
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

@app.get("/")
async def home():
    return FileResponse(str(WEB / "index.html"))

padroes_improd = [
    r"\bfeliz natal\b",
    r"\bfeliz ano novo\b",
    r"\bobrigado\b",
    r"\bagradec",
    r"\bparabéns\b",
    r"\bbom dia\b",
    r"\bboa tarde\b",
    r"\bboa noite\b",
    r"\bgrato\b",
    r"\bobrigada\b",
]

padroes_prod = [
    r"\bstatus\b",
    r"\batualiza",
    r"\bsuporte\b",
    r"\bchamado\b",
    r"\bprotocolo\b",
    r"\bsolicito\b",
    r"\bsolicita\b",
    r"\bpedido\b",
    r"\banexo\b",
    r"\bdocumento\b",
    r"\babrir\s+ticket\b",
    r"\bresolver\b",
    r"\berro\b",
    r"\bproblema\b",
]

def limpar(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:20000]

def classificar_regra(t: str):
    tt = t.lower()
    imp = sum(bool(re.search(p, tt)) for p in padroes_improd)
    pro = sum(bool(re.search(p, tt)) for p in padroes_prod)
    if pro > imp:
        return "Produtivo", min(1.0, 0.6 + 0.1 * pro), ["prod"]
    else:
        return "Improdutivo", min(1.0, 0.6 + 0.1 * imp), ["imp"]

def eh_saudacao_agradecimento_puro(t: str) -> bool:
    tt = t.lower()
    if any(re.search(p, tt) for p in padroes_prod):
        return False
    texto_limpo = re.sub(
        r"\b(bom dia|boa tarde|boa noite|obrigad\w*|agradec\w*|grato|feliz natal|feliz ano novo|parabéns)\b",
        "",
        tt,
        flags=re.I,
    )
    texto_limpo = re.sub(r"[^\w]+", "", texto_limpo, flags=re.I)
    return len(texto_limpo.strip()) == 0 and len(tt) <= 120

HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/models/joeddav/xlm-roberta-large-xnli",
)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

LABELS_HF = [
    "Produtivo: requer ação, pedido, dúvida, problema, erro, prazo ou suporte",
    "Improdutivo: apenas saudação/agradecimento (ex.: obrigado, bom dia), sem pedido",
]

def _normalizar_categoria(label_previsto: str) -> str:
    lp = label_previsto.strip().lower()
    if lp.startswith("improdutivo"):
        return "Improdutivo"
    if lp.startswith("produtivo"):
        return "Produtivo"
    return "Improdutivo"

def classificar(texto: str):
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    inputs_ctx = (
        "Contexto: classifique como PRODUTIVO apenas se há pedido, dúvida, problema, erro, anexo, prazo ou necessidade de suporte. "
        "Se for apenas saudação/agradecimento (ex.: obrigado, bom dia), classifique como IMPRODUTIVO. "
        f"Mensagem: {texto[:3500]}"
    )
    payload = {
        "inputs": inputs_ctx,
        "parameters": {
            "candidate_labels": LABELS_HF,
            "multi_label": False,
            "hypothesis_template": "Este email é {}.",
        },
    }
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        d = r.json()
        if isinstance(d, list):
            labels = d[0].get("labels", [])
            scores = d[0].get("scores", [])
        else:
            labels = d.get("labels", [])
            scores = d.get("scores", [])
        if not labels or not scores:
            return None
        i = max(range(len(scores)), key=lambda k: scores[k])
        categoria = _normalizar_categoria(labels[i])
        extra = {"provider": "hf", "labels": labels, "scores": scores}
        return categoria, float(scores[i]), extra
    except Exception:
        return None

def assunto(c: str) -> str:
    return "Re: Sua solicitação" if c == "Produtivo" else "Re: Mensagem recebida"

def resposta(c: str) -> str:
    if c == "Produtivo":
        return (
            "Olá!\n\n"
            "Recebemos sua mensagem e já estamos analisando. Para agilizar, poderia confirmar:\n"
            "- Número do chamado/protocolo (se houver)\n"
            "- Descrição breve do problema/solicitação\n"
            "- Anexos relevantes (prints, documentos)\n\n"
            "Nosso SLA padrão para primeira resposta é de até 4 horas úteis.\n"
            "Ficamos à disposição.\n\n"
            "Atenciosamente,\nEquipe de Suporte"
        )
    return (
        "Olá!\n\n"
        "Obrigado pela sua mensagem. Registramos seu contato.\n"
        "Se precisar de suporte ou acompanhamento de alguma solicitação, responda este email com os detalhes.\n\n"
        "Abraços,\nEquipe"
    )

@app.post("/api/classificar")
async def api_classificar(
    raw_text: str | None = Form(None),
    file: UploadFile | None = File(None),
    return_debug: bool = Form(False),
):
    try:
        texto = (raw_text or "").strip()

        filename = ""
        if file is not None:
            filename = ((getattr(file, "filename", None) or "").strip().lower())
        if filename:
            assert file is not None
            conteudo = await file.read()
            if filename.endswith(".txt"):
                texto = conteudo.decode("utf-8", errors="ignore")
            elif filename.endswith(".pdf"):
                texto = pdf_bytes(conteudo)
            else:
                raise HTTPException(400, "Formato não suportado. Use .txt ou .pdf")

        texto = limpar(texto)
        if not texto:
            raise HTTPException(400, "Conteúdo vazio")

        LIMIAR_IA = 0.60
        r_ai = classificar(texto)

        if r_ai:
            categoria, score, extra = r_ai
            if score < LIMIAR_IA:
                if eh_saudacao_agradecimento_puro(texto):
                    categoria, score = "Improdutivo", 0.97
                    extra = {"provider": "rule", "why": "saudacao_ou_agradecimento_puro"}
                else:
                    categoria, score, hits = classificar_regra(texto)
                    extra = {"provider": "heuristic", "hits": hits}
        else:
            if eh_saudacao_agradecimento_puro(texto):
                categoria, score = "Improdutivo", 0.97
                extra = {"provider": "rule", "why": "saudacao_ou_agradecimento_puro"}
            else:
                categoria, score, hits = classificar_regra(texto)
                extra = {"provider": "heuristic", "hits": hits}

        result = {
            "category": categoria,
            "score": score,
            "suggested_subject": assunto(categoria),
            "suggested_reply": resposta(categoria),
            "meta": {"chars": len(texto)},
        }
        if return_debug:
            result["explanations"] = extra
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))