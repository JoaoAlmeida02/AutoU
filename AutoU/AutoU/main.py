from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os, re, requests
from utils.pdf import pdf_bytes

BASE = Path(__file__).parent
WEB = BASE / 'web'
app = FastAPI(title='Email AI')
app.mount('/static', StaticFiles(directory=str(WEB)), name='static')

@app.get('/')
async def home():
    return FileResponse(str(WEB / 'index.html'))

padroes_improd = [
    r'\bfeliz natal\b', r'\bfeliz ano novo\b', r'\bobrigado\b', r'\bagradec',
    r'\bparabéns\b', r'\bbom dia\b', r'\bboa tarde\b', r'\bboa noite\b',
    r'\bgrato\b', r'\bobrigada\b'
]
padroes_prod = [
    r'\bstatus\b', r'\batualiza', r'\bsuporte\b', r'\bchamado\b', r'\bprotocolo\b',
    r'\bsolicito\b', r'\bsolicita\b', r'\bpedido\b', r'\banexo\b', r'\bdocumento\b',
    r'\babrir\s+ticket\b', r'\bresolver\b', r'\berro\b', r'\bproblema\b'
]

def limpar(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace('\r\n', '\n').replace('\r', '\n').strip()
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t[:20000]

def classificar_regra(t: str):
    tt = t.lower()
    imp = sum(bool(re.search(p, tt)) for p in padroes_improd)
    pro = sum(bool(re.search(p, tt)) for p in padroes_prod)
    if pro > imp:
        return 'Produtivo', min(1.0, 0.6 + 0.1 * pro), ['prod']
    else:
        return 'Improdutivo', min(1.0, 0.6 + 0.1 * imp), ['imp']

# Preferi endpoint por modelo (plug-and-play). Se quiser Router depois, me fala.
HF_API_URL = os.getenv(
    'HF_API_URL',
    'https://api-inference.huggingface.co/models/joeddav/xlm-roberta-large-xnli'
)
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
rotulos = ['Produtivo', 'Improdutivo']

def classificar(texto: str):
    """Tenta IA; retorna (categoria, score, extra) OU (None, hf_error)"""
    if not HF_API_TOKEN:
        return None, {'reason': 'missing_token'}
    headers = {'Authorization': f'Bearer {HF_API_TOKEN}'}
    payload = {'inputs': texto[:4000],
               'parameters': {'candidate_labels': rotulos, 'multi_label': False}}
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=25)
    except Exception as e:
        return None, {'reason': 'exception', 'exc': str(e)}

    if r.status_code != 200:
        # Não quebra a API: devolve erro para debug e deixa fallback agir
        return None, {'reason': 'non_200', 'status': r.status_code, 'text': r.text[:400]}

    try:
        d = r.json()
        labels = (d.get('labels') if isinstance(d, dict) else d[0].get('labels'))
        scores = (d.get('scores') if isinstance(d, dict) else d[0].get('scores'))
        if not labels or not scores:
            return None, {'reason': 'no_labels'}
        i = max(range(len(scores)), key=lambda x: scores[x])
        extra = {'provider': 'hf', 'labels': labels, 'scores': scores}
        return (labels[i], float(scores[i]), extra), None
    except Exception as e:
        return None, {'reason': 'bad_json', 'exc': str(e), 'body': r.text[:400]}

def assunto(c: str) -> str:
    return 'Re: Sua solicitação' if c == 'Produtivo' else 'Re: Mensagem recebida'

def resposta(c: str) -> str:
    if c == 'Produtivo':
        return ('Olá!\n\nRecebemos sua mensagem e já estamos analisando. Para agilizar, poderia confirmar:\n'
                '- Número do chamado/protocolo (se houver)\n'
                '- Descrição breve do problema/solicitação\n'
                '- Anexos relevantes (prints, documentos)\n\n'
                'Nosso SLA padrão para primeira resposta é de até 4 horas úteis.\n'
                'Ficamos à disposição.\n\n'
                'Atenciosamente,\nEquipe de Suporte')
    return ('Olá!\n\nObrigado pela sua mensagem. Registramos seu contato.\n'
            'Se precisar de suporte ou acompanhamento de alguma solicitação, responda este email com os detalhes.\n\n'
            'Abraços,\nEquipe')

@app.post('/api/classificar')
async def api_classificar(
    raw_text: str | None = Form(None),
    file: UploadFile | None = File(None),
    return_debug: bool = Form(False)
):
    try:
        texto = (raw_text or '').strip()

        # Só processa arquivo se existe e tem nome; senão, usa raw_text
        if file and (file.filename or '').strip():
            nome = file.filename.lower().strip()
            conteudo = await file.read()
            if nome.endswith('.txt'):
                texto = conteudo.decode('utf-8', errors='ignore')
            elif nome.endswith('.pdf'):
                texto = pdf_bytes(conteudo)
            else:
                raise HTTPException(400, 'Formato não suportado. Use .txt ou .pdf')

        texto = limpar(texto)
        if not texto:
            raise HTTPException(400, 'Conteúdo vazio')

        ai_res, ai_err = classificar(texto)
        if ai_res:
            categoria, score, extra = ai_res
        else:
            categoria, score, hits = classificar_regra(texto)
            extra = {'provider': 'heuristic', 'hits': hits}
            if ai_err:
                extra['hf_error'] = ai_err  # aparece só com return_debug=True

        result = {
            'category': categoria,
            'score': score,
            'suggested_subject': assunto(categoria),
            'suggested_reply': resposta(categoria),
            'meta': {'chars': len(texto)}
        }
        if return_debug:
            result['explanations'] = extra

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))