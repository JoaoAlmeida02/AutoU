
# Email AI – FastAPI + Hugging Face (Zero‑Shot) – V2

Correção de envio somente texto pela UI e tratamento de arquivo vazio no backend.

## Como rodar localmente

```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate  |  PowerShell (alternativa): .\.venv\Scripts\python.exe -m pip install -r requirements.txt
pip install -r requirements.txt
# PowerShell
$env:HF_API_TOKEN="seu_token"
# CMD
#set HF_API_TOKEN=seu_token
uvicorn main:app --reload
```

Acesse http://127.0.0.1:8000/

## Deploy no Vercel

1. Suba para o GitHub.
2. No Vercel, Import Project.
3. Em Settings → Environment Variables, adicione `HF_API_TOKEN`.
4. Deploy.

## Notas da V2
- UI envia `file` somente quando realmente existe um arquivo.
- Backend ignora `file` sem nome e faz parsing resiliente de `return_debug`.
- Caso a IA não responda, usa heurística local automaticamente.
