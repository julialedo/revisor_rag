# Revisor técnico com RAG

Aplicação que revisa textos técnicos do agronegócio usando uma base de documentos como referência. Em vez de o modelo corrigir "de memória", ele busca trechos relevantes em documentos técnicos e usa esses trechos para corrigir imprecisões e completar informações.

## Como funciona

1. **Classificação**: o Gemini classifica o texto em uma de três categorias (produto, cultura agrícola ou material técnico). A categoria define em qual coleção de documentos a busca vai acontecer.
2. **Busca vetorial**: o texto vira um embedding (OpenAI `text-embedding-3-small`) e o sistema busca os trechos mais parecidos na coleção correspondente do Astra DB.
3. **Revisão**: os trechos encontrados entram no prompt como contexto, e o modelo devolve o texto revisado junto com a lista de ajustes feitos e a fonte de cada um.
4. **Ajuste incremental**: depois da revisão, dá para pedir mudanças pontuais ("deixe mais curto", "acrescente um parágrafo sobre X") sem refazer tudo.

O usuário também pode escolher a coleção manualmente, sem passar pela classificação automática.

## Estrutura

```
classificacao.py   classificação do texto com Gemini
conexao_banco.py   cliente do Astra DB (busca vetorial via API)
revisor.py         pipeline de revisão: embedding, busca, prompt e geração
main.py            interface em Streamlit
```

Tecnologias: Python, Streamlit, Gemini, OpenAI, Astra DB (banco vetorial).

## Como rodar

Crie um arquivo `.env` com:

```
GEMINI_API_KEY=
OPENAI_API_KEY=
ASTRA_DB_API_ENDPOINT=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_NAMESPACE=
```

```bash
pip install -r requirements.txt
streamlit run main.py
```
