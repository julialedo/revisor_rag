import google.generativeai as genai
import os
import textwrap
from typing import Optional
from dotenv import load_dotenv
# -----------------------------------------------------------
# I. CHAVES E CONFIGURAÇÕES (Do seu código anexo)
# -----------------------------------------------------------
load_dotenv() # Carrega as variáveis do arquivo .env localmente

# ❌ REMOVA A CHAVE EM TEXTO CLARO AQUI!
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Definindo o modelo como no seu notebook
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"❌ ERRO: Falha ao configurar a API do Gemini. Verifique sua API_KEY. Erro: {e}")
    model = None


# -----------------------------------------------------------
# II. FUNÇÃO DE CLASSIFICAÇÃO (Adaptada do seu código anexo)
# -----------------------------------------------------------

def classificar_texto(texto: str) -> Optional[str]:
    """
    Classifica textos relacionados ao agronegócio em PRODUTO, CULTURA ou OUTROS,
    usando a lógica e prompt fornecidos.
    """
    if not model:
        print("❌ MODELO INDISPONÍVEL. Não é possível classificar.")
        return None

    prompt = f"""Analise o texto/arquivo/diretório abaixo e classifique-o em UMA das categorias:

CATEGORIAS:
1. PRODUTO: Se refere a qualquer produto/serviço para venda ou uso agrícola.
   - Nomes comerciais de produtos (ORONDIS®, POLYTRIN, Miravis Pro, Yieldon, Seeker)
   - Argumentários de vendas, apresentações técnicas de produtos
   - Folhetos comerciais, fichas técnicas promocionais
   - Exemplos do que pode surgir: "Argumentário de vendas ORONDIS®", "Apresentação Técnica Curyom"

2. CULTURA: Se foca especificamente em uma cultura agrícola ou plantação.
   - Soja, milho, arroz, trigo, café, algodão, cana, feijão
   - Culturas específicas mencionadas no título/conteúdo principal
   - Exemplos: "Manejo de soja", "Doenças do milho", "Cultivo de trigo"

3. OUTROS: Se for um documento técnico, manual, livro, artigo, guia, publicação científica.
   - Manuais técnicos, livros acadêmicos
   - Artigos científicos, publicações de pesquisa
   - Guias de boas práticas, procedimentos
   - Materiais educacionais, apresentações acadêmicas
   - Normas, regulamentos, editais
   - Exemplos: "Manual de Identificação de Plantas Daninhas", "Fisiologia vegetal",
     "Livro Manejo de Nematoides", "Manual de boas práticas"

Texto para classificar: "{texto}"

REGRA IMPORTANTE:
1. Retorne APENAS: "produto", "cultura" ou "outros"
2. Responda com apenas uma palavra e em capslook: PRODUTO, CULTURA OU OUTROS."""

    try:
        # Gerar resposta do Gemini
        response = model.generate_content(prompt)

        # Extrair e limpar a resposta
        resposta = response.text.strip().upper()
        print(f"DEBUG: Resposta bruta do LLM: {resposta}")
        
        # Sua lógica de validação do notebook (que transforma a saída)
        if "PRODUTO" in resposta:
            return "PRODUTO"
        elif "CULTURA" in resposta:
            return "CULTURA"
        elif "OUTROS" in resposta:
            return "OUTROS"
        else:
            return f"CLASSIFICAÇÃO NÃO RECONHECIDA: {resposta}"

    except Exception as e:
        return f"ERRO ao classificar: {str(e)}"
    


import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv


# -----------------------------------------------------------
# I. CHAVES E CONFIGURAÇÕES DO ASTRA DB
# -----------------------------------------------------------

# Chaves Astra DB (adaptadas do seu notebook anexo)




load_dotenv() # Carrega as variáveis do arquivo .env localmente

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
# -----------------------------------------------------------
# II. CLASSE AstraDBClient (Do seu código anexo)
# -----------------------------------------------------------

class AstraDBClient:
    """Classe wrapper para a conexão e busca no Astra DB."""
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
        print("✅ AstraDBClient inicializado.")
        
    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
        """Realiza busca por similaridade vetorial na coleção especificada."""
        if not collection or collection == "ERRO":
            print("❌ Busca vetorial abortada: Coleção inválida ou erro na classificação.")
            return []
            
        url = f"{self.base_url}/{collection}" 
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit}
            }
        }
        
        print(f"\n--- Chamando Astra DB na Coleção: {collection} ---")
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status() 
            data = response.json()
            
            documents = data.get("data", {}).get("documents", [])
            print(f"✅ Busca realizada. Documentos retornados: {len(documents)}")
            return documents

        except requests.exceptions.HTTPError as e:
            print(f"❌ ERRO HTTP na busca Astra DB (Status: {response.status_code}): {e}")
            return []
        except Exception as e:
            print(f"❌ ERRO Geral na busca Astra DB: {str(e)}")
            return []

astra_client = AstraDBClient()



import openai
import os
import json
import hashlib
from typing import List, Dict, Optional
from dotenv import load_dotenv



load_dotenv() # Carrega as variáveis do arquivo .env localmente

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define a chave de ambiente para o cliente OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
    print("❌ ATENÇÃO: OPENAI_API_KEY não está definida.")

# -----------------------------------------------------------
# II. CLASSE LLMClient (Para gerar a correção)
# -----------------------------------------------------------

class LLMClient:
    """Classe wrapper para o cliente de Chat Completion da OpenAI, simulando 'generate_content'."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        # Inicializa o cliente OpenAI
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"✅ LLMClient inicializado com modelo: {self.model}")

    def generate_content(self, prompt: str) -> str:
        """Método que simula a interface generate_content."""
        print("\n--- Chamando OpenAI Chat Completion ---")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um agente de revisão técnica altamente preciso."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            print(f"❌ ERRO NA GERAÇÃO DO LLM (API Error): {e}")
            return f"ERRO NA GERAÇÃO DO LLM (API Error): {str(e)}"
        except Exception as e:
            print(f"❌ ERRO NA GERAÇÃO DO LLM (Geral): {e}")
            return f"ERRO NA GERAÇÃO DO LLM (Geral): {str(e)}"

# Inicializa o cliente
modelo_texto = LLMClient(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# III. FUNÇÃO get_embedding (Para a busca vetorial)
# -----------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI com diagnóstico (adaptado do seu doc)."""
    print("\n--- Chamando OpenAI Embedding ---")
    try:
        # Usa o cliente já inicializado para embeddings
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # --- DIAGNÓSTICO ---
        print(f"✅ Embedding Gerado. Dimensões: {len(embedding)}. Primeiro valor: {embedding[0]:.6f}")
        # --- FIM DIAGNÓSTICO ---

        return embedding
    except Exception as e:
        print(f"❌ ERRO na API OpenAI para Embedding: {str(e)}. Verifique se a chave está ativa.")
        # Seu fallback de hash foi removido, pois ele falha na busca RAG e queremos testar a conexão real.
        return []



def reescrever_revisor(content: str, colecao_override: Optional[str] = None) -> str:
    """
    Função principal que executa o pipeline RAG completo.
    Atua como um Revisor Técnico, corrigindo imprecisões e enriquecendo o texto.
    Aceita colecao_override para sobrepor a classificação do Gemini.
    """
    
    colecao = None
    
    if colecao_override and colecao_override != "Automática (Classificação Gemini)":
        # 1a. Usa a coleção fornecida pelo usuário
        colecao = colecao_override
        print(f"\n--- 1. COLEÇÃO DEFINIDA PELO USUÁRIO: {colecao} ---")
    else:
        # 1b. Executa a classificação normal do Gemini
        print("\n--- 1. CLASSIFICAÇÃO AUTOMÁTICA (Gemini) ---")
        colecao = classificar_texto(content)
        print(f"Coleção Identificada: {colecao}")
    
    if colecao in ["ERRO", "CLASSIFICAÇÃO NÃO RECONHECIDA:", None]:
        # Retorna a mensagem de erro como string, conforme solicitado.
        return f"Erro na classificação/seleção da coleção. Classificação falhou com: {colecao if colecao else 'ERRO'}. Não foi possível iniciar a busca RAG."

    # 2. EMBEDDING E BUSCA
    embedding = get_embedding(content[:800])
    
    if not embedding or len(embedding) < 1536:
        return "Erro fatal na geração do Embedding. Verifique sua chave OpenAI ativa. Não foi possível buscar no Astra DB."
        
    relevant_docs = astra_client.vector_search(colecao, embedding, limit=10)
    print(f"2. Busca Vetorial concluída na coleção '{colecao}'. Documentos retornados: {len(relevant_docs)}")
    
    # 3. CONSTRÓI CONTEXTO RAG
    rag_context = ""
    if relevant_docs:
        rag_context = "### REFERENCIAL TEÓRICO BUSCADO (RAG) ###\n"
        for i, doc in enumerate(relevant_docs, 1):
            doc_content = str(doc)
            doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
            rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n"
    else:
        rag_context = "Referencial teórico não retornou resultados específicos relevantes."
    
    # 4. PROMPT DE GERAÇÃO AUMENTADA (Mantendo o prompt anterior, mas removendo a 'instrucao_incremental')
    final_prompt = f"""
    Você é um **Revisor Técnico Sênior** com foco na área agrícola, rigoroso, preciso e com a missão de garantir a **veracidade científica absoluta** do texto de entrada.
    Confira se os valores estão idênticos ao banco de dados.

    Seu objetivo é:
    1. **CORRIGIR** automaticamente qualquer imprecisão, erro técnico ou erro científico no texto original.
    2. **ENRICHECER** o texto original, substituindo termos vagos por **terminologia técnica precisa** (ex: troque 'veneno' por 'defensivo agrícola' ou 'fitossanitário').
    3. **ACRESCENTAR** dados concretos, números e informações específicas, *apenas* quando o **REFERENCIAL TEÓRICO** fornecido for relevante para enriquecer ou corrigir o tópico do texto original.
    4. **MANTER** a estrutura e o tamanho do texto original (máximo delta de 5%).
    5. **PROIBIDO** adicionar informações que tangenciem ou desviem do tema central do texto original.

    ---
    ### TEXTO ORIGINAL A SER REVISADO ###
    {content}
    
    ---
    {rag_context}
    ---

    ## ESTRUTURA DE RETORNO OBRIGATÓRIA:

    Retorne o **TEXTO COMPLETAMENTE REVISADO E CORRIGIDO** primeiro.
    
    Após, coloque quais dados foram buscados no banco de dados para essa correção.

    Em seguida, adicione uma subseção chamada "🛠️ Ajustes Técnicos e Correções" listando de forma concisa cada alteração significativa feita (correção ou enriquecimento) e qual fonte foi usada.
    """

    # 5. Geração Final do LLM
    response_text = modelo_texto.generate_content(final_prompt)
        
    return response_text





# -----------------------------------------------------------
# V. FUNÇÃO ajuste_incremental (Para ajustes pós-revisão)
# -----------------------------------------------------------
# -----------------------------------------------------------
# V. FUNÇÃO ajuste_incremental (Para ajustes pós-revisão)
# -----------------------------------------------------------

def ajuste_incremental(texto_revisado: str, instrucao_incremental: str) -> str:
    """
    Aplica uma instrução incremental ao texto já revisado (saída do reescrever_revisor).
    Mantém o formato e adiciona as mudanças solicitadas.
    """
    if not instrucao_incremental:
        return texto_revisado # Retorna o texto original se não houver instrução

    print("\n--- INICIANDO AJUSTE INCREMENTAL ---")
    
    # 1. TENTA ISOLAR APENAS O TEXTO PRINCIPAL DA SAÍDA RAG
    # Isso é crucial para evitar que o LLM inclua as seções de metadados (Ajustes Técnicos) na resposta
    partes = texto_revisado.split("🛠️ Ajustes Técnicos e Correções")
    texto_principal_rag = partes[0].strip()
    
    # PROMPT DE AJUSTE INCREMENTAL REFINADO
    final_prompt = f"""
    Você é um **Editor Sênior** com a única missão de aplicar uma mudança incremental de forma fluida.
    
    Seu objetivo principal é editar o TEXTO PRINCIPAL A SER AJUSTADO:
    1. **APENAS** edite o texto para incorporar as informações da INSTRUÇÃO INCREMENTAL de forma natural, **mantendo o tom técnico**.
    2. Não é para mencionar a instrução incremental na saída.
    3. **PROIBIDO** manter ou incluir as seções de metadados ("🛠️ Ajustes Técnicos e Correções", "Dados Buscados", etc.) na sua resposta.

    ---
    ### TEXTO PRINCIPAL A SER AJUSTADO ###
    {texto_principal_rag}
    
    ---
    ### INSTRUÇÃO INCREMENTAL A SER ACRESCENTADA ###
    {instrucao_incremental}

    ---
    
    Retorne **SOMENTE O TEXTO FINAL RESULTANTE**, completamente editado e pronto.
    """

    try:
        # Usa o cliente LLM para gerar o conteúdo
        response_text = modelo_texto.generate_content(final_prompt)
        print("✅ Ajuste Incremental concluído.")
        return response_text
    except Exception as e:
        print(f"❌ ERRO na Geração do Ajuste Incremental: {str(e)}")
        return texto_revisado # Fallback para o texto original se falhar
    




import streamlit as st

# --- Configurações da Página ---
st.set_page_config(
    page_title="Corretor de Texto ",
    layout="wide"
)

# --- Título e Status Inicial ---
st.title("🛠️ Corretor de Texto ")
# 🚨 Descrição do fluxo atualizada para refletir as duas etapas
st.markdown("**Fluxo de Duas Etapas:** 1. Revisão RAG (Classificação/Busca) ➡️ 2. Ajuste Incremental (Se houver)")
st.markdown("---")

# --- Verificação de Status da Chave OpenAI ---
# Nota: A função get_embedding não é ideal para check, mas mantida para compatibilidade com o revisor.py
if not get_embedding("teste"):
    st.error("❌ ERRO CRÍTICO: Chave OpenAI INATIVA. A busca RAG falhará. Por favor, corrija a chave no 'revisor.py'.")
else:
    st.success("✅ Conexão OpenAI OK. Pronto para rodar o RAG.")
st.markdown("---")

# --- Variáveis de Estado (Simples) ---
if 'saida_final' not in st.session_state:
    st.session_state.saida_final = ""
if 'ajustes_tecnicos' not in st.session_state:
    st.session_state.ajustes_tecnicos = "Nenhum ajuste técnico realizado."
if 'colecao_usada' not in st.session_state:
    st.session_state.colecao_usada = "N/A"

# --- FUNÇÃO AUXILIAR PARA PARSEAR A SAÍDA DO RAG ---
# Como reescrever_revisor retorna uma string única, precisamos extrair o texto final e os ajustes.
def parse_rag_output(full_response: str, colecao: str) -> dict:
    if "Erro na classificação" in full_response or "Erro fatal na geração do Embedding" in full_response:
        return {
            "texto_final": full_response,
            "ajustes_tecnicos": "Falha na Etapa RAG.",
            "colecao_usada": colecao
        }

    # Tenta separar o texto principal dos ajustes técnicos
    partes = full_response.split("🛠️ Ajustes Técnicos e Correções")
    texto_final = partes[0].strip() if partes else full_response
    ajustes_tecnicos = partes[1].strip() if len(partes) > 1 else "Não foi possível extrair a seção de Ajustes Técnicos."
        
    return {
        "texto_final": texto_final,
        "ajustes_tecnicos": ajustes_tecnicos,
        "colecao_usada": colecao
    }


# --- 1. Seção de Entradas ---
st.header("Entradas do Usuário")

col1, col2 = st.columns(2)

with col1:
    texto_base = st.text_area(
        label="Texto Base para Revisão:",
        height=250, 
        placeholder="Insira o texto original aqui.",
    )

with col2:
    # Seletor Opcional de Coleção
    colecoes_disponiveis = [
        "Automática (Classificação Gemini)", # Opção padrão
        "PRODUTO",
        "CULTURA",
        "OUTROS"
    ]
    colecao_selecionada = st.selectbox(
        label="Escolha Opcional da Coleção Astra DB:",
        options=colecoes_disponiveis,
        index=0, # Inicia na opção automática
        help="Selecione uma coleção específica para busca RAG. Se 'Automática' for escolhida, a classificação Gemini será usada."
    )
    
    instrucao_incremental = st.text_area(
        label="Instrução Adicional/Incremental (Opcional):",
        height=150,
        placeholder="Ex: 'Mude o tom para formal' ou 'Aumente o segundo parágrafo em 30 palavras'."
    )
    
# --- Lógica de Execução ---

st.markdown("---")

if st.button("Aplicar Correção", type="primary"):
    
    if not texto_base:
        st.warning("Por favor, insira um Texto Base para revisão.")
    else:
        # Inicializa o resultado final com o texto base em caso de falha
        final_text = texto_base

        # ----------------------------------------------------
        # 🟢 PASSO 1: REVISÃO RAG (reescrever_revisor)
        # ----------------------------------------------------
        with st.spinner(f"1/2 Processando RAG na coleção: {colecao_selecionada}..."):
            # CHAMA A FUNÇÃO CENTRAL DO RAG
            rag_output_str = reescrever_revisor(texto_base, colecao_override=colecao_selecionada)
            
            # PARSEA A SAÍDA PARA SEPARAR O TEXTO FINAL E OS AJUSTES
            resultado_rag_parse = parse_rag_output(rag_output_str, colecao_selecionada)
            
            st.session_state.ajustes_tecnicos = resultado_rag_parse["ajustes_tecnicos"]
            st.session_state.colecao_usada = resultado_rag_parse["colecao_usada"]
            final_text = resultado_rag_parse["texto_final"]
            
            if "Erro" in final_text:
                st.error(f"❌ Erro na Etapa RAG: {final_text}")
            else:
                st.success(f"✅ Etapa 1 (RAG) Concluída. Coleção utilizada: {st.session_state.colecao_usada}")

        # ----------------------------------------------------
        # 🟠 PASSO 2: AJUSTE INCREMENTAL (ajuste_incremental)
        # ----------------------------------------------------
        if instrucao_incremental and "Erro" not in final_text:
            with st.spinner("2/2 Aplicando Ajuste Incremental..."):
                final_text = ajuste_incremental(final_text, instrucao_incremental)
            
            st.success("✨ Ajuste Incremental Aplicado.")
            st.session_state.ajustes_tecnicos += "\n\n--- AJUSTE INCREMENTAL ---\nInstrução Adicional Aplicada."
        elif instrucao_incremental and "Erro" in final_text:
             st.warning("Instrução incremental ignorada devido a um erro na etapa RAG.")


        # ----------------------------------------------------
        # 🏁 ATUALIZAÇÃO FINAL
        # ----------------------------------------------------
        st.session_state.saida_final = final_text

st.markdown("---")

# --- 2. Seção de Saída (Resultado Final) ---
st.header("Resultado Final")

# O resultado principal (texto limpo + dados buscados)
st.text_area(
    label="Texto Corrigido/Final (Resultado do RAG + Ajuste Incremental, se houver):",
    value=st.session_state.saida_final,
    height=450,
    disabled=True 
)

# A seção de ajustes técnicos e fontes (detalhes do RAG)
st.subheader("🛠️ Detalhes da Revisão")
st.code(
    f"Coleção RAG Utilizada: {st.session_state.colecao_usada}\n\n" + st.session_state.ajustes_tecnicos,
    language='markdown'
)
