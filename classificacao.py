import google.generativeai as genai
import os
import textwrap
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"ERRO: Falha ao configurar a API do Gemini. Verifique sua API_KEY. Erro: {e}")
    model = None


def classificar_texto(texto: str) -> Optional[str]:
    """
    Classifica textos relacionados ao agronegócio em PRODUTO, CULTURA ou OUTROS,
    """
    if not model:
        print("MODELO INDISPONÍVEL. Não é possível classificar.")
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
        response = model.generate_content(prompt)

        resposta = response.text.strip().upper()
        print(f"DEBUG: Resposta bruta do LLM: {resposta}")
        
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


def main():
    """Função principal para testar a classificação no terminal."""

    print("Objetivo: Identificar a coleção do Astra DB (PRODUTO, CULTURA, OUTROS).")
    
    texto_para_teste = input("\nInsira o texto para classificar : ")
    
    if not texto_para_teste.strip():
        print("\n Entrada vazia. Saindo do teste.")
        return

    print("\n Analisando texto...")
    
    resultado_colecao = classificar_texto(texto_para_teste)
    
    print("\n" + "=" * 60)
    print(f"Texto Analisado: {texto_para_teste[:80]}...")
    print(f"COLEÇÃO IDENTIFICADA: {resultado_colecao}")
    print("=" * 60)
    
    if resultado_colecao in ["PRODUTO", "CULTURA", "OUTROS"]:
        print(f"\nPróximo passo: Usar a coleção '{resultado_colecao}' para buscar no Astra DB.")
    else:
        print("\n Falha na classificação. A busca RAG não seria possível.")

if __name__ == "__main__":
    main()