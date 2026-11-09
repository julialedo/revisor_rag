import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

try:
    from classificacao import classificar_texto
    print("Módulo 'classificacao.py' importado com sucesso.")
except ImportError:
    print("ERRO: O arquivo 'classificacao.py' deve estar no mesmo diretório para ser importado.")
    exit()


load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")

class AstraDBClient:
    """Classe wrapper para a conexão e busca no Astra DB."""
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
        print("AstraDBClient inicializado.")
        
    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
        """Realiza busca por similaridade vetorial na coleção especificada."""
        if not collection or collection == "ERRO":
            print("Busca vetorial abortada: Coleção inválida ou erro na classificação.")
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
            print(f"Busca realizada. Documentos retornados: {len(documents)}")
            return documents

        except requests.exceptions.HTTPError as e:
            print(f"ERRO HTTP na busca Astra DB (Status: {response.status_code}): {e}")
            return []
        except Exception as e:
            print(f"ERRO Geral na busca Astra DB: {str(e)}")
            return []

astra_client = AstraDBClient()


def main():
    """Função principal para testar a busca após a classificação."""

    print("\n" + "=" * 70)
    print("--- Teste de Fluxo: Classificação (Gemini) -> Busca Astra DB ---")
    print("=" * 70)
    
    texto_para_teste = input("\nInsira o texto para classificar e buscar (Ex: 'Pragas comuns da soja'): ")
    
    if not texto_para_teste.strip():
        print("\n Entrada vazia. Saindo do teste.")
        return

    print("\n Chamando a Classificação...")
    colecao_identificada = classificar_texto(texto_para_teste)
    
    print(f"\n COLEÇÃO IDENTIFICADA: {colecao_identificada}")
    
    if colecao_identificada in ["PRODUTO", "CULTURA", "OUTROS"]:
        print(f"Iniciando busca na coleção: {colecao_identificada}")
    else:
        print(f"Não foi possível identificar uma coleção válida. Abortando busca.")
        return

    simulated_vector = [0.0] * 1536 
    simulated_vector[0] = 0.01 
    
    documentos_encontrados = astra_client.vector_search(
        collection=colecao_identificada, 
        vector=simulated_vector, 
        limit=2
    )
    
    if documentos_encontrados:
        print("\n" + "=" * 70)
        print("FLUXO DE BUSCA (CLASSIFICAÇÃO -> ASTRA DB) BEM SUCEDIDO.")
        print(f"Documentos encontrados: {len(documentos_encontrados)}")
        print("\n--- Conteúdo do 1º Documento (Vetor Omitido) ---")
        
        doc_display = documentos_encontrados[0].copy()
        if '$vector' in doc_display:
            doc_display['$vector'] = "[Vetor Omitido]"
            
        print(json.dumps(doc_display, indent=2, ensure_ascii=False))
        print("=" * 70)
        
    else:
        print("\n FLUXO DE BUSCA FALHOU. Verifique suas chaves Astra DB e a existência das coleções.")


if __name__ == "__main__":
    main()