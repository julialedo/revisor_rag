import requests
import json
from typing import List, Dict
import os


# 🚨 IMPORTANTE: Importa a função do arquivo classificacao.py
try:
    from classificacao import classificar_texto
    print("✅ Módulo 'classificacao.py' importado com sucesso.")
except ImportError:
    print("❌ ERRO: O arquivo 'classificacao.py' deve estar no mesmo diretório para ser importado.")
    exit()

# -----------------------------------------------------------
# I. CHAVES E CONFIGURAÇÕES DO ASTRA DB
# -----------------------------------------------------------


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

