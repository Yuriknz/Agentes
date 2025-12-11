import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Carrega a API Key do arquivo .env
load_dotenv()

def main():
    # Configuração do modelo (Temperature 0 para traduções mais precisas)
    modelo = ChatGoogleGenerativeAI(
        model=os.getenv("google_model", "gemini-2.5-flash-lite"),
        temperature=0
    )

    print("=== Tradutor PT-BR -> EN ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        texto = input("Texto em Português: ").strip()
        
        if texto.lower() == 'sair':
            break
            
        if not texto:
            continue

        # Estrutura de mensagens para tradução
        mensagens = [
            SystemMessage(content="Você é um tradutor profissional. Traduza o texto do usuário do Português Brasileiro para o Inglês. Responda apenas com a tradução"),
            HumanMessage(content=texto)
        ]

        resposta = modelo.invoke(mensagens)
        print(f"Inglês: {resposta.content}\n")

if __name__ == "__main__":
    main()