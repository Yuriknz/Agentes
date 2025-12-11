# assistente_simples.py
import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def criar_assistente():
    """Cria e retorna uma instância do modelo."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("google_model", "gemini-2.5-flash-lite"),
        temperature=0.7,
        max_output_tokens=1024,
        max_retries=3,
        timeout=30
    )

def conversar(modelo, pergunta: str) -> str:
    """Envia uma pergunta ao modelo e retorna a resposta."""
    mensagens = [
        SystemMessage(content="Você é um assistente prestativo que responde em português."),
        HumanMessage(content=pergunta)
    ]
    resposta = modelo.invoke(mensagens)
    return resposta.content

def main():
    print("=== Assistente Simples ===")
    print("Digite 'sair' para encerrar.\n")

    modelo = criar_assistente()

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == 'sair':
            print("Até logo!")
            break

        if not pergunta:
            continue

        resposta = conversar(modelo, pergunta)
        print(f"Assistente: {resposta}\n")

if __name__ == "__main__":
    main()