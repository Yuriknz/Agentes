# /src/ch03/assistente_com_tools.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# === TOOLS ===

@tool
def somar(a: float, b: float) -> str:
    """Soma dois números."""
    return str(a + b)

@tool
def subtrair(a: float, b: float) -> str:
    """Subtrai b de a."""
    return str(a - b)

@tool
def multiplicar(a: float, b: float) -> str:
    """Multiplica dois números."""
    return str(a * b)

@tool
def dividir(a: float, b: float) -> str:
    """Divide a por b."""
    if b == 0:
        return "Erro: divisão por zero não é permitida"
    return str(a / b)

# === ASSISTENTE ===

class AssistenteCalculadora:
    def __init__(self):
        self.tools = [somar, subtrair, multiplicar, dividir]
        self.tools_por_nome = {t.name: t for t in self.tools}

        modelo = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            temperature=0
        )
        self.modelo = modelo.bind_tools(self.tools)

        self.system = SystemMessage(content="""
Você é uma calculadora inteligente.
Use as ferramentas disponíveis para fazer cálculos.
Sempre mostre o resultado de forma clara.
""")

    def processar(self, pergunta: str) -> str:
        mensagens = [self.system, HumanMessage(content=pergunta)]

        while True:
            resposta = self.modelo.invoke(mensagens)
            mensagens.append(resposta)

            # Se não há tool_calls, retornar resposta final
            if not resposta.tool_calls:
                return resposta.content

            # Executar cada tool
            for tool_call in resposta.tool_calls:
                nome = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]

                resultado = self.tools_por_nome[nome].invoke(args)

                mensagens.append(ToolMessage(
                    content=resultado,
                    tool_call_id=call_id
                ))

def main():
    calc = AssistenteCalculadora()

    print("=== Calculadora Inteligente ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
        if entrada:
            resposta = calc.processar(entrada)
            print(f"Calculadora: {resposta}\n")

if __name__ == "__main__":
    main()