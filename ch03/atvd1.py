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

@tool
def converter_temperatura(valor: float, de: str, para: str) -> str:
    """
    Converte temperatura entre Celsius (C), Fahrenheit (F) e Kelvin (K).
    Args:
        valor: O valor numérico da temperatura.
        de: A escala de origem ('C', 'F' ou 'K').
        para: A escala de destino ('C', 'F' ou 'K').
    """
    de = de.upper().strip()
    para = para.upper().strip()
    
    # 1. Converter tudo para Celsius primeiro (como base)
    if de == 'C':
        temp_c = valor
    elif de == 'F':
        temp_c = (valor - 32) * 5 / 9
    elif de == 'K':
        temp_c = valor - 273.15
    else:
        return "Erro: Escala de origem desconhecida. Use C, F ou K."

    # 2. Converter de Celsius para a escala de destino
    if para == 'C':
        return f"{temp_c:.2f} C"
    elif para == 'F':
        res = (temp_c * 9 / 5) + 32
        return f"{res:.2f} F"
    elif para == 'K':
        res = temp_c + 273.15
        return f"{res:.2f} K"
    else:
        return "Erro: Escala de destino desconhecida. Use C, F ou K."

# === ASSISTENTE ===

class AssistenteCalculadora:
    def __init__(self):
        # A lista de tools permanece a mesma, pois a função apenas foi atualizada
        self.tools = [somar, subtrair, multiplicar, dividir, converter_temperatura]
        self.tools_por_nome = {t.name: t for t in self.tools}

        modelo = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"),
            temperature=0
        )
        self.modelo = modelo.bind_tools(self.tools)

        self.system = SystemMessage(content="""
Você é um assistente inteligente de cálculos e conversões.
Use as ferramentas disponíveis para fazer contas matemáticas ou converter unidades (Celsius, Fahrenheit, Kelvin).
Para conversões, identifique a escala de origem e a de destino.
Sempre mostre o resultado de forma clara.
""")

    def processar(self, pergunta: str) -> str:
        mensagens = [self.system, HumanMessage(content=pergunta)]

        while True:
            resposta = self.modelo.invoke(mensagens)
            mensagens.append(resposta)

            if not resposta.tool_calls:
                return resposta.content

            for tool_call in resposta.tool_calls:
                nome = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]

                if nome in self.tools_por_nome:
                    try:
                        resultado = self.tools_por_nome[nome].invoke(args)
                    except Exception as e:
                        resultado = f"Erro ao executar ferramenta: {str(e)}"
                else:
                    resultado = f"Erro: Ferramenta {nome} não encontrada."

                mensagens.append(ToolMessage(
                    content=str(resultado),
                    tool_call_id=call_id
                ))

def main():
    calc = AssistenteCalculadora()

    print("=== Calculadora e Conversor (C/F/K) ===")
    print("Ex: 'Converta 300 Kelvin para Celsius' ou 'Quanto é 0 C em K?'")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
        if entrada:
            try:
                resposta = calc.processar(entrada)
                print(f"Assistente: {resposta}\n")
            except Exception as e:
                print(f"Ocorreu um erro: {e}\n")

if __name__ == "__main__":
    main()