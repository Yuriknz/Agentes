# /src/ch03/binding_tools.py
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# Definir tools
@tool
def calcular(expressao: str) -> str:
    """Calcula uma expressão matemática simples.

    Args:
        expressao: Expressão matemática (ex: "2 + 2", "10 * 5")
    """
    try:
        # ATENÇÃO: eval() é perigoso em produção!
        # Use uma biblioteca segura como numexpr
        resultado = eval(expressao)
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Erro no cálculo: {e}"

@tool
def obter_clima(cidade: str) -> str:
    """Obtém informações sobre o clima de uma cidade.

    Args:
        cidade: Nome da cidade
    """
    # Simulação - em produção, chamaria uma API real
    climas = {
        "são paulo": "Nublado, 22°C",
        "rio de janeiro": "Ensolarado, 32°C",
        "curitiba": "Chuvoso, 15°C",
    }
    return climas.get(cidade.lower(), f"Clima não disponível para {cidade}")

if __name__ == "__main__":
    # Criar modelo COM tools bindadas
    from langchain_google_genai import ChatGoogleGenerativeAI
    modelo = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gpt-4o-mini"))
    modelo_com_tools = modelo.bind_tools([calcular, obter_clima])

    # Testar - o modelo decide qual tool usar
    resposta = modelo_com_tools.invoke([
        HumanMessage(content="Quanto é 15 vezes 8?")
    ])

    print(f"Conteúdo: {resposta.content}")
    print(f"Tool calls: {resposta.tool_calls}")