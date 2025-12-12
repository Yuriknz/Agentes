# /src/ch03/executar_tools.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
import numexpr

load_dotenv()

@tool
def calcular(expressao: str) -> str:
    """Calcula uma expressão matemática."""
    try:
        resultado = numexpr.evaluate(expressao)
        return str(resultado)
    except Exception as e:
        return f"Erro: {e}"

@tool
def obter_clima(cidade: str) -> str:
    """Obtém o clima de uma cidade."""
    climas = {"são paulo": "22°C, Nublado", "rio de janeiro": "32°C, Sol"}
    return climas.get(cidade.lower(), "Dados não disponíveis")

# Mapear tools por nome
tools = [calcular, obter_clima]
tools_por_nome = {t.name: t for t in tools}

# Modelo com tools
modelo = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"))
modelo_com_tools = modelo.bind_tools(tools)

def processar_com_tools(mensagem: str) -> str:
    """Processa uma mensagem, executando tools se necessário."""
    mensagens: list[BaseMessage] = [HumanMessage(content=mensagem)]

    # Primeira chamada ao modelo
    resposta = modelo_com_tools.invoke(mensagens)
    mensagens.append(resposta)

    # Se houver tool_calls, executar
    while resposta.tool_calls:
        for tool_call in resposta.tool_calls:
            nome_tool = tool_call["name"]
            args = tool_call["args"]
            tool_call_id = tool_call["id"]
            print(f"Executando tool: {nome_tool}({args})")
            # Executar a tool
            tool_fn = tools_por_nome[nome_tool]
            resultado = tool_fn.invoke(args)
            # Adicionar resultado como ToolMessage
            mensagens.append(ToolMessage(content=resultado, tool_call_id=tool_call_id))

        # Nova chamada ao modelo com os resultados
        resposta = modelo_com_tools.invoke(mensagens)
        mensagens.append(resposta)
    return resposta.text

# Testar
print(processar_com_tools("Quanto é 25 ao quadrado?"))
print("\n---\n")
print(processar_com_tools("Como está o clima em São Paulo?"))