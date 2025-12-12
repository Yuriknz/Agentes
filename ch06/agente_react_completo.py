# /src/ch06/agente_react_completo.py
import os
import operator
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_config

load_dotenv()

# === CONFIGURAÇÃO INICIAL ===
TAREFAS_DB: dict[int, list[dict]] = {
    1: [
        {"id": 1, "titulo": "Estudar Python", "estado": "pendente", "vencimento": "2025-12-15"},
        {"id": 2, "titulo": "Fazer compras", "estado": "concluida", "vencimento": "2025-12-10"},
    ]
}
NEXT_ID = 3

# === FERRAMENTAS (TOOLS) ===

@tool
def calcular(operacao: str, a: float, b: float) -> str:
    """Calcula operações matemáticas básicas.

    Args:
        operacao: 'somar', 'subtrair', 'multiplicar', 'dividir'
        a: Primeiro número
        b: Segundo número
    """
    operacoes = {
        "somar": lambda x, y: x + y,
        "subtrair": lambda x, y: x - y,
        "multiplicar": lambda x, y: x * y,
        "dividir": lambda x, y: x / y if y != 0 else None,
    }

    if operacao not in operacoes:
        return f"Operação '{operacao}' não reconhecida."

    try:
        resultado = operacoes[operacao](a, b)
        if resultado is None:
            return "Erro: divisão por zero."
        return f"O resultado de {a} {operacao} {b} é {resultado}"
    except Exception as e:
        return f"Erro ao calcular: {e}"

@tool
def obter_hora() -> str:
    """Obtém a hora atual do sistema."""
    agora = datetime.now()
    return f"Agora são {agora.strftime('%H:%M:%S')} do dia {agora.strftime('%d/%m/%Y')}"

@tool
def listar_tarefas(estado: Optional[str] = None) -> str:
    """Lista as tarefas do usuário.

    Args:
        estado: Filtrar por estado (pendente, concluida). Deixe vazio para todas.
    """
    config = get_config()
    usuario_id = config.get("configurable", {}).get("usuario_id", 1)
    tarefas = TAREFAS_DB.get(usuario_id, [])

    if estado:
        tarefas = [t for t in tarefas if t["estado"] == estado]

    if not tarefas:
        return "Nenhuma tarefa encontrada."

    resultado = f"Encontradas {len(tarefas)} tarefa(s):\n"
    for t in tarefas:
        emoji = "⏳" if t["estado"] == "pendente" else "✅"
        resultado += f"\n{emoji} [{t['id']}] {t['titulo']}"
        if t.get("vencimento"):
            resultado += f" (vence: {t['vencimento']})"

    return resultado

class CriarTarefaInput(BaseModel):
    titulo: str = Field(description="Título da tarefa")
    vencimento: Optional[str] = Field(default=None, description="Data de vencimento (YYYY-MM-DD)")

@tool(args_schema=CriarTarefaInput)
def criar_tarefa(titulo: str, vencimento: Optional[str] = None) -> str:
    """Cria uma nova tarefa."""
    global NEXT_ID
    config = get_config()
    usuario_id = config.get("configurable", {}).get("usuario_id", 1)

    if usuario_id not in TAREFAS_DB:
        TAREFAS_DB[usuario_id] = []

    nova_tarefa = {
        "id": NEXT_ID,
        "titulo": titulo,
        "estado": "pendente",
        "vencimento": vencimento
    }
    TAREFAS_DB[usuario_id].append(nova_tarefa)
    NEXT_ID += 1

    return f"Tarefa criada com sucesso! ID: {nova_tarefa['id']}, Título: {titulo}"

@tool
def concluir_tarefa(tarefa_id: int) -> str:
    """Marca uma tarefa como concluída."""
    config = get_config()
    usuario_id = config.get("configurable", {}).get("usuario_id", 1)
    tarefas = TAREFAS_DB.get(usuario_id, [])

    for tarefa in tarefas:
        if tarefa["id"] == tarefa_id:
            tarefa["estado"] = "concluida"
            return f"Tarefa '{tarefa['titulo']}' marcada como concluída!"

    return f"Tarefa com ID {tarefa_id} não encontrada."

ALL_TOOLS = [calcular, obter_hora, listar_tarefas, criar_tarefa, concluir_tarefa]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

# === DEFINIR ESTADO ===
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# === CONFIGURAR MODELO ===
SYSTEM_PROMPT = """Você é um assistente inteligente com acesso a ferramentas.

## Ferramentas Disponíveis
- calcular: Para fazer cálculos matemáticos (somar, subtrair, multiplicar, dividir)
- obter_hora: Para saber a hora atual
- listar_tarefas: Para listar tarefas (opcionalmente filtradas por estado)
- criar_tarefa: Para criar novas tarefas
- concluir_tarefa: Para marcar tarefas como concluídas

## Instruções
- Responda sempre em português brasileiro
- Use as ferramentas quando necessário
- Seja conciso e direto nas respostas
- Para datas, use formato DD/MM/AAAA
"""

modelo = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite"), temperature=0)
modelo_com_tools = modelo.bind_tools(ALL_TOOLS)

# === NÓS DO GRAFO ===
# Referência: seção "Padrões Reutilizáveis"

def llm_call(state: AgentState) -> dict:
    """Nó que chama o LLM."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = modelo_com_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    """Nó que executa tools."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        try:
            result = TOOLS_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
        except Exception as e:
            result = f"Erro ao executar {tool_call['name']}: {e}"

        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Função de decisão."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return "__end__"

# === CONSTRUIR E COMPILAR O GRAFO ===
def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", "__end__": END}
    )
    graph.add_edge("tool_node", "llm_call")
    return graph.compile()

# === TESTAR O AGENTE ===
def main():
    agent = create_agent()
    usuario_id = 1

    print("=== Agente ReAct Multi-Funcional ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == "sair":
            break
        if not entrada:
            continue

        resultado = agent.invoke(
            {"messages": [HumanMessage(content=entrada)]},
            config={"configurable": {"usuario_id": usuario_id}}
        )

        print(f"Agente: {resultado['messages'][-1].content}\n")

if __name__ == "__main__":
    main()