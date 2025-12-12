# /src/ch06/grafo_contador.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. Definir estado
class ContadorState(TypedDict):
    valor: int
    historico: list[str]

# 2. Definir nós
def incrementar(state: ContadorState) -> dict:
    """Incrementa o contador."""
    novo_valor = state["valor"] + 1
    return {
        "valor": novo_valor,
        "historico": state["historico"] + [f"Incrementado para {novo_valor}"]
    }

def verificar(state: ContadorState) -> dict:
    """Apenas passa pelo nó de verificação."""
    return {"historico": state["historico"] + ["Verificando..."]}

# 3. Função de decisão
def deve_continuar(state: ContadorState) -> Literal["continuar", "parar"]:
    if state["valor"] < 3:
        return "continuar"
    return "parar"

# 4. Construir grafo
grafo = StateGraph(ContadorState)

# Adicionar nós
grafo.add_node("incrementar", incrementar)
grafo.add_node("verificar", verificar)

# Adicionar arestas
grafo.add_edge(START, "incrementar")
grafo.add_edge("incrementar", "verificar")
grafo.add_conditional_edges(
    "verificar",
    deve_continuar,
    {
        "continuar": "incrementar",  # Loop!
        "parar": END
    }
)

# 5. Compilar
app = grafo.compile()

# 6. Executar
estado_inicial = {"valor": 0, "historico": ["Início"]}
resultado = app.invoke(estado_inicial)

print(f"Valor final: {resultado['valor']}")
print("Histórico:")
for item in resultado["historico"]:
    print(f"  - {item}")