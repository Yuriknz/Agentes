# /src/ch03/tool_tarefas.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Literal

from enum import Enum

class EstadoTarefa(str, Enum):
    PENDENTE = "pendente"
    CONCLUIDA = "concluida"
    ARQUIVADA = "arquivada"

class ListarTarefasInput(BaseModel):
    """Schema para listagem de tarefas."""
    estado: Optional[Literal["pendente", "concluida", "arquivada"]] = Field(
        default=None,
        description="Filtrar por estado da tarefa"
    )
    categoria: Optional[str] = Field(
        default=None,
        description="Filtrar por categoria"
    )

# Banco de dados simulado
TAREFAS_DB = [
    {"id": 1, "titulo": "Estudar Python", "estado": "pendente", "categoria": "Estudos"},
    {"id": 2, "titulo": "Fazer compras", "estado": "concluida", "categoria": "Pessoal"},
    {"id": 3, "titulo": "Reunião de equipe", "estado": "pendente", "categoria": "Trabalho"},
]

@tool(args_schema=ListarTarefasInput)
def listar_tarefas(
    estado: Optional[str] = None,
    categoria: Optional[str] = None
) -> str:
    """Lista as tarefas do usuário com filtros opcionais.

    Use esta ferramenta para mostrar tarefas existentes.
    Pode filtrar por estado (pendente, concluida, arquivada) e/ou categoria.
    """
    tarefas = TAREFAS_DB.copy()

    # Aplicar filtros
    if estado:
        tarefas = [t for t in tarefas if t["estado"] == estado]
    if categoria:
        tarefas = [t for t in tarefas if t["categoria"].lower() == categoria.lower()]

    if not tarefas:
        return "Nenhuma tarefa encontrada com os filtros especificados."

    # Formatar resultado
    resultado = f"Encontradas {len(tarefas)} tarefa(s):\n\n"
    for t in tarefas:
        emoji = "⏳" if t["estado"] == "pendente" else "✅" if t["estado"] == "concluida" else "📦"
        resultado += f"{emoji} [{t['id']}] {t['titulo']}\n"
        resultado += f"   Categoria: {t['categoria']} | Estado: {t['estado']}\n\n"

    return resultado

# Testar
print(listar_tarefas.invoke({"estado": "pendente"}))