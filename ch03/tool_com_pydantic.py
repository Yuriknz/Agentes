# /src/ch03/tool_com_pydantic.py
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class CriarTarefaInput(BaseModel):
    """Schema para criação de tarefas."""

    titulo: str = Field(description="Título da tarefa")
    descricao: Optional[str] = Field(default=None, description="Descrição detalhada")
    data_vencimento: Optional[date] = Field(
        default=None, description="Data de vencimento (YYYY-MM-DD)"
    )

@tool(args_schema=CriarTarefaInput)
def criar_tarefa(
    titulo: str,
    descricao: Optional[str] = None,
    data_vencimento: Optional[date] = None,
) -> str:
    """Cria uma nova tarefa no sistema.
    Use esta ferramenta quando o usuário quiser adicionar uma nova tarefa,
    atividade ou lembrete.
    """
    # Simulação - em produção, salvaria no banco de dados
    tarefa_id = 123
    resultado = f"Tarefa criada com sucesso!\n"
    resultado += f"- ID: {tarefa_id}\n"
    resultado += f"- Título: {titulo}\n"
    if descricao:
        resultado += f"- Descrição: {descricao}\n"
    if data_vencimento:
        resultado += f"- Vencimento: {data_vencimento.strftime('%d/%m/%Y')}\n"
    return resultado

if __name__ == "__main__":
    # Testar a tool diretamente
    print(
        criar_tarefa.invoke(
            {
                "titulo": "Estudar LangChain",
                "descricao": "Completar tutorial",
                "data_vencimento": "2025-12-15",
            }
        )
    )

    # Exemplo adicional: inspeção da tool
    print(f"\n--- Informações da Tool ---")
    print(f"Nome: {criar_tarefa.name}")
    print(f"Descrição: {criar_tarefa.description}")
    print(f"Schema: {criar_tarefa.args}")