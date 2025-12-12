# /src/ch03/tool_simples.py
from langchain.tools import tool

@tool
def somar(a: int, b: int) -> int:
    """Soma dois números inteiros.
    Args:
        a: Primeiro número
        b: Segundo número
    Returns:
        A soma dos dois números
    """
    return a + b


if __name__ == "__main__":
    # Inspecionar a tool
    print(f"Nome: {somar.name}")
    print(f"Descrição: {somar.description}")
    print(f"Schema: {somar.args}")

    # Exemplo de uso direto
    resultado = somar.invoke({"a": 5, "b": 3})
    print(f"Resultado: {resultado}")