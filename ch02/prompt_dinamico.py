# prompt_dinamico.py
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

# Template com variáveis
template = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente pessoal.
Data atual: {data_atual}
Nome do usuário: {nome_usuario}
Responda sempre de forma personalizada."""),
    ("human", "{pergunta}")
])

# Preencher as variáveis
mensagens = template.invoke({
    "data_atual": datetime.now().strftime("%d/%m/%Y"),
    "nome_usuario": "Maria",
    "pergunta": "Que dia é hoje?"
})

print(mensagens)
# Saída: lista de mensagens com as variáveis substituídas