# conversa_estruturada.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

modelo = ChatGoogleGenerativeAI(model=os.getenv("google_model", "gemini-2.5-flash-lite"))

# Histórico da conversa
historico = [
    SystemMessage(content="Você é um professor de história. Responda em português."),
    HumanMessage(content="Quem descobriu o Brasil?"),
    AIMessage(content="O Brasil foi oficialmente descoberto por Pedro Álvares Cabral em 22 de abril de 1500."),
    HumanMessage(content="E em que cidade ele desembarcou?"),
]

# O modelo tem acesso a todo o histórico
resposta = modelo.invoke(historico)
print(resposta.content)
# Saída: Pedro Álvares Cabral desembarcou na região que hoje é Porto Seguro, na Bahia.