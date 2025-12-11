import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar instância do modelo
modelo = ChatGoogleGenerativeAI(
    model=os.getenv("google_model", "gemini-2.5-flash-lite"),
    temperature=0,  # 0 = determinístico, 1 = criativo
    max_output_tokens=1024,
    max_retries=3,
    timeout=30
)

# Criar uma mensagem do usuário
mensagem = HumanMessage(content="Olá! Qual é a capital do Brasil?")

# Invocar o modelo
resposta = modelo.invoke([mensagem])

# Exibir a resposta
print(f"Resposta do modelo: {resposta.content}")