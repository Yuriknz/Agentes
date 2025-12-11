# prompt_com_data.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI

load_dotenv()

def get_system_prompt() -> str:
    """
    Gera o system prompt com data/hora atual.
    IMPORTANTE: Chamar a cada invocação para garantir data atualizada.
    """
    agora = datetime.now()

    # Tradução dos dias da semana
    dias_semana = {
        'Monday': 'segunda-feira',
        'Tuesday': 'terça-feira',
        'Wednesday': 'quarta-feira',
        'Thursday': 'quinta-feira',
        'Friday': 'sexta-feira',
        'Saturday': 'sábado',
        'Sunday': 'domingo'
    }

    dia_semana = dias_semana.get(agora.strftime('%A'), agora.strftime('%A'))
    data_formatada = agora.strftime('%d/%m/%Y')
    hora_formatada = agora.strftime('%H:%M')

    return f"""Você é um assistente pessoal inteligente.

## Informações Temporais
- Data atual: {data_formatada} ({dia_semana})
- Hora atual: {hora_formatada}

## Instruções
- Responda sempre em português brasileiro
- Use as informações temporais quando relevante
- Seja cordial e prestativo
"""

def main():
    modelo = ChatGoogleGenerativeAI(model=os.getenv("google_model", "gemini-2.5-flash"))

    mensagens = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content="Que dia é hoje? E que horas são?")
    ]

    resposta = modelo.invoke(mensagens)
    print(resposta.content)

if __name__ == "__main__":
    main()