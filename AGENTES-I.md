# Tutorial Completo de LangChain e LangGraph v1.0+

**Desenvolvimento de Agentes Inteligentes com Python**

---

## Sumário

### Parte I: Fundamentos
1. [Introdução ao Ecossistema LangChain](#capítulo-1-introdução-ao-ecossistema-langchain)
2. [Mensagens e Prompts](#capítulo-2-mensagens-e-prompts)
3. [Tools (Ferramentas)](#capítulo-3-tools-ferramentas)

### Parte II: LCEL e Composição Declarativa
4. [LangChain Expression Language (LCEL)](#capítulo-4-langchain-expression-language-lcel)
5. [Modularidade e Interoperabilidade](#capítulo-5-modularidade-e-interoperabilidade)

---

# PARTE I: FUNDAMENTOS

---

## Capítulo 1: Introdução ao Ecossistema LangChain

### 1.1 O Que São LLMs e Por Que Precisamos de Frameworks?

**Large Language Models (LLMs)** são modelos de inteligência artificial treinados em vastas quantidades de texto para compreender e gerar linguagem natural. Exemplos incluem GPT-4, Claude, Llama e Gemini.

Embora as APIs desses modelos sejam poderosas, construir aplicações robustas diretamente sobre elas apresenta desafios:

- **Gerenciamento de conversas**: Manter histórico, contexto e estado
- **Integração com ferramentas**: Permitir que o modelo execute ações no mundo real
- **Tratamento de erros**: Lidar com falhas, timeouts e respostas inesperadas
- **Orquestração complexa**: Coordenar múltiplas chamadas e decisões

É aqui que frameworks como **LangChain** e **LangGraph** entram em cena.

### 1.2 LangChain vs LangGraph: Qual a Diferença?

| Aspecto | LangChain | LangGraph |
|---------|-----------|-----------|
| **Foco** | Componentes e primitivos | Orquestração e fluxo |
| **Abstração** | Mensagens, prompts, tools | Grafos de estado, nós, arestas |
| **Uso** | Blocos de construção | Coordenação de agentes |
| **Analogia** | "Peças de Lego" | "Manual de montagem" |

**LangChain v1.0+** fornece os componentes fundamentais:
- `ChatOpenAI`, `ChatAnthropic` - Interfaces para LLMs
- `SystemMessage`, `HumanMessage`, `AIMessage` - Tipos de mensagens
- `@tool` - Decorator para criar ferramentas
- Schemas Pydantic para validação

**LangGraph v1.0+** fornece a orquestração:
- `StateGraph` - Grafos de estado para fluxos complexos
- Checkpointers - Persistência e recuperação de estado
- Streaming - Respostas em tempo real
- Human-in-the-Loop - Intervenção humana

**Resumo**: Use LangChain para os **componentes** e LangGraph para **coordená-los**.

### 1.3 Configuração do Ambiente

#### Pré-requisitos
- Python 3.10 ou superior
- Uma chave de API da OpenAI (ou outro provedor)

#### Criando o Projeto

```bash
# Criar diretório do projeto
mkdir meu_agente
cd meu_agente

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências com pip
pip install langchain>=1.0.0 langchain-openai>=0.3.0 langgraph>=1.0.0 python-dotenv
```

**Alternativa com uv (mais rápido)**:
```bash
# Instalar uv usando pip (se ainda não tiver)
pip install uv
# Instalar uv standalone com curl
curl -LsSf https://astral.sh/uv/install.sh | sh
# Instalar uv standalone com wget
wget -qO- https://astral.sh/uv/install.sh | sh
# Criar projeto e instalar dependências
uv init meu_agente
cd meu_agente
uv add langchain langchain-openai langgraph python-dotenv
```

#### Configurando Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# .env
OPENAI_API_KEY=sk-sua-chave-aqui
OPENAI_MODEL=gpt-4o-mini
```

> **Importante**: Nunca commite o arquivo `.env` no Git! Adicione-o ao `.gitignore`.

### 1.4 Primeira Chamada a um LLM

Vamos criar nosso primeiro programa que se comunica com um LLM.

```python
# hello_llm.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar instância do modelo
modelo = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0  # 0 = determinístico, 1 = criativo
)

# Criar uma mensagem do usuário
mensagem = HumanMessage(content="Olá! Qual é a capital do Brasil?")

# Invocar o modelo
resposta = modelo.invoke([mensagem])

# Exibir a resposta
print(f"Resposta do modelo: {resposta.content}")
```

**Executando**:
```bash
python hello_llm.py
# Saída: Resposta do modelo: A capital do Brasil é Brasília.
```

### 1.5 Entendendo o Código

Vamos analisar cada parte:

#### 1. Importações
```python
from langchain_openai import ChatOpenAI          # Interface para OpenAI
from langchain_core.messages import HumanMessage # Tipo de mensagem
```

O pacote `langchain_openai` é separado do `langchain` principal. Isso segue o padrão modular do LangChain v1.0+, onde cada provedor tem seu próprio pacote.

#### 2. Criação do Modelo
```python
modelo = ChatOpenAI(
    model="gpt-4o-mini",  # Modelo a usar
    temperature=0          # Controle de aleatoriedade
)
```

Parâmetros importantes:
- `model`: Qual modelo usar (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- `temperature`: 0.0 (determinístico) a 1.0 (criativo)
- `max_tokens`: Limite de tokens na resposta
- `timeout`: Tempo máximo de espera

#### 3. Mensagens
```python
mensagem = HumanMessage(content="Olá!")
resposta = modelo.invoke([mensagem])
```

O modelo recebe uma **lista de mensagens**. Isso é fundamental para manter conversas, como veremos no próximo capítulo.

### 1.6 Exemplo Completo: Assistente Simples

Vamos criar um assistente que responde perguntas em um loop:

```python
# assistente_simples.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def criar_assistente():
    """Cria e retorna uma instância do modelo."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7
    )

def conversar(modelo, pergunta: str) -> str:
    """Envia uma pergunta ao modelo e retorna a resposta."""
    mensagens = [
        SystemMessage(content="Você é um assistente prestativo que responde em português."),
        HumanMessage(content=pergunta)
    ]
    resposta = modelo.invoke(mensagens)
    return resposta.content

def main():
    print("=== Assistente Simples ===")
    print("Digite 'sair' para encerrar.\n")

    modelo = criar_assistente()

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == 'sair':
            print("Até logo!")
            break

        if not pergunta:
            continue

        resposta = conversar(modelo, pergunta)
        print(f"Assistente: {resposta}\n")

if __name__ == "__main__":
    main()
```

**Executando**:
```bash
python assistente_simples.py
```

```
=== Assistente Simples ===
Digite 'sair' para encerrar.

Você: Qual é o maior planeta do sistema solar?
Assistente: O maior planeta do sistema solar é Júpiter.

Você: E qual é o menor?
Assistente: O menor planeta do sistema solar é Mercúrio.

Você: sair
Até logo!
```

> **Nota**: Este assistente ainda não tem memória - cada pergunta é independente. No próximo capítulo, aprenderemos como manter o contexto da conversa.

### 1.7 Resumo do Capítulo

Neste capítulo, você aprendeu:

- **LLMs** são modelos de linguagem que compreendem e geram texto
- **LangChain** fornece componentes (mensagens, modelos, tools)
- **LangGraph** fornece orquestração (grafos, estado, persistência)
- Como **configurar o ambiente** com Python e variáveis de ambiente
- Como fazer sua **primeira chamada** a um LLM
- A estrutura básica de **mensagens** (HumanMessage, SystemMessage)

### 1.8 Exercícios

1. **Modifique o assistente** para usar `temperature=0` e depois `temperature=1`. Observe as diferenças nas respostas.

2. **Crie um tradutor** que receba texto em português e traduza para inglês.

3. **Experimente diferentes modelos**: Troque `gpt-4o-mini` por `gpt-4o` e compare a qualidade das respostas.

---

## Capítulo 2: Mensagens e Prompts

### 2.1 Tipos de Mensagens no LangChain

No LangChain v1.0+, a comunicação com LLMs é baseada em **mensagens tipadas**. Cada tipo de mensagem tem um papel específico na conversa:

```python
from langchain_core.messages import (
    SystemMessage,   # Instruções do sistema
    HumanMessage,    # Mensagens do usuário
    AIMessage,       # Respostas do assistente
    ToolMessage,     # Resultados de ferramentas
)
```

#### SystemMessage - Definindo o Comportamento

A `SystemMessage` define **quem o assistente é** e **como ele deve se comportar**. É sempre a primeira mensagem da conversa.

```python
from langchain_core.messages import SystemMessage

system = SystemMessage(content="""
Você é um assistente especializado em programação Python.
Responda sempre em português brasileiro.
Seja conciso e forneça exemplos de código quando apropriado.
""")
```

#### HumanMessage - Entrada do Usuário

A `HumanMessage` representa as mensagens enviadas pelo usuário:

```python
from langchain_core.messages import HumanMessage

pergunta = HumanMessage(content="Como criar uma lista em Python?")
```

#### AIMessage - Resposta do Assistente

A `AIMessage` representa as respostas geradas pelo modelo. Você a recebe como retorno do `invoke()`:

```python
resposta = modelo.invoke([system, pergunta])
# resposta é uma AIMessage
print(type(resposta))  # <class 'langchain_core.messages.ai.AIMessage'>
print(resposta.content)  # "Para criar uma lista em Python..."
```

#### ToolMessage - Resultado de Ferramentas

A `ToolMessage` carrega o resultado da execução de uma ferramenta. Veremos isso em detalhes no Capítulo 3.

```python
from langchain_core.messages import ToolMessage

resultado = ToolMessage(
    content="A tarefa foi criada com sucesso.",
    tool_call_id="call_abc123"  # ID da chamada da ferramenta
)
```

### 2.2 Estrutura de uma Conversa

Uma conversa é uma **lista de mensagens** que cresce ao longo do tempo:

```python
# conversa_estruturada.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

modelo = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

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
```

> **Importante**: O modelo não tem memória interna. Você precisa enviar **todo o histórico** a cada chamada para manter o contexto.

### 2.3 Chatbot com Memória Manual

Vamos criar um chatbot que mantém o histórico da conversa:

```python
# chatbot_com_memoria.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class Chatbot:
    def __init__(self, instrucoes: str):
        self.modelo = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7
        )
        self.historico = [SystemMessage(content=instrucoes)]

    def conversar(self, mensagem: str) -> str:
        # Adicionar mensagem do usuário ao histórico
        self.historico.append(HumanMessage(content=mensagem))

        # Obter resposta do modelo
        resposta = self.modelo.invoke(self.historico)

        # Adicionar resposta ao histórico
        self.historico.append(resposta)

        return resposta.content

    def limpar_historico(self):
        # Mantém apenas a SystemMessage
        self.historico = [self.historico[0]]

def main():
    bot = Chatbot(
        instrucoes="Você é um assistente amigável. Responda em português de forma concisa."
    )

    print("=== Chatbot com Memória ===")
    print("Comandos: 'sair' para encerrar, 'limpar' para reiniciar conversa\n")

    while True:
        entrada = input("Você: ").strip()

        if entrada.lower() == 'sair':
            print("Até logo!")
            break
        elif entrada.lower() == 'limpar':
            bot.limpar_historico()
            print("Histórico limpo!\n")
            continue
        elif not entrada:
            continue

        resposta = bot.conversar(entrada)
        print(f"Bot: {resposta}\n")

if __name__ == "__main__":
    main()
```

**Testando a memória**:
```
Você: Meu nome é João
Bot: Olá, João! Prazer em conhecê-lo. Como posso ajudá-lo hoje?

Você: Qual é o meu nome?
Bot: Seu nome é João, conforme você me disse agora há pouco.
```

### 2.4 Templates de Prompt Dinâmicos

Muitas vezes precisamos criar prompts com variáveis dinâmicas. O LangChain oferece templates para isso:

```python
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
```

### 2.5 Injeção de Contexto: Data e Hora Atual

Um padrão comum é injetar informações dinâmicas no system prompt. Veja como o projeto JarvisChat faz isso:

```python
# prompt_com_data.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
    modelo = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    mensagens = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content="Que dia é hoje? E que horas são?")
    ]

    resposta = modelo.invoke(mensagens)
    print(resposta.content)

if __name__ == "__main__":
    main()
```

**Saída exemplo**:
```
Hoje é 10/12/2025, uma terça-feira, e são aproximadamente 15:30.
```

### 2.6 Boas Práticas para System Prompts

Um bom system prompt deve ser:

1. **Claro e específico**: Defina exatamente o papel do assistente
2. **Estruturado**: Use seções e formatação
3. **Contextualizado**: Inclua informações relevantes (data, usuário, etc.)
4. **Com exemplos**: Mostre o formato esperado das respostas

**Exemplo de system prompt bem estruturado**:

```python
SYSTEM_PROMPT = """
# Papel
Você é um assistente de gerenciamento de tarefas.

# Capacidades
- Criar, listar, atualizar e excluir tarefas
- Organizar tarefas por categorias
- Definir datas de vencimento

# Regras
1. Sempre confirme ações destrutivas (exclusão)
2. Use formato de data brasileiro (DD/MM/AAAA)
3. Seja conciso nas respostas

# Formato de Resposta
- Para listagem: use bullets (-)
- Para confirmações: use ✓ ou ✗
- Para datas: sempre em português

# Contexto
Data atual: {data_atual}
Usuário: {nome_usuario}
"""
```

### 2.7 Exemplo Completo: Assistente com Contexto

```python
# assistente_contextualizado.py
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class AssistenteContextualizado:
    def __init__(self, nome_usuario: str):
        self.modelo = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7
        )
        self.nome_usuario = nome_usuario
        self.historico = []

    def _get_system_prompt(self) -> str:
        agora = datetime.now()
        return f"""Você é um assistente pessoal chamado Jarvis.

## Contexto
- Usuário: {self.nome_usuario}
- Data: {agora.strftime('%d/%m/%Y')}
- Hora: {agora.strftime('%H:%M')}

## Personalidade
- Seja cordial e use o nome do usuário ocasionalmente
- Responda em português brasileiro
- Seja conciso, mas completo
"""

    def conversar(self, mensagem: str) -> str:
        # System prompt atualizado a cada chamada (data/hora atual)
        system = SystemMessage(content=self._get_system_prompt())

        # Montar mensagens: system + histórico + nova mensagem
        mensagens = [system] + self.historico + [HumanMessage(content=mensagem)]

        # Obter resposta
        resposta = self.modelo.invoke(mensagens)

        # Atualizar histórico (sem o system, que é recriado)
        self.historico.append(HumanMessage(content=mensagem))
        self.historico.append(resposta)

        return resposta.content

def main():
    nome = input("Qual é o seu nome? ").strip() or "Usuário"
    assistente = AssistenteContextualizado(nome_usuario=nome)

    print(f"\nOlá, {nome}! Sou o Jarvis, seu assistente pessoal.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            print(f"Até logo, {nome}!")
            break
        if entrada:
            resposta = assistente.conversar(entrada)
            print(f"Jarvis: {resposta}\n")

if __name__ == "__main__":
    main()
```

### 2.8 Resumo do Capítulo

Neste capítulo, você aprendeu:

- Os **4 tipos de mensagens**: SystemMessage, HumanMessage, AIMessage, ToolMessage
- Como **estruturar conversas** com listas de mensagens
- A implementar **memória manual** mantendo o histórico
- A criar **templates dinâmicos** com variáveis
- Como **injetar contexto** (data, hora, usuário) no prompt
- **Boas práticas** para system prompts eficazes

### 2.9 Exercícios

1. **Crie um chatbot temático**: Um assistente especializado em receitas culinárias que pergunta sobre ingredientes disponíveis.

2. **Limite de contexto**: Modifique o chatbot para manter apenas as últimas 10 mensagens no histórico (evitando estourar o limite de tokens).

3. **Prompt multilíngue**: Crie um assistente que detecta o idioma da pergunta e responde no mesmo idioma.

---

## Capítulo 3: Tools (Ferramentas)

### 3.1 O Que São Tools e Por Que São Importantes?

**Tools** (ferramentas) são funções que o modelo pode **decidir chamar** para realizar ações no mundo real. Sem tools, o modelo é apenas um gerador de texto. Com tools, ele se torna um **agente** capaz de:

- Buscar informações em bancos de dados
- Fazer cálculos matemáticos
- Criar, atualizar e excluir dados
- Interagir com APIs externas
- Executar código

#### Fluxo de Execução com Tools

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Usuário   │ ──▶  │    LLM      │ ──▶  │    Tool     │
│  "Calcule   │      │  (decide    │      │ (executa    │
│   2 + 2"    │      │  usar tool) │      │  cálculo)   │
└─────────────┘      └─────────────┘      └─────────────┘
                            │                    │
                            ▼                    ▼
                     ┌─────────────┐      ┌─────────────┐
                     │    LLM      │ ◀──  │  Resultado  │
                     │  (formata   │      │    "4"      │
                     │  resposta)  │      └─────────────┘
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  "2 + 2 = 4"│
                     └─────────────┘
```

### 3.2 Criando Tools com o Decorator @tool

O LangChain v1.0+ usa o decorator `@tool` para criar ferramentas:

```python
# tool_simples.py
from langchain_core.tools import tool

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

# Inspecionar a tool
print(f"Nome: {somar.name}")
print(f"Descrição: {somar.description}")
print(f"Schema: {somar.args_schema.schema()}")
```

**Saída**:
```
Nome: somar
Descrição: Soma dois números inteiros.
Schema: {'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}, 'required': ['a', 'b'], 'type': 'object'}
```

> **Importante**: A **docstring** é fundamental! O LLM usa a descrição para decidir quando usar a tool.

### 3.3 Tools com Schemas Pydantic

Para tools mais complexas, use modelos Pydantic para validação:

```python
# tool_com_pydantic.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class CriarTarefaInput(BaseModel):
    """Schema para criação de tarefas."""
    titulo: str = Field(description="Título da tarefa")
    descricao: Optional[str] = Field(default=None, description="Descrição detalhada")
    data_vencimento: Optional[date] = Field(default=None, description="Data de vencimento (YYYY-MM-DD)")

@tool(args_schema=CriarTarefaInput)
def criar_tarefa(titulo: str, descricao: Optional[str] = None, data_vencimento: Optional[date] = None) -> str:
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

# Testar a tool diretamente
print(criar_tarefa.invoke({
    "titulo": "Estudar LangChain",
    "descricao": "Completar tutorial",
    "data_vencimento": "2025-12-15"
}))
```

### 3.4 Binding Tools ao Modelo

Para que o modelo possa usar as tools, precisamos "bindá-las":

```python
# binding_tools.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# Definir tools
@tool
def calcular(expressao: str) -> str:
    """Calcula uma expressão matemática simples.

    Args:
        expressao: Expressão matemática (ex: "2 + 2", "10 * 5")
    """
    try:
        # ATENÇÃO: eval() é perigoso em produção!
        # Use uma biblioteca segura como numexpr
        resultado = eval(expressao)
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Erro no cálculo: {e}"

@tool
def obter_clima(cidade: str) -> str:
    """Obtém informações sobre o clima de uma cidade.

    Args:
        cidade: Nome da cidade
    """
    # Simulação - em produção, chamaria uma API real
    climas = {
        "são paulo": "Nublado, 22°C",
        "rio de janeiro": "Ensolarado, 32°C",
        "curitiba": "Chuvoso, 15°C",
    }
    return climas.get(cidade.lower(), f"Clima não disponível para {cidade}")

# Criar modelo COM tools bindadas
modelo = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
modelo_com_tools = modelo.bind_tools([calcular, obter_clima])

# Testar - o modelo decide qual tool usar
resposta = modelo_com_tools.invoke([
    HumanMessage(content="Quanto é 15 vezes 8?")
])

print(f"Conteúdo: {resposta.content}")
print(f"Tool calls: {resposta.tool_calls}")
```

**Saída**:
```
Conteúdo:
Tool calls: [{'name': 'calcular', 'args': {'expressao': '15 * 8'}, 'id': 'call_abc123', 'type': 'tool_call'}]
```

> **Observe**: Quando o modelo decide usar uma tool, o `content` fica vazio e os argumentos vão em `tool_calls`.

### 3.5 Executando Tools e Retornando Resultados

O fluxo completo envolve:
1. Enviar mensagem ao modelo
2. Verificar se há tool_calls
3. Executar as tools
4. Enviar resultados de volta ao modelo
5. Obter resposta final

```python
# executar_tools.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def calcular(expressao: str) -> str:
    """Calcula uma expressão matemática."""
    try:
        resultado = eval(expressao)
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
modelo = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
modelo_com_tools = modelo.bind_tools(tools)

def processar_com_tools(mensagem: str) -> str:
    """Processa uma mensagem, executando tools se necessário."""
    mensagens = [HumanMessage(content=mensagem)]

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
            mensagens.append(ToolMessage(
                content=resultado,
                tool_call_id=tool_call_id
            ))

        # Nova chamada ao modelo com os resultados
        resposta = modelo_com_tools.invoke(mensagens)
        mensagens.append(resposta)

    return resposta.content

# Testar
print(processar_com_tools("Quanto é 25 ao quadrado?"))
print("\n---\n")
print(processar_com_tools("Como está o clima em São Paulo?"))
```

**Saída**:
```
Executando tool: calcular({'expressao': '25 ** 2'})
25 ao quadrado é igual a 625.

---

Executando tool: obter_clima({'cidade': 'São Paulo'})
O clima em São Paulo está em torno de 22°C, com céu nublado.
```

### 3.6 Tool com Múltiplos Parâmetros

Vamos criar uma tool mais complexa para gerenciar tarefas:

```python
# tool_tarefas.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date
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
```

### 3.7 Boas Práticas para Tools

1. **Docstrings descritivas**: O LLM usa a descrição para decidir quando usar a tool
2. **Nomes claros**: Use verbos no infinitivo (criar, listar, atualizar, excluir)
3. **Validação com Pydantic**: Garante que os argumentos estão corretos
4. **Tratamento de erros**: Retorne mensagens de erro úteis, não exceções
5. **Retorno informativo**: Confirme o que foi feito, não apenas "sucesso"

```python
# Exemplo de tool bem documentada
@tool
def criar_tarefa(titulo: str, data_vencimento: Optional[str] = None) -> str:
    """Cria uma nova tarefa no sistema de gerenciamento.

    Use esta ferramenta quando o usuário quiser:
    - Adicionar uma nova tarefa
    - Criar um lembrete
    - Agendar uma atividade

    Args:
        titulo: Título descritivo da tarefa (obrigatório)
        data_vencimento: Data limite no formato YYYY-MM-DD (opcional)

    Returns:
        Confirmação com detalhes da tarefa criada

    Exemplos de uso:
        - "Crie uma tarefa para estudar Python"
        - "Adicione lembrete: reunião dia 15/12"
    """
    # Implementação...
```

### 3.8 Exemplo Completo: Assistente com Tools

```python
# assistente_com_tools.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

# === TOOLS ===

@tool
def somar(a: float, b: float) -> str:
    """Soma dois números."""
    return str(a + b)

@tool
def subtrair(a: float, b: float) -> str:
    """Subtrai b de a."""
    return str(a - b)

@tool
def multiplicar(a: float, b: float) -> str:
    """Multiplica dois números."""
    return str(a * b)

@tool
def dividir(a: float, b: float) -> str:
    """Divide a por b."""
    if b == 0:
        return "Erro: divisão por zero não é permitida"
    return str(a / b)

# === ASSISTENTE ===

class AssistenteCalculadora:
    def __init__(self):
        self.tools = [somar, subtrair, multiplicar, dividir]
        self.tools_por_nome = {t.name: t for t in self.tools}

        modelo = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0
        )
        self.modelo = modelo.bind_tools(self.tools)

        self.system = SystemMessage(content="""
Você é uma calculadora inteligente.
Use as ferramentas disponíveis para fazer cálculos.
Sempre mostre o resultado de forma clara.
""")

    def processar(self, pergunta: str) -> str:
        mensagens = [self.system, HumanMessage(content=pergunta)]

        while True:
            resposta = self.modelo.invoke(mensagens)
            mensagens.append(resposta)

            # Se não há tool_calls, retornar resposta final
            if not resposta.tool_calls:
                return resposta.content

            # Executar cada tool
            for tool_call in resposta.tool_calls:
                nome = tool_call["name"]
                args = tool_call["args"]

                resultado = self.tools_por_nome[nome].invoke(args)

                mensagens.append(ToolMessage(
                    content=resultado,
                    tool_call_id=tool_call["id"]
                ))

def main():
    calc = AssistenteCalculadora()

    print("=== Calculadora Inteligente ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
        if entrada:
            resposta = calc.processar(entrada)
            print(f"Calculadora: {resposta}\n")

if __name__ == "__main__":
    main()
```

**Testando**:
```
Você: Quanto é 15 vezes 8?
Calculadora: 15 vezes 8 é igual a 120.

Você: Agora divida o resultado por 3
Calculadora: 120 dividido por 3 é igual a 40.

Você: Some 100 e 200, depois multiplique por 2
Calculadora: (100 + 200) × 2 = 600
```

### 3.9 Resumo do Capítulo

Neste capítulo, você aprendeu:

- O que são **tools** e por que são essenciais para agentes
- Como criar tools com o **decorator @tool**
- Como usar **Pydantic** para validação de argumentos
- Como fazer **bind_tools()** ao modelo
- O fluxo completo de **execução de tools**
- **Boas práticas** para documentação de tools

### 3.10 Exercícios

1. **Crie uma tool de conversão**: Converta temperaturas entre Celsius e Fahrenheit.

2. **Tool de busca**: Crie uma tool que simula busca em uma lista de produtos.

3. **Múltiplas tools**: Crie um assistente com tools para CRUD completo de uma entidade.

---

# PARTE II: LCEL E COMPOSIÇÃO DECLARATIVA

---

## Capítulo 4: LangChain Expression Language (LCEL)

### 4.1 O Que É LCEL?

Até agora, construímos pipelines "manualmente": criamos prompts, passamos para modelos, parseamos outputs. À medida que sistemas crescem, esse padrão se torna repetitivo e frágil.

**LangChain Expression Language (LCEL)** é a resposta: uma linguagem declarativa para compor componentes LangChain usando o operador **pipe** (`|`).

> **LCEL** permite construir pipelines reutilizáveis, testáveis e serializáveis sem "glue code" manual.

### 4.2 Composição com Pipes

A ideia é simples: componentes podem ser encadeados com `|`, criando uma pipeline declarativa:

```python
# Antes: manual
prompt = ChatPromptTemplate.from_messages([...])
model = ChatOpenAI(...)
parser = StrOutputParser()

messages = prompt.invoke({"topic": "Python"})
response = model.invoke(messages)
resposta_final = parser.invoke(response)

# Depois: LCEL com pipes
chain = prompt | model | parser
resposta_final = chain.invoke({"topic": "Python"})
```

O operador `|` é sintaticamente açúcar, mas semanticamente poderoso: transforma funções em objetos compostos que sabem como:
- Invocar (`.invoke()`)
- Fazer streaming (`.stream()`)
- Executar em batch (`.batch()`)

### 4.3 Conceitos Essenciais de RAG (Retrieval Augmented Generation)

Antes de explorar exemplos práticos de LCEL, precisamos entender alguns conceitos fundamentais que aparecem nos próximos exemplos.

#### 4.3.1 O que é RAG?

**RAG (Retrieval Augmented Generation)** é uma arquitetura que combina dois componentes:
1. **Retrieval** (Busca) - encontrar documentos relevantes
2. **Generation** (Geração) - usar um LLM para criar respostas baseadas nos documentos

**Por que RAG?**
- LLMs treinados têm conhecimento limitado e podem ficar desatualizados
- RAG permite adicionar conhecimento externo (seus documentos, bases de dados, etc.)
- Respostas são baseadas em fontes confiáveis e atualizadas
- Reduz alucinações (respostas inventadas) do modelo

**Fluxo básico de uma arquitetura RAG**:
```
Pergunta do Usuário
    ↓
Buscar Documentos Relevantes (Retriever)
    ↓
Passar Documentos + Pergunta para o LLM
    ↓
LLM Gera Resposta com Base no Contexto
    ↓
Resposta Fundamentada
```

#### 4.3.2 Embeddings - Representações Vetoriais

**O que são?**
- Vetores numéricos que representam o significado de um texto
- Cada palavra ou documento é convertido em uma lista de números
- Textos com significado similar têm vetores próximos no espaço vetorial

**Exemplo Visual**:
```
Texto: "cachorro"      → Vetor: [0.2, 0.8, 0.1, 0.5, ...]
Texto: "cão"           → Vetor: [0.19, 0.82, 0.09, 0.51, ...]  ← Muito Similar!
Texto: "planeta Terra" → Vetor: [0.91, 0.05, 0.87, 0.2, ...]   ← Diferente
```

**Como funcionam?**
- Modelos treinados (OpenAI, Cohere, HuggingFace, etc.) convertam texto em vetores
- Esses vetores codificam informações semânticas
- A similaridade entre vetores pode ser medida (ex: produto escalar, cosseno)

**Por que é importante?**
- Permite buscar documentos por significado, não por palavras-chave
- Base do funcionamento de vector stores

#### 4.3.3 Vector Stores - Bancos de Dados de Vetores

**O que são?**
- Bancos de dados especializados em armazenar e buscar vetores
- Otimizados para encontrar vetores similares rapidamente
- Exemplos populares: FAISS, Chroma, Pinecone, Qdrant, Weaviate

**Operações principais**:
1. **Adicionar documentos**:
   - Documento original → Embedding (vetor) → Armazenar no banco

2. **Buscar documentos**:
   - Query do usuário → Embedding (vetor) → Buscar K vetores mais similares → Retornar documentos originais

**Exemplo Visual**:
```
Vector Store
├─ "Manual Python" → [0.1, 0.9, 0.2, ...]
├─ "Guia JavaScript" → [0.2, 0.8, 0.3, ...]
├─ "Tutorial LangChain" → [0.15, 0.85, 0.25, ...]
└─ "Receita de Bolo" → [0.8, 0.1, 0.9, ...]

Busca: "Como usar Python?" → Embedding → [0.11, 0.88, 0.21, ...]
Resultado: "Manual Python" (mais similar), "Tutorial LangChain" (2º mais similar)
```

#### 4.3.4 Retriever - Interface de Busca Simplificada

**O que é?**
- Uma abstração que encapsula a busca em vector stores
- Interface padrão para recuperar documentos relevantes
- Criado a partir de um vector store: `vectorstore.as_retriever()`

**Parâmetros comuns**:
- `k`: Número de documentos a retornar (exemplo: `k=2` retorna top 2 mais similares)
- `search_type`: Tipo de busca (default: "similarity")

**Como usar**:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("minha pergunta")  # Retorna lista com 2 documentos
```

> **📚 Nota Importante**: Este é um resumo introdutório de RAG. Os Capítulos 12, 13 e 14 exploram RAG em profundidade, incluindo técnicas avançadas como chunking, embedding models diferentes, prompt optimization, e retrieval strategies.

### 4.4 Runnables em LCEL - Blocos de Construção

LCEL usa o conceito de **Runnable** como bloco fundamental. Entender Runnables é essencial para compor pipelines.

#### 4.4.1 O que é um Runnable?

**Definição**:
- Interface padrão do LangChain para qualquer componente que pode ser executado
- Permite composição via operador `|`
- Suporta operações padrão: `invoke()`, `stream()`, `batch()`

**Exemplos de objetos que são Runnables**:
- Prompts (`ChatPromptTemplate`)
- LLMs (`ChatOpenAI`, `Anthropic`)
- Parsers (`StrOutputParser`)
- Retrievers (criados de vector stores)
- Funções Python customizadas (via `RunnableLambda`)
- Dicionários (automaticamente convertidos em `RunnableParallel`)

**Por que Runnables?**
- Interface consistente para todos os componentes
- Permite encadear qualquer coisa com `|`
- Automaticamente suporta async, streaming, batch

#### 4.4.2 RunnablePassthrough

**O que faz?**
- Passa dados **inalterados** através da cadeia
- Útil para preservar inputs originais em pipelines complexos
- Não faz nenhuma transformação

**Caso de Uso Principal - RAG**:
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)
```

**Explicação do Fluxo**:
- `retriever` busca documentos baseado no input
- `RunnablePassthrough()` mantém a pergunta original intacta
- Ambos são passados para o prompt como variáveis `{context}` e `{question}`

**Diagrama Visual**:
```
Input: "O que é Python?"
    │
    ├─→ retriever(input) → ["Python é uma linguagem..."] → context
    │
    └─→ RunnablePassthrough() → "O que é Python?" → question
         │
         ├─→ prompt.invoke({"context": "...", "question": "O que é Python?"})
         │
         ├─→ model.invoke(prompt_formatted)
         │
         └─→ parser.invoke(model_output) → resposta final
```

#### 4.4.3 RunnableLambda

**O que faz?**
- Transforma **qualquer função Python** em um Runnable
- Permite integrar lógica customizada em cadeias LCEL
- A função fica "native" à cadeia (suporta stream, batch, etc.)

**Exemplo de Uso**:
```python
def adicionar_timestamp(input):
    from datetime import datetime
    return f"[{datetime.now()}] {input}"

chain = (
    RunnableLambda(adicionar_timestamp)
    | prompt
    | model
)
```

**Quando usar RunnableLambda?**
- Transformar dados entre componentes
- Adicionar logs ou monitoramento
- Aplicar regras de negócio customizadas
- Integrar com APIs ou bancos de dados externos
- Fazer pré/pós-processamento de texto

**Comparação**:

| Sem RunnableLambda | Com RunnableLambda |
|-------------------|-------------------|
| Função separada, fora da cadeia | Integrado à cadeia |
| Precisa chamar manualmente | Executa automaticamente no fluxo |
| Não suporta `.stream()` automático | Suporta todas operações Runnable |
| Código mais verboso | Código mais limpo |

#### 4.4.4 Resumo - Padrões Comuns de Runnables

| Runnable | Entrada | Saída | Caso de Uso Principal |
|----------|---------|-------|----------------------|
| `RunnablePassthrough` | Qualquer input | Input inalterado | Passar dados através da cadeia |
| `RunnableLambda` | Qualquer input | Resultado da função | Transformações customizadas |
| Dicionário `{...}` | Input simples | Dict com chaves | Agrupar múltiplas operações (automaticamente `RunnableParallel`) |
| `RunnableParallel` | Input simples | Dict com resultados | Executar múltiplas cadeias em paralelo |

**Dica**: A sintaxe `{"key": runnable}` é automaticamente convertida em `RunnableParallel`, que executa múltiplos Runnables em paralelo e combina resultados.

### 4.5 Exemplo Prático: Pipeline RAG Simples

Vamos construir um pipeline RAG básico com LCEL:

```python
# rag_lcel_simples.py
"""
Exemplo: Pipeline RAG com LCEL

Este exemplo demonstra como combinar:
- Retriever: Busca documentos relevantes (vector store)
- Prompt: Formata pergunta + contexto
- LLM: Gera resposta baseada no contexto
- Parser: Extrai texto do LLM output

Conceitos-chave:
- RunnablePassthrough: passa a pergunta original
- RunnableParallel (via dict): executa retriever e passthrough em paralelo
- LCEL pipe operator: composição declarativa
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Preparar documentos
documentos = [
    "Python é uma linguagem de programação de alto nível.",
    "LangChain facilita a construção de aplicações com LLMs.",
    "RAG combina busca de documentos com geração de LLMs.",
]

# 2. Criar embeddings e vector store
# Embeddings: Convertidos documentos em vetores numéricos
embeddings = OpenAIEmbeddings()

# Vector Store: Armazena documentos com seus embeddings para busca por similaridade
vectorstore = FAISS.from_texts(documentos, embeddings)

# Retriever: Interface para buscar documentos relevantes
# search_kwargs={"k": 2} significa: retornar os 2 documentos mais similares
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Criar prompt template
# Este template tem duas variáveis:
# - {context}: documentos retornados pelo retriever
# - {question}: pergunta do usuário (passada por RunnablePassthrough)
prompt_template = """
Você é um assistente especializado. Use os documentos fornecidos para responder.

Documentos:
{context}

Pergunta: {question}
Resposta:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 4. Configurar modelo e parser
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()  # Extrai apenas o texto da resposta

# 5. Montar a pipeline LCEL
# Sintaxe explicada:
# {"context": retriever, "question": RunnablePassthrough()}
#   ↳ Executa retriever e RunnablePassthrough em PARALELO
#   ↳ retriever busca docs baseado no input
#   ↳ RunnablePassthrough() passa a pergunta original intacta
#   ↳ Resultado: {"context": [...docs...], "question": "...pergunta..."}
#
# | prompt ↳ Formata as variáveis no template
# | model  ↳ Envia para o LLM
# | parser ↳ Extrai texto da resposta
rag_chain = (
    {
        "context": retriever,              # Busca K docs mais similares
        "question": RunnablePassthrough()  # Mantém pergunta original
    }
    | prompt   # ChatPromptTemplate
    | model    # ChatOpenAI
    | parser   # StrOutputParser - retorna apenas texto
)

# 6. Usar a pipeline
pergunta = "O que é RAG?"
resposta = rag_chain.invoke(pergunta)
print(f"Resposta: {resposta}")

# 7. Streaming (um dos benefícios do LCEL)
# Suportado automaticamente porque todos os componentes são Runnables
print("\nStreaming:")
for chunk in rag_chain.stream(pergunta):
    print(chunk, end="", flush=True)
```

**O que acontece aqui?**

1. **Retriever**: Busca documentos relevantes (dict `{"context": ...}`)
2. **Prompt**: Formata a pergunta e contexto
3. **Model**: Gera resposta
4. **Parser**: Extrai texto da resposta

Tudo isso declarativamente, sem loops ou código intermediário.

### 4.6 Composição Avançada: LLMChain com Memory

LCEL também pode incluir lógica de memória usando **RunnableLambda**:

```python
from langchain_core.runnables import RunnableLambda

"""
Exemplo: Pipeline com Memória Simplificada

Este exemplo mostra como usar RunnableLambda para adicionar lógica customizada
(neste caso, manutenção de histórico) à cadeia LCEL.

RunnableLambda transforma uma função Python comum em um Runnable que:
- Integra-se naturalmente à cadeia
- Suporta streaming automaticamente
- Pode ser testado independentemente
"""

# Memória simplificada (em produção, use LangChain's memory classes)
mensagens = []

def adicionar_historico(entrada):
    """
    Função customizada que adiciona entrada ao histórico.

    Recebe: string (pergunta do usuário)
    Retorna: dict com histórico e entrada (para o prompt usar)

    Nota: RunnableLambda envolve esta função para integrá-la à cadeia
    """
    mensagens.append(entrada)
    # Retorna dicionário com as variáveis que o prompt precisa
    return {
        "historico": "\n".join(mensagens[-5:]),  # Últimas 5 mensagens
        "entrada": entrada                        # Pergunta atual
    }

# Pipeline com memória
# 1. RunnablePassthrough(): passa a pergunta original
# 2. RunnableLambda(adicionar_historico): função customizada que mantém histórico
#    - Recebe a pergunta
#    - Adiciona ao histórico
#    - Retorna {"historico": "...", "entrada": "..."}
# 3. prompt: template usa {historico} e {entrada}
# 4. model: gera resposta
# 5. parser: extrai texto
chat_chain = (
    RunnablePassthrough()                        # Passa pergunta intacta
    | RunnableLambda(adicionar_historico)        # Adiciona lógica customizada (memória)
    | prompt                                      # Formata com variáveis
    | model                                       # LLM gera resposta
    | parser                                      # Extrai texto
)

# Testando a pipeline com memória
resposta1 = chat_chain.invoke("Olá, qual é seu nome?")
print(f"Resposta 1: {resposta1}")

# Agora, adicionar_historico terá a pergunta anterior no histórico
resposta2 = chat_chain.invoke("O que você acabou de me contar?")
print(f"Resposta 2: {resposta2}")

# Observação: O histórico agora contém ambas as mensagens
print(f"Histórico completo: {mensagens}")
```

**Por que RunnableLambda é poderoso aqui?**
- Transforma uma função Python em um Runnable
- Integra-se perfeitamente com LCEL
- Suporta automaticamente `.stream()`, `.batch()`, operações assíncronas
- Sem precisar de classes ou código boilerplate

### 4.7 Por Que LCEL Não É Suficiente: Motivação para LangGraph

LCEL é excelente para **pipelines determinísticas lineares**, mas tem limitações importantes quando você precisa de lógica mais complexa:

| Característica | LCEL | LangGraph |
|---|---|---|
| **Pipes lineares** | ✅ Perfeito | ✅ Perfeito |
| **Streaming** | ✅ Automático | ✅ Automático |
| **Loops/Iterações** | ❌ Impossível | ✅ Nativo |
| **Decisões condicionais** | ❌ Impossível | ✅ Nativo |
| **Estado complexo (TypedDict)** | ❌ Apenas variáveis simples | ✅ Estados tipados |
| **Persistência/Checkpointing** | ❌ Não | ✅ Sim |
| **Human-in-the-Loop** | ❌ Não | ✅ Sim (pausar, retomar) |
| **Multi-agente / paralelização** | ❌ Não (apenas serial) | ✅ Sim |

**Exemplo prático de limitação**: Um agente ReAct que faz o ciclo:
1. **Think** (raciocina)
2. **Act** (executa ação/tool)
3. **Observe** (observa resultado)
4. **Decide** (continua ou para?)

Este ciclo é **impossível em LCEL puro** porque não há suporte para loops. É aqui que **LangGraph** entra, permitindo estados, nós e arestas para representar fluxos complexos.

> **📚 Referência**: O Capítulo 6 (LangGraph) explora como construir agents e workflows com loops e decisões condicionais.

### 4.8 Resumo do Capítulo

Neste capítulo, você aprendeu:

- O que é **LCEL** e por que simplifica composição
- Como usar o operador **pipe (`|`)** para encadear componentes
- Construir um **RAG simples com LCEL**
- Limitações do LCEL (loops, decisões complexas)
- **Próximo passo**: LangGraph para agentes com ciclos

### 4.9 Exercícios

1. **Modifique o RAG**: Adicione um nó de pré-processamento que converte a pergunta em 3 variações antes de fazer a busca.

2. **Reuse de chains**: Crie uma função que retorna chains reutilizáveis para diferentes tarefas (Q&A, sumarização, tradução).

3. **Streaming**: Implemente um chatbot simples que usa LCEL com streaming de respostas.

---

## Capítulo 5: Modularidade e Interoperabilidade

### 5.1 Separação de Pacotes no LangChain v1.0

Um grande problema das versões anteriores era: "Para usar OpenAI, instalo `langchain` e `openai`?". A resposta era confusa.

**LangChain v1.0+** resolve isso com uma arquitetura modular clara:

```
langchain-core
├── Tipos, mensagens, LCEL
└── Interfaces abstratas (LLM, ChatModel, Tool, etc.)

langchain
├── Construções de alto nível
└── Abstrações agnósticas

langchain-openai
├── ChatOpenAI, OpenAIEmbeddings
└── Implementação específica de OpenAI

langchain-anthropic
├── ChatAnthropic
└── Implementação específica de Anthropic

langchain-google-genai
├── ChatGoogleGenerativeAI
└── Implementação específica de Google

langchain-community
├── Conectores mantidos pela comunidade
└── Integrações experimentais
```

**Benefício**: Você instala apenas o que precisa.

```bash
# Uso com OpenAI
pip install langchain langchain-core langchain-openai

# Trocar para Anthropic (sem quebrar seu código LCEL)
pip uninstall langchain-openai
pip install langchain-anthropic
# Só troca a importação: ChatOpenAI → ChatAnthropic
```

### 5.2 Standard Content Blocks (Interoperabilidade de Output)

Diferentes provedores retornam respostas em formatos diferentes. **Standard Content Blocks** (v1.0) normalizam isso.

Uma mensagem agora pode conter múltiplos blocos estruturados:

```python
from langchain_core.messages import AIMessage

# Output normalizado (funciona com OpenAI, Claude, Gemini)
message = AIMessage(
    content="Aqui está a resposta",
    content_blocks=[
        {"type": "text", "text": "Resposta principal"},
        {"type": "reasoning", "text": "Meu raciocínio..."},
        {"type": "tool_call", "tool": "search", "args": {...}},
        {"type": "citation", "source": "documento_1.pdf"}
    ]
)
```

Isso permite que **uma ferramenta de UI** renderize respostas de qualquer modelo sem mudanças:

```python
def renderizar_resposta(message: AIMessage):
    """Renderiza qualquer mensagem de qualquer provedor igual."""
    for block in message.content_blocks:
        if block["type"] == "text":
            print(block["text"])
        elif block["type"] == "reasoning":
            print(f"[Raciocínio] {block['text']}")
        elif block["type"] == "tool_call":
            print(f"[Tool] {block['tool']}")
```

### 5.3 Portabilidade: Trocar Provedores sem Refatoração

Graças à separação de pacotes e Standard Content Blocks, seu código se torna **agnóstico ao provedor**:

```python
# config.py
import os
from langchain_core.language_model import LLM

def get_model() -> LLM:
    """Factory que retorna o modelo configurado."""
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet")

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    else:
        raise ValueError(f"Provider {provider} não suportado")

# seu_app.py
from config import get_model

model = get_model()
resposta = model.invoke([...])  # Funciona com qualquer modelo!
```

**Uso**:

```bash
# Usar OpenAI
LLM_PROVIDER=openai python seu_app.py

# Trocar para Anthropic (sem mexer no código!)
LLM_PROVIDER=anthropic python seu_app.py

# Trocar para Google (sem mexer no código!)
LLM_PROVIDER=google python seu_app.py
```

### 5.4 Exemplo: Pipeline Multi-Provider

Construa um pipeline que usa múltiplos provedores:

```python
# pipeline_multi_provider.py
import os
from dotenv import load_dotenv
from config import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompts específicos para cada tarefa
analise_prompt = ChatPromptTemplate.from_template(
    "Analise criticamente: {texto}"
)

review_prompt = ChatPromptTemplate.from_template(
    "Revise e aprove ou critique: {analise}"
)

# Modelos: use diferentes provedores para diferentes tarefas
modelo_analista = get_model()  # Padrão: OpenAI (rápido)
modelo_critico = get_model()   # Poderia ser Anthropic (mais cuidadoso)

# Pipeline
pipeline = (
    {"texto": RunnablePassthrough()}
    | analise_prompt
    | modelo_analista
    | StrOutputParser()
    | {"analise": RunnablePassthrough()}
    | review_prompt
    | modelo_critico
    | StrOutputParser()
)

resultado = pipeline.invoke("Escreva uma proposta de negócio")
print(f"Resultado final:\n{resultado}")
```

### 5.5 Boas Práticas: Dependências Mínimas

Ao desenvolver bibliotecas ou aplicações, siga este padrão:

```python
# Seu pacote: requirements.txt
langchain-core>=1.0.0  # Apenas abstrações
pydantic>=2.0

# Seu código: suporta múltiplos provedores
def processar_com_llm(model: Optional[LLM] = None):
    """
    Se model é None, usa OpenAI. Caso contrário, usa o model passado.
    Assim, o usuário pode injetar qualquer modelo.
    """
    if model is None:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI()

    # Use model de forma agnóstica
    return model.invoke(...)
```

### 5.6 Resumo do Capítulo

Neste capítulo, você aprendeu:

- A **arquitetura modular** do LangChain v1.0+
- Como **separar pacotes** por provedor
- **Standard Content Blocks** para interoperabilidade
- Como construir código **agnóstico ao provedor**
- Boas práticas para **dependências mínimas**

### 5.7 Exercícios

1. **Factory pattern**: Crie uma classe `LLMFactory` que instancia modelos baseado em variáveis de ambiente.

2. **Multi-provider pipeline**: Construa um pipeline que usa OpenAI para rascunho e Anthropic para revisão.

3. **Teste de portabilidade**: Implemente um teste que roda o mesmo código com 3 provedores diferentes.
