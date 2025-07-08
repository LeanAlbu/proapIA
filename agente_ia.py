import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 1. CONFIGURAÇÃO INICIAL ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Verificação para garantir que a chave foi inserida
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "SUA_CHAVE_API_AQUI":
    print("ERRO: Por favor, insira sua chave de API do Google na variável GOOGLE_API_KEY.")
    exit()

# Nome do arquivo PDF que será lido
NOME_ARQUIVO_PDF = "seu_documento.pdf"

# Verificação se o arquivo PDF existe
if not os.path.exists(NOME_ARQUIVO_PDF):
    print(f"ERRO: O arquivo '{NOME_ARQUIVO_PDF}' não foi encontrado.")
    print("Por favor, certifique-se de que o arquivo PDF está na mesma pasta que este script.")
    exit()

print("Iniciando o processo de configuração do agente de IA...")

# --- 2. CARREGAMENTO E PROCESSAMENTO DO DOCUMENTO ---
# Carrega o documento PDF
print(f"Carregando o documento '{NOME_ARQUIVO_PDF}'...")
loader = PyPDFLoader(NOME_ARQUIVO_PDF, Arquivo2)
docs = loader.load()

# Divide o documento em pedaços menores (chunks)
print("Dividindo o documento em pedaços (chunks)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3. CRIAÇÃO DOS EMBEDDINGS E BANCO DE DADOS VETORIAL ---
# Cria os embeddings (vetores de significado) e os armazena no ChromaDB
# O ChromaDB é um banco de dados vetorial leve que roda localmente.
print("Criando embeddings e armazenando no banco de dados vetorial (ChromaDB)...")
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Cria o "retriever", que é o componente responsável por buscar os chunks relevantes
retriever = vectorstore.as_retriever(search_kwargs = {'k':5})

# --- 4. CONFIGURAÇÃO DO MODELO DE LINGUAGEM (LLM) E DA CADEIA (CHAIN) ---
# Inicializa o modelo de linguagem (LLM) que vai gerar as respostas
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) # Usamos o gemini-1.5-flash que é rápido e eficiente

# Criação do prompt template. Este é o modelo de instrução para a IA.
# Ele instrui a IA a responder a pergunta usando SOMENTE o contexto fornecido.
prompt_template = """
Você é um assistente especializado em responder perguntas com base em documentos.
Responda à pergunta do usuário utilizando somente as informações do contexto abaixo.
Se a resposta não estiver no contexto, diga "Com base nos documentos fornecidos, não encontrei uma resposta para essa pergunta."

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# Criação da cadeia (chain) RAG (Retrieval-Augmented Generation)
# Esta cadeia irá:
# 1. Recuperar (retrieve) documentos relevantes usando o `retriever`.
# 2. "Rechear" (stuff) os documentos no prompt.
# 3. Passar o prompt para o modelo de linguagem (LLM) para gerar a resposta.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ Agente de IA pronto! Faça suas perguntas sobre o documento.")
print('   Digite "sair" a qualquer momento para encerrar o programa.\n')


# --- 5. LOOP DE INTERAÇÃO COM O USUÁRIO ---
while True:
    try:
        # Pede ao usuário para digitar uma pergunta
        user_question = input("Sua pergunta: ")

        # Verifica se o usuário quer sair
        if user_question.lower() == 'sair':
            print("Encerrando o programa. Até mais!")
            break
        
        if not user_question.strip():
            continue

        # Invoca a cadeia RAG com a pergunta do usuário e exibe a resposta
        print("\nBuscando a resposta...")
        response = rag_chain.invoke(user_question)
        print("\nResposta do Agente:")
        print(response)
        print("-" * 50)

    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        print("Reiniciando o loop de perguntas.")
        continue
