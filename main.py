from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """Você é um assistente especializado em direito tributário brasileiro.

Com base EXCLUSIVAMENTE no seguinte contexto:

{base_conhecimento}

Responda à seguinte pergunta de forma precisa e detalhada:

{pergunta}

**Regras importantes:**
- Responda APENAS com base nas informações do contexto fornecido
- Se a informação não estiver no contexto, diga: "Não encontrei informações específicas sobre isso no banco de dados."
- Não invente informações nem use conhecimento externo
- Seja claro e preciso

Resposta:"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    funcao_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embeddings
    )

    documentos = db.similarity_search(pergunta, k=5)

    if not documentos:
        print("❌ Nenhuma informação relevante encontrada no banco de dados.")
        return

    textos_resultado = [doc.page_content for doc in documentos]
    base_conhecimento = "\n\n---\n\n".join(textos_resultado)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    mensagem = prompt.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    })

    modelo = ChatOllama(
        model="llama3.1:8b",
        temperature=0.3,
        num_predict=1500
    )
    
    resposta = modelo.invoke(mensagem)

    print("\n💬 Resposta:")
    print("=" * 60)
    print(resposta.content)
    print("=" * 60)

perguntar()
