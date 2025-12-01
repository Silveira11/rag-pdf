# RAG PDF - Assistente de Documentos Fiscais

Sistema de perguntas e respostas sobre documentos PDF da área fiscal usando IA local com Ollama.

## Funcionalidades

- Carregar e processar documentos PDF fiscais
- Busca semântica por similaridade
- Respostas usando Ollama com modelos locais
- Interface simples de perguntas e respostas

## Instalação

```bash
git clone https://github.com/Silveira11/rag-pdf.git
cd rag-pdf
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b
