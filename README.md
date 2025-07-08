# proapIA

Agente de IA em desenvolvimento para a PROAP

## Descrição

O `proapIA` é um agente de inteligência artificial focado em responder perguntas com base em documentos fornecidos, utilizando tecnologia de embeddings, banco de dados vetorial e modelos de linguagem generativos. O objetivo é facilitar o acesso à informação presente em documentos da PROAP de forma rápida e contextualizada.

## Como funciona

- Carrega um arquivo PDF definido pelo usuário e realiza a divisão do conteúdo em pedaços (chunks) para facilitar o processamento.
- Cria embeddings desses pedaços e armazena em um banco de dados vetorial leve (ChromaDB) local.
- Utiliza um modelo generativo da Google (gemini-1.5-flash) para responder perguntas do usuário usando apenas as informações extraídas do documento.
- O sistema segue o paradigma Retrieval-Augmented Generation (RAG): ao receber uma pergunta, busca os trechos mais relevantes do documento e gera uma resposta baseada nesses trechos.
- Se a resposta não estiver no contexto dos documentos, o agente informa o usuário.

## Requisitos

- Python 3.x
- Chave de API do Google (para uso do modelo generativo)
- Dependências Python:
  - `google.generativeai`
  - `langchain_google_genai`
  - `langchain_community`
  - `langchain`
  - `chromadb`

## Uso

1. Coloque o arquivo PDF que deseja consultar na mesma pasta do script, e ajuste o nome do arquivo no código (`NOME_ARQUIVO_PDF`).
2. Insira sua chave de API do Google na variável de ambiente `GOOGLE_API_KEY`.
3. Execute o script `agente_ia.py`.
4. Faça perguntas sobre o conteúdo do documento pelo terminal.
5. Para encerrar, digite `sair`.

## Exemplo de execução

```
Iniciando o processo de configuração do agente de IA...
Carregando o documento 'seu_documento.pdf'...
Dividindo o documento em pedaços (chunks)...
Criando embeddings e armazenando no banco de dados vetorial (ChromaDB)...

✅ Agente de IA pronto! Faça suas perguntas sobre o documento.
   Digite "sair" a qualquer momento para encerrar o programa.

Sua pergunta: Qual é o objetivo do documento?
Buscando a resposta...

Resposta do Agente:
[Resposta gerada com base no documento]
--------------------------------------------------
```

## Licença

Este projeto está em desenvolvimento e ainda não possui uma licença definida.

## Autor

[LeanAlbu](https://github.com/LeanAlbu)
