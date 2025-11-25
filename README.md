# OPA – Observe, Preprocess, and Act: A Multi-Agentic Framework for Data Preprocessing and Predictive Intelligence

LLM project

# Architecture

>  <p align="left">
> <img width="662" height="311" alt="opadiagram" src="https://github.com/user-attachments/assets/624dc244-aa83-4254-a924-1988f9efdda7" />
>  </p> 

## Repo Structure 

```plaintext

agenteiadp/
├── agentai/                # Módulos principais
|   ├── FAISS_DB/           # Datasets
|   ├── datasets/           # Datasets
|   ├── modules/            # Utilitários
│   ├── __init__.py
│   ├── agents.py           # Agente inteligente principal
│   ├── base_rag.txt        # RAG
│   ├── nodes.py            # Nós
│   ├── rag.py              # RAG
│   ├── tools.py            # Ferramentas dos agentes
│   └── workflow.py         # Grafo de orquestração
│   ├── workflow_graph.png  # Arquitetura
├── help/                   # Algumas orientações
├── notebooks/              # Notebooks testes
│   └── datasets/           # Datasets utilizados nos notebooks
├── app.py                  # Streamlit
├── main.py                 # Executer
├── requirements.txt        # Dependências do projeto
└── README.md               #  Este arquivo
```
