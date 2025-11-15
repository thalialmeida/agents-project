import streamlit as st
import sys
import os
import pandas as pd
import tempfile
from uuid import uuid4

# Adiciona o diretório do projeto ao path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importações principais do agente
from agentai.workflow import WorkflowExecutor
from langchain_community.chat_models import ChatDeepInfra

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("Agente de Pré-processamento e Forecasting")
st.markdown(
    "Carregue um arquivo CSV e descreva em linguagem natural a tarefa desejada. "
    "O agente fará o pré-processamento, análise e previsão automaticamente."
)

# Estados persistentes
if "executor" not in st.session_state:
    st.session_state.executor = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_modificado" not in st.session_state:
    st.session_state.df_modificado = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None


with st.sidebar:
    st.header("1️⃣ Carregar Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        if st.button("🚀 Iniciar Agente"):
            with st.spinner("Inicializando agente..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        csv_path = tmp_file.name

                    df = pd.read_csv(csv_path)
                    st.session_state.df_original = df.copy()
                    st.session_state.df_modificado = None

                    llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")

                    base_dir = os.path.dirname(__file__)
                    results_path = os.path.join(base_dir, "agentai", "results", "forecast", "test")

                    st.session_state.executor = WorkflowExecutor(
                        llm=llm,
                        csv_path=csv_path,
                        plot_images_path=results_path
                    )
                    st.success("✅ Agente pronto para receber instruções!")
                except Exception as e:
                    st.error(f"Erro ao iniciar agente: {e}")
                    st.session_state.executor = None

if st.session_state.executor is None:
    st.info("Carregue um CSV na barra lateral para começar.")
else:
    st.header("📊 Dados Carregados")
    st.dataframe(st.session_state.df_original, use_container_width=True)

    st.header("🗣️ Instrução para o Agente")
    prompt_usuario = st.text_area(
        "Descreva a tarefa que o agente deve executar:",
        height=150,
        placeholder="Exemplo: verifique valores ausentes e preveja a coluna 'pressure'."
    )

    if st.button("▶️ Executar"):
        if not prompt_usuario.strip():
            st.warning("Por favor, insira uma instrução antes de executar.")
        else:
            thread_id = str(uuid4())
            executor_instance = st.session_state.executor

            # Espaços dinâmicos de UI
            st.markdown("---")
            progress = st.progress(0)
            log_container = st.empty()

            logs_temp = []
            total_nodes = 6  # aproximadamente: supervisor, inspect, imputator, feature, automl, summarizer
            current_node = 0

            with st.spinner("O agente está trabalhando..."):
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    initial_state = {
                        "msg": prompt_usuario,
                        "logs": [],
                        "main_goal": prompt_usuario,
                        "is_before_dp": True,
                    }

                    # Execução em tempo real
                    for chunk in executor_instance.graph.stream(initial_state, config=config, recursion_limit=35):
                        for node_name, state in chunk.items():
                            current_node += 1
                            log_entry = f"### 🧩 [{node_name.upper()}]\n"
                            if report := state.get("subagents_report"):
                                log_entry += f"{report}\n\n"
                            if node_name == "supervisor":
                                log_entry += f"➡️ **Próximo passo:** `{state.get('next')}`\n"

                            logs_temp.append(log_entry)
                            progress.progress(min(current_node / total_nodes, 1.0))
                            log_container.markdown("\n---\n".join(logs_temp))

                            final_state = state  # armazena último estado válido

                    # Atualiza sessão
                    st.session_state.df_modificado = executor_instance.df.copy()
                    st.session_state.last_result = final_state
                    st.success("✅ Execução concluída com sucesso!")

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a execução: {e}")
                    st.session_state.last_result = None

if st.session_state.df_modificado is not None:
    st.header("📋 DataFrame Processado")
    st.dataframe(st.session_state.df_modificado, use_container_width=True)

    st.header("📊 Visualizações Geradas")
    images_path = os.path.join(os.path.dirname(__file__), "agentai", "results", "forecast", "test")

    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(images_path)
        for file in files if file.lower().endswith(".png")
    ]

    if image_files:
        cols = st.columns(2)
        for idx, img_path in enumerate(sorted(image_files)):
            with cols[idx % 2]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info(f"Nenhuma imagem encontrada em: `{images_path}`")

if st.session_state.last_result:
    with st.expander("📝 Logs Finais da Execução", expanded=False):
        logs = st.session_state.last_result.get("logs", [])
        if logs:
            log_formatado = "\n".join([f"- {log}" for log in logs])
            st.markdown(f"```\n{log_formatado}\n```")
        else:
            st.info("Nenhum log registrado.")
