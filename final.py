import streamlit as st
import os
import pandas as pd
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Configurar chave da API OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Chave da API OpenAI não encontrada. Configure o arquivo .env corretamente.")

# Configuração da interface Streamlit
st.title("Sistema de Comparação de Arquivos SCI e REI com IA")
st.subheader("Extração e Comparação de Dados de Equipamentos com Justificativas Baseadas em IA")

# Upload dos arquivos
st.write("**Upload dos Arquivos**")
file1 = st.file_uploader("Upload do arquivo SCI (Excel)", type=["xlsx"])
file2 = st.file_uploader("Upload do arquivo REI (Excel)", type=["xlsx"])

# Entrada de parâmetro para filtro
filtro_site_name = st.text_input("Digite o nome do site a ser filtrado no REI", value="")

# Função para localizar o cabeçalho no SCI
def localizar_inicio_tabela(df, termos_cabecalho):
    for idx, row in df.iterrows():
        if all(term in row.astype(str).values for term in termos_cabecalho):
            return idx
    return None

# Função para extrair e agrupar a tabela SCI
def extrair_e_agrupar_tabela_sci(file):
    try:
        st.write("[INFO] Extraindo e agrupando dados do arquivo SCI")
        sci_df = pd.read_excel(file, header=None)
        termos_cabecalho = ["TIPO DE SOLICITAÇÃO"]  # Define o cabeçalho principal
        inicio_tabela = localizar_inicio_tabela(sci_df, termos_cabecalho)
        if inicio_tabela is None:
            raise ValueError("Cabeçalho não encontrado no SCI.")
        tabela = sci_df.iloc[inicio_tabela + 1:].reset_index(drop=True)
        tabela.columns = sci_df.iloc[inicio_tabela].tolist()
        tabela = tabela.dropna(subset=["TIPO DE SOLICITAÇÃO"]).reset_index(drop=True)

        # Agrupamento por TIPO DE SOLICITAÇÃO e MODELO, somando QTDE
        tabela_agrupada = tabela.groupby(
            ["TIPO DE SOLICITAÇÃO", "MODELO", "TIPO DO EQUIPAMENTO"],
            as_index=False
        ).agg({
            "FABRICANTE": "first",
            "FREQUÊNCIA DE OPERAÇÃO": "first",
            "DIÂMETRO ANTENA DE MW": "first",
            "QTDE": "sum",
            "RAD CENTER": "first"  # Mantém o primeiro registro
        })

        st.write("[INFO] Extração e agrupamento da tabela SCI concluídos")
        st.write("**Tabela SCI Agrupada:**")
        st.dataframe(tabela_agrupada)
        return tabela_agrupada
    except Exception as e:
        st.error(f"Erro ao extrair e agrupar a tabela do arquivo SCI: {e}")
        return pd.DataFrame()

# Função para extrair e agrupar a tabela REI
def extrair_e_agrupar_tabela_rei(file, filtro_site_name):
    try:
        st.write("[INFO] Extraindo e agrupando dados do arquivo REI")
        rei_df = pd.read_excel(file)
        rei_df = rei_df.dropna(how="all").reset_index(drop=True)
        if "Customer Equipment Status" in rei_df.columns:
            rei_df = rei_df[rei_df["Customer Equipment Status"].isin(["Real", "Contracted", "Proposed", "Proposed Real / Irregular", "Proposed Real"])]
        if "Site Name" in rei_df.columns and filtro_site_name in rei_df["Site Name"].values:
            rei_filtrado = rei_df[rei_df["Site Name"].str.contains(filtro_site_name, na=False)]
        elif "EnderecoId" in rei_df.columns:
            rei_filtrado = rei_df[rei_df["EnderecoId"].str.contains(filtro_site_name, na=False)]
        else:
            raise ValueError("Colunas 'Site Name' ou 'EnderecoId' não encontradas no REI.")

        # Agrupamento por Antenna Model e Customer Equipment Status
        rei_agrupado = rei_filtrado.groupby(
            ["Antenna Model", "Customer Equipment Status"],
            as_index=False
        ).agg({
            "Customer Name": "first",
            "Antenna Manufacturer": "first",
            "Antenna Count": "sum",
            "Tower Height of Antenna": "mean",  # Média da altura
            "Site Name": "first",
            "EnderecoId": "first",
            "Azimuth": "first",
            "Antenna Type": "first"
        })

        st.write("[INFO] Extração e agrupamento da tabela REI concluídos")
        st.write("**Tabela REI Agrupada:**")
        st.dataframe(rei_agrupado)
        return rei_agrupado
    except Exception as e:
        st.error(f"Erro ao extrair e agrupar a tabela do arquivo REI: {e}")
        return pd.DataFrame()

# Função para comparar os modelos usando LangChain
def comparar_modelos_com_langchain(sci_df, rei_df):
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)

    template = PromptTemplate(
        input_variables=["equipamentos_sci", "equipamentos_rei"],
        template="""
        Você é especialista no setor de Telecomunicações.
        Analise a correspondência entre os equipamentos do SCI e do REI. Para cada equipamento do SCI, verifique o modelo, quantidade, altura e tipo de equipamento.
        Se um modelo não for encontrado diretamente, utilize o tipo de equipamento (SCI) e a antenna type (REI) para buscar correspondências.
        Considere alturas como correspondentes se estiverem dentro de uma diferença de 4 metros.
        Para cada equipamento do SCI, classifique como "Exato", "Similar" ou "Não Encontrado" e forneça uma justificativa.

        Equipamentos SCI:
        {equipamentos_sci}

        Equipamentos REI:
        {equipamentos_rei}

        Retorne uma lista estruturada com:
        | Equipamento SCI | Tipo de Solicitação | Tipo de Equipamento | Quantidade SCI | Quantidade REI | Altura SCI | Altura REI | Correspondência | Justificativa |
        """
    )

    chain = LLMChain(llm=llm, prompt=template)

    # Garantir que todos os valores sejam strings para o prompt
    equipamentos_sci = [
        f"{row['MODELO']} ({row['TIPO DE SOLICITAÇÃO']}, {row['TIPO DO EQUIPAMENTO']}, QTD: {row['QTDE']}, ALT: {row['RAD CENTER']})"
        for _, row in sci_df.iterrows()
    ]
    equipamentos_rei = [
        f"{row['Antenna Model']} ({row['Customer Equipment Status']}, QTD: {row['Antenna Count']}, ALT: {row['Tower Height of Antenna']})"
        for _, row in rei_df.iterrows()
    ]

    resposta = chain.run({
        "equipamentos_sci": "\n".join(equipamentos_sci),
        "equipamentos_rei": "\n".join(equipamentos_rei)
    })

    linhas = re.findall(r'\| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|', resposta)
    return pd.DataFrame(linhas, columns=[
        "Equipamento SCI", "Tipo de Solicitação", "Tipo de Equipamento", 
        "Quantidade SCI", "Quantidade REI", "Altura SCI", 
        "Altura REI", "Correspondência", "Justificativa"
    ])

def verificar_regras_de_negocio(df_resultados):
    alertas = []
    fabricantes_instalacao = []
    fabricantes_remocao = []

    # Coletar os fabricantes para as condições
    for _, row in df_resultados.iterrows():
        if row["Tipo de Solicitação"].strip().lower() == "instalação":
            fabricantes_instalacao.append(row["Equipamento SCI"].split()[0].upper())
        elif row["Tipo de Solicitação"].strip().lower() == "remoção":
            fabricantes_remocao.append(row["Equipamento SCI"].split()[0].upper())

    # Validar cada linha para regras específicas
    for _, row in df_resultados.iterrows():
        tipo = row["Tipo de Solicitação"].strip().lower()
        fabricante = row["Equipamento SCI"].split()[0].upper()

        if tipo == "instalação":
            if fabricante in ["NOKIA", "HUAWEI"] and "ERICSSON" in fabricantes_remocao:
                # Caso NOKIA ou HUAWEI na instalação e ERICSSON na remoção
                continue  # Não gerar alerta
            elif row["Correspondência"].strip().lower() in ["exato", "similar"]:
                alertas.append(f"Alerta: Possível gap - Equipamento {row['Equipamento SCI']} encontrado no REI para instalação.")
        elif tipo == "remoção":
            if row["Correspondência"].strip().lower() not in ["exato", "similar"]:
                alertas.append(f"Alerta: Equipamento {row['Equipamento SCI']} para remoção não encontrado no REI.")

    if not alertas:
        return "SCI OK"
    else:
        return "\n".join(alertas)


# Botão para iniciar o processo de comparação
if st.button("Iniciar Comparação com IA"):
    if file1 is not None and file2 is not None:
        try:
            # Extração e agrupamento
            sci_df_agrupado = extrair_e_agrupar_tabela_sci(file1)
            rei_df_agrupado = extrair_e_agrupar_tabela_rei(file2, filtro_site_name)

                        # Continuar com a lógica de LangChain para comparar os modelos
            st.write("[INFO] Processo de análise de correspondência com LangChain iniciado")
            df_resultados = comparar_modelos_com_langchain(sci_df_agrupado, rei_df_agrupado)

            # Exibir os resultados
            st.write("**Resultados da Comparação:**")
            st.dataframe(df_resultados)

            # Verificar regras de negócio
            st.write("**Relatório Final:**")
            relatorio_final = verificar_regras_de_negocio(df_resultados)
            st.text(relatorio_final)

        except Exception as e:
            st.error(f"Erro inesperado ao processar os arquivos: {e}")
    else:
        st.warning("Por favor, faça o upload dos dois arquivos para continuar.")

# Rodapé da aplicação Streamlit
st.write("---")
st.caption("Powered by Tech in Torres and OpenAI")

