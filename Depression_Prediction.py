import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Carregar o modelo salvo
modelo = load_model('modelo_final')

# Título e descrição do projeto
st.title("Detecção de Pacientes com Potencial Diagnóstico de Depressão")
st.write(
    "Este projeto de Ciência de Dados visa identificar pacientes com maior probabilidade de estarem enfrentando "
    "depressão, utilizando técnicas de Machine Learning. Faça o upload de um arquivo CSV com os dados para obter as previsões."
)

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregue o arquivo CSV com os dados dos pacientes", type=["csv"])

if uploaded_file is not None:
    # Ler o arquivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Verificar se a coluna 'id' está presente
    if 'id' not in df.columns:
        st.error("O arquivo CSV precisa conter uma coluna chamada 'id'.")
    else:
        # Fazer predições
        predicoes = predict_model(modelo, data=df)
        
        # Criar DataFrame com ID e previsão
        resultado = pd.DataFrame({
            'id': df['id'],
            'Depression': predicoes['prediction_label']
        })
        
        # Exibir o resultado
        st.write("### Resultado das Predições:")
        st.dataframe(resultado)
        
        # Botão para baixar os resultados
        csv = resultado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar Resultados", 
            data=csv, 
            file_name="predicao_depressao.csv", 
            mime="text/csv"
        )