import pandas as pd
import streamlit as st
import plotly.express as px

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv('data/diabetes_data_upload.csv')

# função para treinar o modelo
def train_model():
    data = get_data()
    
    #Separando os Dados de Treino e de Teste

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    # transformadand os atributos em dados categóricos
    labelencoder_X = LabelEncoder()
    for i in range(1,X.shape[1]):
        X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
    # transformadand a saída em dados categóricos
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)  
  
    #Redimensionando os Dados - Padronização com o MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    X = mm.fit_transform(X)

    #Treinamento da Máquina Preditiva
    from sklearn.svm import SVC
    Maquina_preditiva = SVC(kernel='linear', gamma=1e-5, C=10, random_state=7)
    Maquina_preditiva.fit(X, y)
    return Maquina_preditiva

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

# título
st.title("Sistema de Previsão de Diagnóstico de Diabetes - By Daniel Gleison")

# subtítulo
st.markdown("Dataset: http://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.# .")

st.sidebar.subheader("Selecione os sintomas para análise do diagnóstico")

# mapeando dados do usuário para cada atributo
idade = 1
genero = 1
poliuria = 1
polidipsia = 1
perda_peso = 1
fraqueza = 1
polifagia = 1
tordo_genital = 1
embacamento_visual = 1
coceira = 1
irritabilidade = 1
demora_cura = 1
paresia_parcial = 1
rigidez_muscular = 1
alopecia = 1
obesidade = 1

#indice_inad = st.sidebar.number_input("Índice de Inadimplência", value=data.indice_inad.mean())
#anot_cadastrais = st.sidebar.number_input("Anotações Cadastrais", value=data.anot_cadastrais.mean())
#class_renda = st.sidebar.number_input("Classificação da Renda", value=data.class_renda.mean())
#saldo_contas = st.sidebar.number_input("Saldo de Contas", value=data.saldo_contas.mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição do Diagnóstico")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[idade,
                             genero,
                             poliuria,
                             polidipsia,
                             perda_peso,
                             fraqueza,
                             polifagia,
                             tordo_genital,
                             embacamento_visual,
                             coceira,
                             irritabilidade,
                             demora_cura,
                             paresia_parcial,
                             rigidez_muscular,
                             alopecia,
                             obesidade]])
    
    st.subheader("O diagnóstico previsto é:")
    result = result[0]
    st.write(result)

# verificando o dataset
#st.subheader("Selecionando as Variáveis de análise dos clientes")

# atributos para serem exibidos por padrão
#defaultcols = ["anot_cadastrais","indice_inad","class_renda","saldo_contas"]

# defindo atributos a partir do multiselect
#cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 8 registro do dataframe
#st.dataframe(data[cols].head(7))