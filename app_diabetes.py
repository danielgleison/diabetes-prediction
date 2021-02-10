import pandas as pd
import streamlit as st

# função para carregar o dataset
@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv('data/diabetes_data_upload.csv')

# função para treinar o modelo
def train_model():

    data = get_data()
    # transformando os atributos em dados categóricos
    from sklearn.preprocessing import LabelEncoder 
    objectList = data.select_dtypes(include = 'object').columns
    le = LabelEncoder()
    for i in objectList:
        data[i] = le.fit_transform(data[i])

    # mapa de correlação
    corr = data.corr()['class']

    #Separando as variaveis de entrada e saída
    X = data.drop(["class"],axis=1)
    y = data["class"]
    
    # padronização da coluna Age
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    X[['Age']] = mm.fit_transform(X[['Age']])
  
    #Separando os Dados de Treino e de Teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2) 

    #Treinamento dos modelos
    modelos = ['SVC','Random Forest','KNeighbors','LogisticRegression','Naive_Bayes',]

    column_names = ["Modelo","Acuracia","Precisao","Recall","F1",
                        "Total_Positivos","Total_Negativos", "Falsos_Positivos", "Falsos_Negativos",
                        "Classificador"]
    results = pd.DataFrame(columns = column_names)

    for i in range(0,len(modelos)):
        
        if i == 0:
            from sklearn.svm import SVC
            classifier = SVC(kernel='linear', gamma= 1e-5, C=10,random_state=7)
         
        elif  i == 1:
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators=100)
    
        elif  i == 2:
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=30)
    
        elif  i == 3:
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression()  
    
        elif  i == 4:
            from sklearn.naive_bayes import GaussianNB
            classifier = GaussianNB()
    

        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
    
        from sklearn import metrics
        acc = metrics.accuracy_score(y_test, y_pred)*100
        prc = metrics.precision_score(y_test, y_pred)*100
        rec = metrics.recall_score(y_test, y_pred)*100
        f1 = metrics.f1_score(y_test, y_pred)*100
    
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        data = [[modelos[i],acc, prc, rec, f1, tp, tn, fp, fn, classifier]]
        column_names = ["Modelo","Acuracia","Precisao","Recall","F1",
                        "Total_Positivos","Total_Negativos", "Falsos_Positivos", "Falsos_Negativos",
                        "Classificador"]
        model_results = pd.DataFrame(data = data, columns = column_names)
        results = results.append(model_results, ignore_index = True)

    return results, corr

# criando um dataframe
data = get_data()

# salvando o resultado do treinamento no dataframe
results, corr = train_model()

st.sidebar.subheader("Perfil de análise")

# mapeando dados do usuário para cada atributo
In1 =  st.sidebar.number_input("Idade", min_value=20,max_value=65,step=1)
In2 =  st.sidebar.selectbox("Gênero:", ["Masculino","Femenino"])
In3 =  st.sidebar.selectbox("Poliúria:",["Não","Sim"])
In4 =  st.sidebar.selectbox("Polidipsia:",["Não","Sim"])
In5 =  st.sidebar.selectbox("Perda repentina de peso:",["Não","Sim"])
In6 =  st.sidebar.selectbox("Fraqueza:",["Não","Sim"])
In7 =  st.sidebar.selectbox("Polifagia:",["Não","Sim"])
In8 =  st.sidebar.selectbox("Tordo genital:",["Não","Sim"])
In9 =  st.sidebar.selectbox("Embaçamento visual:",["Não","Sim"])
In10 = st.sidebar.selectbox("Coceira:",["Não","Sim"])
In11 = st.sidebar.selectbox("Irritabilidade:",["Não","Sim"])
In12 = st.sidebar.selectbox("Demora de cura:",["Não","Sim"])
In13 = st.sidebar.selectbox("Paresia parcial:",["Não","Sim"])
In14 = st.sidebar.selectbox("Rigidez muscular:",["Não","Sim"])
In15 = st.sidebar.selectbox("Alopecia:",["Não","Sim"])
In16 = st.sidebar.selectbox("Obesidade:",["Não","Sim"])

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

st.sidebar.write('') 
st.sidebar.write('**Daniel Gleison M. Lira**')
st.sidebar.write('**Mestrado em Ciências da Computação**')
st.sidebar.write('**Universidade Estadual do Ceará**')
st.sidebar.write('mailto:daniel.gleison@aluno.uece.br')
st.sidebar.markdown('https://github.com/danielgleison')

# título
#image = Image.open('Logo.png')
#st.image(image, use_column_width=True, use_column_height=True)
st.title("Sistema de Classificação Preditiva de Diabetes")



# subtítulo
st.write ('Aplicação acadêmica para classificação preditiva de diabetes ' +
        'utilizando técnicas de Machine Learning (Aprendizado de Máquina). ' +
        'A Inteligência Artificial na área de Saúde tem por objetivo prover ' +
        'análises preditivas e auxiliar a tomada de decisão, não devendo ser utilizada ' +
        'em substituição ao diagnóstico de profissional qualificado. ' +
        'Preencha o perfil de análise e clique no botão Realizar Predição para a visualização dos resultados.')
st.markdown("Dataset: https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset")
 
# verifica se o botão foi acionado
if btn_predict:

    data = [In1,In2,In3,In4,In5,In6,In7,In8,In9,In10,In11,In12,In13,In14,In15,In16]
    column_names = ["Idade","Genero","Poliuria","Polidipsia","Perda_Peso","Fraqueza","Polifagia","Tordo_genital",\
                    "Embacamento_visual","Coceira","Irritabilidade",\
                    "Demora_cura","Paresia_parcial","Rigidez_muscular","Alopecia","Obesidade"]
    df = pd.DataFrame(data, column_names)

    if  df[0][1] == 'Masculino': df[0][1] = 1 
    elif df[0][1]  == 'Femenino': df[0][1] = 0
            
    for x in range(2,16):
        if   df[0][x] == 'Sim': df[0][x] = 1 
        elif df[0][x]  == 'Não':df[0][x] = 0


    # padronização do input Idade
    df[0][0] = (df[0][0] - 16) / 74
        
    # resultado da predição
        
    pred = [list(df[0])]

    classifier_best = results['Classificador'][results['Recall'] == results['Recall'].max()].values
    classifier = classifier_best[0]

    model_best = results['Modelo'][results['Recall'] == results['Recall'].max()].values
    model = model_best[0]

    result = classifier.predict(pred)  
    result = result[0]

    if result == 0: st.write("Resultado com maior probabilidade: **NEGATIVO**")
    if result == 1: st.write("Resultado com maior probabilidade:: **POSITIVO**")
    st.write("Modelo com melhor recall (sensibilidade): ", model)
    st.write(results)
        
    st.write("Matriz de Correlação")
    st.write(corr)

    #import matplotlib.pyplot as plot
    #graf_corr = corr.plot.bar(figsize = (16,6), title = 'Correlação com Diabetes', fontsize = 15, rot = 90, grid = True)
    #st.write(graf_corr)

    # verificando o dataset
    #st.subheader("Selecionando as Variáveis de Avaliação")

    # atributos para serem exibidos por padrão
    #defaultcols = ["Modelo","Acurácia","Precisão","Recall","F1", "Tempo Treino", "FP", "FN", "TP", "TN" ]

    # defindo atributos a partir do multiselect
    #cols = st.multiselect("Atributos", results.columns.tolist(), default=defaultcols)

    # exibindo os top 8 registro do dataframe
    #st.dataframe(results[cols].head(7))
