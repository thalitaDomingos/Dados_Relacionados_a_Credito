from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report  #metricas de validação
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Banco_de_Dados.Banco_de_Dados import Database
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd


# ("Servidor Local", "Usuario", "Senha") do mysql
bancoDados = Database("127.0.0.1", "root", "root")

# criando o banco de dados
mydb = bancoDados.createDatabase()


# Convert Database to Pandas Dataframe
db_connection_str = 'mysql+pymysql://root:' + 'root' + '@localhost:3306/statlog'
db_connection = create_engine(db_connection_str)
df = pd.read_sql('select * from `germancredit`', con=db_connection)


x = df.drop(['id', 'kredit'], axis=1)
y = df['kredit']


# Separando treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# Instanciando modelos
clf_tree = DecisionTreeClassifier()  # Decision Tree
clf_neighbors = KNeighborsClassifier()  # k-Nearest Neighbors
clf_neural_network = MLPClassifier()  # Multilayer Perceptron
clf_naive_bayes = GaussianNB()  # Naıve Bayes
clf_linear_model = Perceptron()  # Perceptron


'''
# x = MinMaxScaler().fit_transform(df.drop(['id', 'kredit'], axis=1))
parameters = {
    'hidden_layer_sizes': [(20), (40, 20), (80, 40, 20), (160, 80, 40, 20), (320, 160, 80, 40, 20)],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    'momentum': [5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e0],
    'max_iter': [3000]
}

clf_neural_network = GridSearchCV(
    clf_neural_network_base, parameters, n_jobs=-1, verbose=10)

'''

# Treinando modelos
clf_tree.fit(x_train, y_train)  # Decision Tree
clf_neighbors.fit(x_train, y_train)  # k-Nearest Neighbors
clf_neural_network.fit(x_train, y_train)  # Multilayer Perceptron
clf_naive_bayes.fit(x_train, y_train)  # Naıve Bayes
clf_linear_model.fit(x_train, y_train)  # Perceptron

# Prevendo valores
clf_tree_predict = clf_tree.predict(x_test)
clf_neighbors_predict = clf_neighbors.predict(x_test)
clf_neural_network_predict = clf_neural_network.predict(x_test)
clf_naive_bayes_predict = clf_naive_bayes.predict(x_test)
clf_linear_model_predict = clf_linear_model.predict(x_test)

# Avaliação
clf_tree_score = accuracy_score(y_test, clf_tree_predict)
clf_neighbors_score = accuracy_score(y_test, clf_neighbors_predict)
clf_neural_network_score = accuracy_score(y_test, clf_neural_network_predict)
clf_naive_bayes_score = accuracy_score(y_test, clf_naive_bayes_predict)
clf_linear_model_score = accuracy_score(y_test, clf_linear_model_predict)

#Metricas de validação 
clf_naive_bayes_classification_report = classification_report(
    y_test, clf_naive_bayes_predict)

while True:

    option = eval(input(
        "\n1 - Simular crédito \n2 - Informações da Inteligência Artificial (IA) \n3 - Sair\n\nDigite a opção desejada:\n"))
    print(" ")

    if option == 1:

        active = True
        flag = True

        while active:

            datalist = []

            while flag:
                status = int(input("Laufkont: "))
                
                if (status >= 1 and status <= 4):
                    datalist.append(status)
                    flag = False
                    break

            duration = int(input("Laufzeit: "))
            datalist.append(duration)

            flag = True
            while flag:
                credit_history = int(input("Moral: "))
                if (credit_history >= 0  and credit_history <= 4):
                    datalist.append(credit_history)
                    flag = False
                    break

            flag = True
            while flag:
                purpose = int(input("Verw: "))
                if (purpose >= 0 and purpose <= 10):
                    datalist.append(purpose)
                    flag = False
                    break

            amount = int(input("Hoehe: "))
            datalist.append(amount)

            flag = True
            while flag:
                savings = int(input("Sparkont: "))
                if (savings >= 1 and savings <= 5):
                    datalist.append(savings)
                    flag = False
                    break

            flag = True
            while flag:
                employment_duration = int(input("Beszeit: "))
                if (employment_duration >= 1 and employment_duration <= 5):
                    datalist.append(employment_duration)
                    flag = False
                    break

            flag = True
            while flag:
                installment_rate = int(input("Rate: "))
                if (installment_rate >= 1 and installment_rate <= 4):
                    datalist.append(installment_rate)
                    flag = False
                    break

            flag = True
            while flag:
                personal_status_sex = int(input("Famges: "))
                if (personal_status_sex >= 1 and personal_status_sex <= 4):
                    datalist.append(personal_status_sex)
                    flag = False
                    break

            flag = True
            while flag:
                other_debtors = int(input("Buerge: "))
                if (other_debtors >= 1 and other_debtors <= 3):
                    datalist.append(other_debtors)
                    flag = False
                    break

            flag = True
            while flag:
                present_residence = int(input("Wohnzeit: "))
                if (present_residence >= 1 and present_residence <= 4):
                    datalist.append(present_residence)
                    flag = False
                    break

            flag = True
            while flag:
                property_ = int(input("Verm: "))
                if (property_ >= 1 and property_ <= 4):
                    datalist.append(property_)
                    flag = False
                    break

            age = int(input("Alter: "))
            datalist.append(age)

            flag = True
            while flag:
                other_installment_plans = int(input("Weitkred: "))
                if (other_installment_plans >= 1 and other_installment_plans <= 3):
                    datalist.append(other_installment_plans)
                    flag = False
                    break

            flag = True
            while flag:
                housing = int(input("Wohn: "))
                if (housing >= 1 and housing <= 3):
                    datalist.append(housing)
                    flag = False
                    break

            flag = True
            while flag:
                number_credits = int(input("Bishkred: "))
                if (number_credits >= 1 and number_credits <= 4):
                    datalist.append(number_credits)
                    flag = False
                    break

            flag = True
            while flag:
                job = int(input("Beruf: "))
                if (job >= 1 and job <= 4):
                    datalist.append(job)
                    flag = False
                    break

            flag = True
            while flag:
                people_liable = int(input("Pers: "))
                if (people_liable == 1 or people_liable == 2):
                    datalist.append(people_liable)
                    flag = False
                    break

            flag = True
            while flag:
                telephone = int(input("Telef: "))
                if (telephone == 1 or telephone == 2):
                    datalist.append(telephone)
                    flag = False
                    break

            flag = True
            while flag:
                foreign_worker = int(input("Gastarb: "))
                if (foreign_worker == 1 or foreign_worker == 2):
                    datalist.append(foreign_worker)
                    flag = False
                    break

            db = pd.DataFrame([datalist], columns=["laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont", "beszeit", "rate",
                                                   "famges", "buerge", "wohnzeit", "verm", "alter", "weitkred", "wohn", "bishkred", "beruf", "pers", "telef", "gastarb"])

            active = False

            y_predict = clf_naive_bayes.predict(db)
            print(y_predict)

            if (y_predict == 0):
                print("Bad")
            else:
                print("Good")

    elif option == 2:

        print(f"Tamanho x de treino: {x_train.shape}")
        print(f"Tamanho x de teste: {x_test.shape}")
        print(f"Tamanho y de treino: {y_train.shape}")
        print(f"Tamanho y de teste: {y_test.shape}")
        print(" ")

        print(f"Pontuação Decision Tree: {clf_tree_score}")
        print(f"Pontuação k-Nearest Neighbors: {clf_neighbors_score}")
        print(f"Pontuação Multilayer Perceptron: {clf_neural_network_score }")
        print(f"Pontuação Naıve Bayes: {clf_naive_bayes_score}")
        print(f"Pontuação Perceptron: {clf_linear_model_score}")
        print(" ")

        print('\033[1m' + 'Modelos de classificação: Naive Bayes' + '\033[0m')
        acuracia = accuracy_score(y_test, clf_naive_bayes_predict)
        print('Acurácia: %f' % acuracia)

        precision = precision_score(y_test, clf_naive_bayes_predict)
        print('Precision: %f' % precision)

        recall = recall_score(y_test, clf_naive_bayes_predict)
        print('Recall: %f' % recall)

        f1 = f1_score(y_test, clf_naive_bayes_predict)
        print('F1-Score: %f' % f1)
        print("\n")

        print('\033[1m' + '                         Métricas de Validação' + '\033[0m')
        print(clf_naive_bayes_classification_report)
        print("")

    else:
        break
    print("\n")

print('\033[1m' + 'Você saiu' + '\033[0m')
