#KAGGLE TITANIC

#Changer le répertoire parent
import os
rep_cour = os.getcwd()
print (rep_cour)

from os import chdir
chdir("C:\Documents\S621802\Etudes\Data Science")

#Import
import numpy as np
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pydot
from sklearn.externals.six import StringIO  


#Import des bases de données
data=pd.read_csv("train.csv")
data

test=pd.read_csv("test.csv")
test

#Statistiques descriptives
data.head()
data.tail()
data.describe()
data.info()

    #Par catégories
data["Survived"]=pd.Categorical(data["Survived"],ordered=False)
data["Pclass"]=pd.Categorical(data["Pclass"],ordered=False)
data["Sex"]=pd.Categorical(data["Sex"],ordered=False)
data["Embarked"]=pd.Categorical(data["Embarked"],ordered=False)

data['Age']=data['Age'].fillna(data['Age'].median())
data['Embarked']=data['Embarked'].fillna("S")

    #Suppression de colonnes
colonnes = ['PassengerId','Name','Ticket','Cabin']
data = data.drop(colonnes,axis=1)

data_ind=pd.get_dummies(data[["Pclass","Sex","Embarked"]])
data_ind.info()
colonnes = ['Pclass_2','Pclass_3','Sex_male','Embarked_Q','Embarked_S',]
data_ind = data_ind.drop(colonnes,axis=1)

data_ind_2=data[["Survived","Age","SibSp","Parch","Fare"]]
data_ind_2=data[["Survived","Age","SibSp","Parch","Fare"]]

data_tree=pd.concat([data_ind,data_ind_2],axis=1)
data_tree.info()
    #Nombre de survivants sur le titanic
data.Survived.value_counts().plot(kind='bar')
plt.title("Distribution des Survivants, (1 = Survivant)")

    #Nombre de survivants sur le titanic par variables
stat = pd.crosstab(data.Sex, data.Survived)
stat = pd.crosstab(data.Pclass, data.Survived)
stat = pd.crosstab(data.SibSp, data.Survived)
stat = pd.crosstab(data.Parch, data.Survived)
stat = pd.crosstab(data.Pclass,data.Fare)
stat = pd.crosstab([data.Pclass, data.Sex], data.Survived)
stat.plot(kind='bar')     

    #Âge moyen
data['Age'].mean()

    #Distribution de l'âge par classe
data.Age[data.Pclass == 1].plot(kind='kde')    
data.Age[data.Pclass == 2].plot(kind='kde')
data.Age[data.Pclass == 3].plot(kind='kde')
                 # Titre des axes et du graphique
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
                # Légende
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

    # Bases d'apprentissage et de test
data_t=data_tree.drop(["Survived"],axis=1)
surv_t=data_tree["Survived"]
data_train,data_test,surv_train,surv_test=train_test_split(data_t,surv_t,test_size=0.3)

surv_test.value_counts()



#ARBRES DE DECISIONS

#Version classique
tree=DecisionTreeClassifier()
tree_class=tree.fit(data_train,surv_train)
    # Estimation de l’erreur de prévision
1-tree_class.score(data_test,surv_test)

#Version avec spécification d'options
tree=DecisionTreeClassifier(max_depth=3)
tree_opt=tree.fit(data_train,surv_train)
    # Estimation de l’erreur de prévision
1-tree_opt.score(data_test,surv_test)

#Affichage de l'arbre
#WARNING ne marche pas
dot_data = StringIO() 
export_graphviz(tree_opt, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("arbre1.pdf") 

# prévision de l’échantillon test
pred = tree_opt.predict(data_test)
# matrice de confusion
table=pd.crosstab(surv_test,pred)
table



#REGRESSION LOGISTIQUE

data_train.info()
surv_train.info()
logit = LogisticRegression()
logit_class=logit.fit(data_train, surv_train)
# Erreur
logit_class.score(data_train, surv_train)
logit_class.score(data_test, surv_test)

#Equation coefficient and Intercept
print('Coefficient: \n', logit_class.coef_)
print('Intercept: \n', logit_class.intercept_)
#Predict Output
pred= logit_class.predict(data_train)
pred

table=pd.crosstab(surv_train,pred)
table