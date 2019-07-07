from tkinter import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris=load_iris()
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

#defining the accuracy scores of the 4 SML algos and the input variables
acc_lr,acc_knn,acc_nb,acc_dt,sep_len,sep_wid,pet_len,pet_wid=0,0,0,0,0,0,0,0

def logi_reg():
	global sep_len,sep_wid,pet_len,pet_wid,acc_lr,acc_knn,acc_nb,acc_dt
	from sklearn.linear_model import LogisticRegression
	lr=LogisticRegression(solver="liblinear",multi_class='auto')
	lr.fit(x_train,y_train)
	y_pred=lr.predict(x_test)
	print(y_test,"\n",y_pred)
	acc_lr=round(accuracy_score(y_test,y_pred)*100,2)
	print("LR:\nthe accuracy is ",acc_lr,"%")
	vAcc.set(acc_lr)
	vAns.set(lr.predict([[sep_len,sep_wid,pet_len,pet_wid]]))

def sub():
	global sep_len,sep_wid,pet_len,pet_wid
	sep_len=float(vSep_len.get())
	sep_wid=float(vSep_wid.get())
	pet_len=float(vPet_len.get())
	pet_wid=float(vPet_wid.get())
	print(sep_len,sep_wid,pet_len,pet_wid)
	

def nb():
	global sep_len,sep_wid,pet_len,pet_wid,acc_lr,acc_knn,acc_nb,acc_dt
	from sklearn.naive_bayes import GaussianNB
	nb=GaussianNB()
	nb.fit(x_train,y_train)
	y_pred=nb.predict(x_test)
	print(y_test,"\n",y_pred)
	acc_nb=round(accuracy_score(y_test,y_pred)*100,2)
	print("NB:\nthe accuracy is ",acc_nb,"%")
	vAcc.set(acc_nb)
	print(sep_len,sep_wid,pet_len,pet_wid)
	vAns.set(nb.predict([[sep_len,sep_wid,pet_len,pet_wid]]))

def knn():
	global sep_len,sep_wid,pet_len,pet_wid,acc_lr,acc_knn,acc_nb,acc_dt
	from sklearn.neighbors import KNeighborsClassifier
	K=KNeighborsClassifier(n_neighbors=5)
#train the model by using training dataset
	K.fit(x_train,y_train)
#test the model
	y_pred=K.predict(x_test)
	print(y_pred)
	print(y_test)
	acc_knn=round(accuracy_score(y_test,y_pred)*100,2)
	print("KNN:\nthe accuracy is ",acc_knn,"%")
	vAcc.set(acc_knn)
	vAns.set(K.predict([[sep_len,sep_wid,pet_len,pet_wid]]))

def dt():
	global sep_len,sep_wid,pet_len,pet_wid,acc_lr,acc_knn,acc_nb,acc_dt
	from sklearn.tree import DecisionTreeClassifier
	dt=DecisionTreeClassifier()
	dt.fit(x_train,y_train)
	y_pred=dt.predict(x_test)
	print(y_test,"\n",y_pred)
	acc_dt=round(accuracy_score(y_test,y_pred)*100,2)
	print("DT:\nthe accuracy is ",acc_dt,"%")
	vAcc.set(acc_dt)
	vAns.set(dt.predict([[sep_len,sep_wid,pet_len,pet_wid]]))

def clear():
	vSep_len.set("")
	vSep_wid.set("")
	vPet_len.set("")
	vPet_wid.set("")
	vAcc.set("")
	vAns.set("")

def compare():
	global acc_lr,acc_knn,acc_nb,acc_dt
	import matplotlib.pyplot as plt
	model=['LR','KNN','NB','DT']
	accuracy=[acc_lr,acc_knn,acc_nb,acc_dt]
	plt.title("Iris flower model")
	plt.bar(model,accuracy,color=['orange','green','red','aqua'])
	plt.xlabel("models")
	plt.ylabel("accuracy")
	plt.show()

w=Tk()
vSep_len=StringVar()
vSep_wid=StringVar()
vPet_len=StringVar()
vPet_wid=StringVar()
vAcc=StringVar()
vAns=StringVar()

# L=Frame(w,bg="red")
title=Label(w,text="IRIS FLOWER PREDICTOR",font=("Algerian",20,"bold"))
lSep_len=Label(w,text="Sepal length",font=("Consolas",20,"bold"))
lSep_wid=Label(w,text="Sepal width",font=("Consolas",20,"bold"))
lPet_len=Label(w,text="Petal length",font=("Consolas",20,"bold"))
lPet_wid=Label(w,text="Petal width",font=("Consolas",20,"bold"))
lAcc=Label(w,text="Accuracy Score",font=("Consolas",20,"bold"))
lAns=Label(w,text="Answer",font=("Consolas",20,"bold"))
# img=PhotoImage(file="F:/python/flower.gif")

bLR=Button(w,text="Logistic Regression",font=("Consolas",20,"bold"),command=logi_reg)
bKNN=Button(w,text="K-Nearest Neighbor",font=("Consolas",20,"bold"),command=knn)
bNB=Button(w,text="Naive Bayes",font=("Consolas",20,"bold"),command=nb)
bDT=Button(w,text="Decision Tree",font=("Consolas",20,"bold"),command=dt)
bSub=Button(w,text="Submit",font=("Consolas",20,"bold"),command=sub)
bRes=Button(w,text="Reset",font=("Consolas",20,"bold"),command=clear)
bCom=Button(w,text="Compare",font=("Consolas",20,"bold"),command=compare)

eAns=Entry(w,font=("Consolas",20,"bold"),textvariable=vAns)
eSep_len=Entry(w,font=("Consolas",20,"bold"),textvariable=vSep_len)
eSep_wid=Entry(w,font=("Consolas",20,"bold"),textvariable=vSep_wid)
ePet_len=Entry(w,font=("Consolas",20,"bold"),textvariable=vPet_len)
ePet_wid=Entry(w,font=("Consolas",20,"bold"),textvariable=vPet_wid)
eAcc=Entry(w,font=("Arial",20,"bold"),textvariable=vAcc)
# packing
# L.grid(row=1,column=1,rowspan=6,columnspan=5)
title.grid(row=1,column=1,columnspan=5)
bLR.grid(row=2,column=1)
bKNN.grid(row=3,column=1)
bNB.grid(row=4,column=1)
bDT.grid(row=5,column=1)
bCom.grid(row=6,column=4)
lSep_len.grid(row=2,column=2)
lSep_wid.grid(row=3,column=2)
lPet_len.grid(row=4,column=2)
lPet_wid.grid(row=5,column=2)
bSub.grid(row=6,column=2)
eSep_len.grid(row=2,column=3)
eSep_wid.grid(row=3,column=3)
ePet_len.grid(row=4,column=3)
ePet_wid.grid(row=5,column=3)
bRes.grid(row=6,column=3)
lAcc.grid(row=3,column=4)
eAcc.grid(row=3,column=5)
lAns.grid(row=4,column=4)
eAns.grid(row=4,column=5)
w.mainloop()