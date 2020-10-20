#Enes Sadi Uysal   (contributor : Şafak Bilici)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('insurance.csv') # veriyi okuduk 

df.drop('region', axis = 1, inplace = True) # gereksiz sütunu attık

def sex(each):
    if each == 'male':
        return 0
    else:
        return 1

df['sex'] = df['sex'].apply(sex)    #string olan binary değişkenleri 1 ve 0 olarak değiştirdik

def smoker(each):
    if each == 'yes':
        return 1
    else:
        return 0

df['smoker'] = df['smoker'].apply(smoker)   

df = (df - np.mean(df)) / np.std(df)  #veriyi normalize ettik ki overflow error almayalım

df_train = df.iloc[ :int(df.shape[0]*0.7), : ]  #verinin %70'ini train kalanını test olarak ayırdık

df_test = df.iloc[int(df.shape[0]*0.7): , :]  #shape[0] satır sayısını #shape[1] sütun sayısını verir

x_train = df_train.iloc[:, :-1]
x_train['ones'] = 1
x_train = np.array(x_train)

y_train = np.array(df_train.iloc[:,-1])
y_train.shape = (y_train.shape[0],1)    # bu satırda shape düzenlemesi yaparak (5, ) olan değişkeni (5,1) yaptık

x_test = df_test.iloc[:, :-1] #test için verimizi hazırladık
x_test['ones'] = 1
x_test = np.array(x_test)

y_test = np.array(df_test.iloc[:,-1])
y_test.shape = (y_test.shape[0],1)

m = df_train.shape[0] #training sample sayısını atadık
lr = 0.2 # learning rate
iteration = 1000 # işlem tekrar sayısı

thetas = np.zeros(shape=(1,x_train.shape[1])) #katsayılar vektörünü oluşturduk

x_axis = np.zeros(iteration) #grafik oluştururken kullanacağımız x ekseni
loss_train = []     #loss değerlerini grafikte gözlemleyebilmek için kullanacağız
loss_test = []

def cost(x,y,thetas):       #gönderilen x y ve katsayılar için loss'u hesaplar (x_train veya x_test gibi)
    cost =(1/(2*x.shape[0]))*((x @ thetas.T - y)**2).sum()
    return cost


for i in range(iteration):      #Gradient Descent Algoritması
    thetas = thetas - (1/m) * (lr) * (x_train * (x_train @ thetas.T - y_train)).sum(axis = 0)
    loss_train.append(cost(x_train,y_train,thetas)) #her tur, bulunan katsayılar ile test ve train datası üzerinde işlem yapılır
    loss_test.append(cost(x_test,y_test,thetas))
    x_axis[i] = i       #Grafik için x ekseni oluşturulur.
    
plt.plot(x_axis[0:20],loss_train[0:20],color='red')     #grafikleri çizdiriyoruz
plt.plot(x_axis[0:20],loss_test[0:20])      
