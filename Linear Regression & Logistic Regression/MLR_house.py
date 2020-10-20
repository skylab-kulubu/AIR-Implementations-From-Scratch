#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('kc_house_data.csv') # veriyi okuduk

df.drop(['id','date','zipcode'], axis = 1, inplace = True)

df = (df - np.mean(df)) / np.std(df) 

df_train = df.iloc[ :int(df.shape[0]*0.7), : ]  #verinin %70'ini train kalanını test olarak ayırdık

df_test = df.iloc[int(df.shape[0]*0.7): , :] 

x_train = df_train.iloc[:, 1:]
x_train['ones'] = 1
x_train = np.array(x_train)

y_train = np.array(df_train.iloc[:,0])
y_train.shape = (y_train.shape[0],1)    # bu satırda shape düzenlemesi yaparak (5, ) olan değişkeni (5,1) yaptık

x_test = df_test.iloc[:, 1:] #test için verimizi hazırladık
x_test['ones'] = 1
x_test = np.array(x_test)

y_test = np.array(df_test.iloc[:,0])
y_test.shape = (y_test.shape[0],1)



m = df_train.shape[0] #training sample sayısını atadık
lr = 0.2 # learning rate
iteration = 1000 # işlem tekrar sayısı
lmd = 100 # Regularization term


thetas = np.zeros(shape=(1,x_train.shape[1])) #katsayılar vektörünü oluşturduk

x_axis = np.zeros(iteration) #grafik oluştururken kullanacağımız x ekseni
loss_train = []     #loss değerlerini grafikte gözlemleyebilmek için kullanacağız
loss_test = []

def cost(x,y,thetas,lmd):       #gönderilen x y ve katsayılar için loss'u hesaplar (x_train veya x_test gibi)
    thetascpy = thetas.copy()
    thetascpy[0] = 0
    cost =(1/(2*x.shape[0]))*((x @ thetas.T - y)**2).sum() + lmd* np.sum(np.power(thetascpy,2))
    return cost


for i in range(iteration):      #Grediant Descent algoritması
    thetascpy = thetas.copy()
    thetascpy[0] = 0
    thetas = thetas - (1/m) * (lr) * ((x_train * (x_train @ thetas.T - y_train)).sum(axis = 0) + 2*lmd*thetascpy)
    loss_train.append(cost(x_train,y_train,thetas,lmd)) #her tur bulunan katsayılar ile test ve train datası üzerinde işlem yapılır
    loss_test.append(cost(x_test,y_test,thetas,lmd))
    x_axis[i] = i       #x eksenini bir yandan oluşturuyoruz
    
plt.plot(x_axis[0:20],loss_train[0:20],color='red')     #grafikleri çizdiriyoruz
plt.plot(x_axis[0:20],loss_test[0:20])

# plt.scatter(y_test, x_test @ thetas.T)
