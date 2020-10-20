def det(l): #determinant function
    n=len(l)
    if (n>2):
        i=1
        t=0
        sum=0
        while t<=n-1:
            d={}
            t1=1
            while t1<=n-1:
                m=0
                d[t1]=[]
                while m<=n-1:
                    if (m==t):
                        u=0
                    else:
                        d[t1].append(l[t1][m])
                    m+=1
                t1+=1
            l1=[d[x] for x in d]
            sum=sum+i*(l[0][t])*(det(l1))
            i=i*(-1)
            t+=1
        return sum
    else:
        return (l[0][0]*l[1][1]-l[0][1]*l[1][0])
import numpy as np

train_sample_number = int(input("sample number : "))

X_train = []
Y_train = []

for i in range(train_sample_number):
    X_train.append(float(input("Enter train x values : ")))
    Y_train.append(float(input("Enter train y values : ")))

degree = int(input("polynom degree : "))
matris = np.full((degree+1,degree+1),0).astype("float")

for i in range(degree+1):

    for j in range(degree+1):

        sum_x = 0
        for k in range(train_sample_number):
            sum_x += X_train[k]**(i+j)
        matris[i][j] = sum_x

print(matris)
sonuc = []
for i in range(degree+1):
    sum = 0
    for j in range(train_sample_number):
        sum = sum +  (Y_train[j]*(X_train[j]**i))
    sonuc.append(sum)

array = np.array(sonuc)
array.reshape(-1,1)

delta = det(matris)
dete = []

x = np.full((degree+1,degree+1),0).astype("float")

for i in range(degree+1):
    for j in range(degree+1):
        for k in range(degree+1):
            x[j][k] = matris[j][k]
    print(x)
    for j in range(degree+1):
        x[j][i] = array[j]
    print(det(x))
    dete.append(det(x)/delta)

prediction = float(input("Prediction Value : "))

result = 0
for i in range(degree+1):
    result = result + dete[i]*(prediction**i)

print(result)
