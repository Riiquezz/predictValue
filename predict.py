#/usr/bin/python3
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

datas = []
precos = []

def ArquivoCSV(nomeArquivo):
    with open(nomeArquivo, 'r') as CSVArchive:
        lendoCSV = csv.reader(CSVArchive)
        next(lendoCSV)
        for row in lendoCSV:
            datas.append(int(row[0].split('-')[2]))
            precos.append(float(row[1]))
    return

def predict_prices(datas, precos, x):
    datas = np.reshape(datas, (len(datas), 1))
    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(datas, precos)
    svr_poly.fit(datas, precos)
    svr_rbf.fit(datas, precos)

    plt.scatter(datas, precos, color='black', label='Data')
    plt.plot(datas, svr_rbf.predict(datas), color='red', label='Modelo RBF')
    plt.plot(datas, svr_lin.predict(datas), color='green', label='Modelo Linear')
    plt.plot(datas, svr_poly.predict(datas), color='blue', label='Modelo Polinomial')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Support Vector Regression - SVR')
    plt.legend()
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
ArquivoCSV('FB.csv')
predicted_price = predict_prices(datas, precos, [[29]])
print(predicted_price)