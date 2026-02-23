import numpy
import math
from decimal import Decimal 
import statistics
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
def arrotondamento_decimale(value, unc):
    value_str = str(value)
    unc_str = str(unc)
    if ("." not in unc_str):
        decimali_unc = 0
    else:
        parte_intera, parte_decimale_unc = unc_str.split(".")
        decimali_unc = len(parte_decimale_unc)
    if ("." not in value_str):
        decimali_value = 0
    else:
        parte_intera, parte_decimale_value = value_str.split(".")
        decimali_value = len(parte_decimale_value)
    if(decimali_value > decimali_unc):
        return float((round(value, decimali_unc)))
    elif(decimali_value < decimali_unc):
        if("." not in value_str):
            value_str = value_str + "."
        while(decimali_value != decimali_unc):
            value_str = value_str + "0"
            decimali_value = decimali_value + 1
        return (float(Decimal(value_str)))
    else:
        return (float(value))
def strumento_analogico(value, fondoscala, classe_di_precisione,arrvalue, arrerror):
    a = (classe_di_precisione/100)*fondoscala
    uncertainty=a/numpy.sqrt(3)   
    unc_rounded = round(uncertainty, -int(math.floor(math.log10(abs(uncertainty)))))
    arrvalue.append(arrotondamento_decimale(value, unc_rounded))
    arrerror.append(unc_rounded)
def strumento_digitale(value, classe_di_precisione, num_digit, arrvalue, arrerror):
    value_str=str(abs(value))
    if ("." not in value_str):
        digit = 1
    else:
        parte_intera, parte_decimale = value_str.split(".")
        digit = 10**(-len(parte_decimale))
    uncertainty=(((classe_di_precisione/100)*value) + (num_digit*digit))/numpy.sqrt(3)
    unc_rounded = round(uncertainty, -int(math.floor(math.log10(abs(uncertainty)))))
    arrvalue.append(arrotondamento_decimale(value, unc_rounded))
    arrerror.append(unc_rounded)
#GENERAZIONE VALORI
xmin = 0
xmax = 1
passo = 0.1
A=2
B=3 
arrx_singoli = numpy.arange(xmin, xmax, passo)
arrx_ripetuti = []
arry_singoli = []
arry_ripetuti = []
arry = []
for x in arrx_singoli:
    arry_singoli.append(round(A*x+B,1))
arrxvalue = []
arrxerror = []
arryvalue = []
arryerror = []
matrice = []
#MISURAZIONE DELLE X
misura_x = int(input("Premere 1 per misure singole di x, premere 2 per misure ripetute di x: "))
while misura_x != 1 and misura_x != 2:
    misura_x=int(input("Scelta non valida. Scegliere tra 1 o 2: "))
if misura_x == 1:
    strumento_x = int(input("Scegliere 1 per lo strumento analogico di x, 2 per lo strumento digitale di x: "))
    while strumento_x != 1 and strumento_x != 2:
        strumento_x=int(input("Scelta non valida. Scegliere tra 1 o 2: "))
    if strumento_x == 1:
        fondoscala=int(input("Inserire valore di fondoscala: "))
        classe_di_precisione = int(input("Inserire classe di precisione: "))
        for el in arrx_singoli:
            strumento_analogico(el, fondoscala, classe_di_precisione, arrxvalue, arrxerror)
    else:
        classe_di_precisione = int(input("Inserire classe di precisione: "))
        num_digit = int(input("Inserire numero di digit: "))
        for el in arrx_singoli:
            strumento_digitale(el, classe_di_precisione, num_digit, arrxvalue, arrxerror)
else:
    n_misurazioni_x = int(input("Quante volte misuri x? "))
    for x in arrx_singoli:
        riga = []
        for i in range(n_misurazioni_x):
            riga.append(x+numpy.random.normal(loc=0, scale=0.01))
        arrx_ripetuti.append(riga)
    for row in arrx_ripetuti:
        media = statistics.mean(row)
        dev_std = statistics.stdev(row)
        esponente = math.floor(math.log10(abs(dev_std)))
        unc_rounded = round(dev_std/10**esponente)*(10**esponente)
        value_str=str(abs(unc_rounded))
        if ("." not in value_str):
            decimali = 0
        elif float(dev_std)==math.floor(float(dev_std)):
            decimali = 0
        else:
            parte_intera, parte_decimale = value_str.split(".")
            decimali = len(parte_decimale)
        arrxvalue.append(round(media,decimali))
        arrxerror.append(unc_rounded)
#MISURAZIONE DELLE Y
misura_y = int(input("Premere 1 per misure singole di y, premere 2 per misure ripetute di y: "))
while misura_y != 1 and misura_y != 2:
    misura_y=int(input("Scelta non valida. Scegliere tra 1 o 2: "))
if misura_y == 1:
    strumento_y = int(input("Scegliere 1 per lo strumento analogico di y, 2 per lo strumento digitale di y: "))
    while strumento_y != 1 and strumento_y != 2:
        strumento_y=int(input("Scelta non valida. Scegliere tra 1 o 2: "))
    if strumento_y == 1:
        fondoscala=int(input("Inserire valore di fondoscala: "))
        classe_di_precisione = int(input("Inserire classe di precisione: "))
        for el in arry_singoli:
            strumento_analogico(el, fondoscala, classe_di_precisione, arryvalue, arryerror)
    else:
        classe_di_precisione = int(input("Inserire classe di precisione: "))
        num_digit = int(input("Inserire numero di digit: "))
        for el in arry_singoli:
            strumento_digitale(el, classe_di_precisione, num_digit, arryvalue, arryerror)
else:
    n_misurazioni_y = int(input("Quante volte misuri y? "))
    for y in arry_singoli:
        riga = []
        for i in range(n_misurazioni_y):
            riga.append(y+numpy.random.normal(loc=0, scale=0.01))
        arry_ripetuti.append(riga)
    for row in arry_ripetuti:
        media = statistics.mean(row)
        dev_std = statistics.stdev(row)
        esponente = math.floor(math.log10(abs(dev_std)))
        unc_rounded = round(dev_std/10**esponente)*(10**esponente)
        value_str=str(abs(unc_rounded))
        if ("." not in value_str):
            decimali = 0
        elif float(dev_std)==math.floor(float(dev_std)):
            decimali = 0
        else:
            parte_intera, parte_decimale = value_str.split(".")
            decimali = len(parte_decimale)
        arryvalue.append(round(media,decimali))
        arryerror.append(unc_rounded)
#GRAFICO DI PRIMA APPROSSIMAZIONE
fig, grafico = plt.subplots()
grafico.errorbar(arrxvalue, arryvalue, xerr = 0, yerr = arryerror, fmt='o', markersize='3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#TEST CHI QUADRO
def retta(x, A, B):
    y = A*x + B
    return y
parameters, covariance = curve_fit(retta, arrxvalue, arryvalue, sigma=arryerror, absolute_sigma=True)
Ap = parameters[0]
sigma_Ap = numpy.sqrt(covariance[0][0])
sigma_Ap_rounded = round(sigma_Ap, -int(math.floor(math.log10(abs(sigma_Ap)))))
value_str=str(abs(sigma_Ap_rounded))
if ("." not in value_str):
    decimali = 0
else:
    parte_intera, parte_decimale = value_str.split(".")
    decimali = len(parte_decimale)
Ap_rounded=round(Ap, decimali)
Bp = (parameters[1])
sigma_Bp = numpy.sqrt(covariance[1][1])
sigma_Bp_rounded = round(sigma_Bp, -int(math.floor(math.log10(abs(sigma_Bp)))))
value_str=str(abs(sigma_Bp_rounded))
if ("." not in value_str):
    decimali = 0
else:
    parte_intera, parte_decimale = value_str.split(".")
    decimali = len(parte_decimale)
Bp_rounded=round(Bp, decimali)
print("A=", Ap_rounded, "±", sigma_Ap_rounded)
print("B=", Bp_rounded, "±", sigma_Bp_rounded)
chiquadro = 0
for i in range(len(arrxvalue)):
    chiquadro = chiquadro + (((arryvalue[i]-Bp-Ap*arrxvalue[i])**2)/(arryerror[i])**2)
dof = len(arrxvalue)-len(parameters)
prob = chi2.cdf(chiquadro, df=dof)
print("Chiquadro: ", chiquadro)
if prob<0.95:
  print("Test accettato")
else:
  print("Test rifiutato")
chi2in_x = numpy.arange(0, 70, 0.001)
plt.plot(chi2in_x, chi2.pdf(chi2in_x, df = dof))
plt.axvline(x=chiquadro, color = "red")
plt.show()
#APPROSSIMAZIONE FINALE
arryerrorfinal = []
for i in range(len(arryerror)):
    uy=numpy.sqrt((arryerror[i]**2)+((Ap*arrxerror[i])**2))
    arryerrorfinal.append(uy)
fig, grafico = plt.subplots()
grafico.errorbar(arrxvalue, arryvalue, xerr = arrxerror, yerr = arryerrorfinal, fmt='o', markersize='3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#TEST CHI QUADRO
parameters, covariance = curve_fit(retta, arrxvalue, arryvalue, sigma=arryerrorfinal, absolute_sigma=True)
Ap2 = (parameters[0])
sigma_Ap2 = numpy.sqrt(covariance[0][0])
sigma_Ap2_rounded = round(sigma_Ap2, -int(math.floor(math.log10(abs(sigma_Ap2)))))
value_str=str(abs(sigma_Ap2_rounded))
if ("." not in value_str):
    decimali = 0
else:
    parte_intera, parte_decimale = value_str.split(".")
    decimali = len(parte_decimale)
Ap2_rounded=round(Ap2, decimali)
Bp2 = (parameters[1])
sigma_Bp2 = numpy.sqrt(covariance[1][1])
sigma_Bp2_rounded = round(sigma_Bp2, -int(math.floor(math.log10(abs(sigma_Bp2)))))
value_str=str(abs(sigma_Bp2_rounded))
if ("." not in value_str):
    decimali = 0
else:
    parte_intera, parte_decimale = value_str.split(".")
    decimali = len(parte_decimale)
Bp2_rounded=round(Bp, decimali)
print("A=", Ap2_rounded, "±", sigma_Ap2_rounded)
print("B=", Bp2_rounded, "±", sigma_Bp2_rounded)
chiquadro2 = 0
for i in range(len(arrxvalue)):
    chiquadro2 = chiquadro2 + (((arryvalue[i]-Bp-Ap*arrxvalue[i])**2)/(arryerrorfinal[i])**2)
dof = len(arrxvalue)-len(parameters)
prob = chi2.cdf(chiquadro2, df=dof)
print("Chiquadro: ", chiquadro2)
if prob<0.95:
  print("Test accettato")
else:
  print("Test rifiutato")
chi2_x=numpy.arange(0,70,0.001)
plt.plot(chi2_x, chi2.pdf(chi2_x, df = dof))
plt.axvline(x=chiquadro2, color = "red")
plt.show()