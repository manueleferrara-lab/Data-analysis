from math import pi,sqrt,log10,log,e, floor
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import numpy as np
def errore_pp(valore_pp, div_vert):
    return (3/100)*valore_pp+0.05*div_vert
def errore_periodo(valore_sfas, div_oriz):
    return div_oriz/250 + 0.0001*valore_sfas + 0.0000000004
def errore_delta_t(ritardo, div_oriz):
    return sqrt((1/3)*(errore_periodo(ritardo, div_oriz))**2+(0.2*div_oriz)**2)
def funzione(x,A):
    return -10*np.log10(1+(x/A)**2)
def arctan(x,A):
    return np.arctan(x/A)
def funzione_2(x,A):
    return -20*np.log10(1+(x/A)**2)
def arctan2(x,A):
    return 2*np.arctan(x/A)
def err_res(R):
    if R>200:
        return (0.008*R+0.001)/3**(1/2)
    else:
        return (0.008*R+0.003)/3**(1/2)
def err_cond(C):
    return 0.04*C+3*10**round(log10(C/1000))  #forse 4% lettura piu 3 digit
def errore_arr(n):
    exp=floor(log10(n))
    a=n/10**exp
    if floor(a)!=floor(a+0.5):
        a+=1
    nn=floor(a)*10**exp
    if exp<-1:
        nn=round(floor(a)*10**exp,abs(exp))
    return nn
def misura_arr(mis, err):
    dec=floor(log10(err))
    a=mis/10**dec
    if floor(a)!=floor(a+0.5):
        a+=1
    nn=floor(a)*10**dec
    if dec<-1:
        nn=round(floor(a)*10**dec,abs(dec))
    return nn
def fit(Funzione, valorix, valoriy, Sigmax=None, Sigmay=None, nome_x=None, unit_x=None, nome_y=None, unit_y=None, titolo=None, scala_x="linear", scala_y="linear"):
    parameters,covariance = curve_fit(Funzione,valorix, valoriy, sigma=Sigmay, absolute_sigma=True)
    print("il valore di", titolo, "è:", misura_arr(parameters[0], sqrt(covariance[0][0])) ," ± ", errore_arr(sqrt(covariance[0][0])))
    plt.plot(valorix, Funzione(valorix, parameters[0]), "r")
    plt.errorbar(valorix, valoriy, xerr = Sigmax, yerr = Sigmay, fmt='o', markersize='3')
    plt.xscale(scala_x)
    plt.yscale(scala_y)
    plt.xlabel(nome_x)
    plt.ylabel(nome_y)
    plt.show()
    chiquadro = 0
    for i in range(len(valoriy)):
        chiquadro = chiquadro + (((valoriy[i]-Funzione(valorix[i],parameters))**2)/(Sigmay[i])**2)
    dof = len(valoriy)-len(parameters)
    prob = chi2.cdf(chiquadro, df=dof)
    print(f'\u03C72_0 = {np.round(chiquadro, decimals=1)}, e P(\u03C72 > \u03C72_0)= {prob}')
    if prob<0.95:
      print("Test accettato")
    else:
      print("Test rifiutato")
    chi2in_x = np.arange(0, 100, 0.001)
    #plt.title("\u03C72 "+ titolo)
    plt.plot(chi2in_x, chi2.pdf(chi2in_x, df = dof))
    plt.axvline(x=chiquadro, color = "red")
    plt.show()
def arr_lettura_(n):
    exp=floor(log10(n)-1)
    a=n/10**exp
    if floor(a)!=floor(a+0.5):
        a+=1
    nn=floor(a)*10**exp
    if exp<-1:
        nn=round(floor(a)*10**exp,abs(exp))
    return nn
def lettura_dati(nome_file, righe, colonne):
    dati=pd.read_excel(nome_file ,usecols=range(colonne), nrows=righe)
    errori_periodo=[]
    errori_risposta=[]
    errori_puls=[]
    errori_ritardo=[]
    errori_sfas=[]
    for i in range(len(dati["periodo condensatore CH2(s)"])):
        errori_periodo.append(errore_periodo(arr_lettura_(dati["periodo condensatore CH2(s)"][i]), arr_lettura_(dati["divisioni tempo(s)"][i])))
    for i in range(len(dati["tensione condensatore CH2 (V)"])):
        errori_risposta.append(10*(sqrt((errore_pp(arr_lettura_(dati["tensione condensatore CH2 (V)"][i]),arr_lettura_(dati["divisioni CH2 (V)"][i]))/(arr_lettura_(dati["tensione condensatore CH2 (V)"][i])*log(10,e)))**2)+(errore_pp(arr_lettura_(dati["tensione pp generatore CH1 (V)"][i]),arr_lettura_(dati["divisioni CH1 (V)"][i]))/(arr_lettura_(dati["tensione pp generatore CH1 (V)"][i])*log(10,e)))**2))
    for i in range(len(dati["periodo condensatore CH2(s)"])):
        errori_puls.append(2*pi*sqrt(errore_periodo(arr_lettura_(dati["periodo condensatore CH2(s)"][i]), arr_lettura_(dati["divisioni tempo(s)"][i]))/arr_lettura_(dati["periodo condensatore CH2(s)"][i])**2)**2)
    for i in range(len(dati["ritardo(s)"])):
        errori_ritardo.append(errore_delta_t(arr_lettura_(dati["ritardo(s)"][i]), arr_lettura_(dati["divisioni ritardo(s)"][i])))
    for i in range(len(dati["ritardo(s)"])):
        errori_sfas.append(2*pi*(errori_ritardo[i]/arr_lettura_(dati["periodo condensatore CH2(s)"][i])+errori_periodo[i]*arr_lettura_(dati["ritardo(s)"][i])/(arr_lettura_(dati["periodo condensatore CH2(s)"][i])**2)))
    return dati,errori_periodo, errori_risposta, errori_puls, errori_ritardo, errori_sfas

epsilon=10
R1=989
err_R1=err_res(R1)
R2=9960
err_R2=err_res(R2)
R_gen=50

C1=3.25*(10**-8)
err_C1=err_cond(C1)
print("Capacità condensatore 1:",misura_arr(C1, err_C1)," ± ",errore_arr(err_C1))
print("resistenza 1:",misura_arr(R1, err_R1)," ± ",errore_arr(err_R1))
C2=3.45*(10**-9)
err_C2=err_cond(C2)
print("Capacità condensatore 2:",misura_arr(C2, err_C2)," ± ",errore_arr(err_C2))
print("resistenza 1:",misura_arr(R2, err_R2)," ± ",errore_arr(err_R2))
w0_1=1/(R1*C1)
err_w0_1=w0_1*(err_C1/C1+err_R1/R1)
print("Pulsazione di taglio 1° stadio:",misura_arr(w0_1,err_w0_1)," ± ",errore_arr(err_w0_1))
f0_1=w0_1/(2*pi)
err_f0_1=err_w0_1/(2*pi)
print("Frequenza di taglio 1° stadio:",misura_arr(f0_1,err_f0_1)," ± ",errore_arr(err_f0_1))
w0_2=1/(R2*C2)
err_w0_2=w0_2*(err_C2/C2+err_R2/R2)
print("Pulsazione di taglio 2° stadio:",misura_arr(w0_2,err_w0_2)," ± ",errore_arr(err_w0_2))
f0_2=w0_2/(2*pi)
err_f0_2=err_w0_2/(2*pi)
print("Frequenza di taglio 2° stadio:",misura_arr(f0_2,err_f0_2)," ± ",errore_arr(err_f0_2))
tau_1=1/w0_1
err_tau_1=err_w0_1/(w0_1**2)
tau_2=1/w0_2
err_tau_2=err_w0_2/(w0_2**2)

print("\nPrimo stadio: \n")
dati, errori_periodo, errori_risposta, errori_puls, errori_ritardo, errori_sfas =lettura_dati("esperienza 3 primo stadio primo giorno.xlsx", 42, 14)
fit(funzione,dati["pulsazione"], dati["risposta in frequenza"],errori_puls, errori_risposta, "1/rad", "($\omega$)","dB", "(dB)", "pulsazione di taglio della guadagno", "log")
fit(arctan,dati["pulsazione"], dati["sfasamento"],errori_puls, errori_sfas, "1/rad", "($\omega$)","rad", "(rad)", "pulsazione di taglio dello sfasamento", "log")
print("array degli errori 1")
print(errori_periodo)
print(errori_risposta)
print(errori_puls)
print(errori_ritardo)
print(errori_sfas)
print(len(errori_periodo))
print("\nSecondo stadio: \n")
dati, errori_periodo, errori_risposta, errori_puls, errori_ritardo, errori_sfas =lettura_dati("esperienza 3 secondo stadio.xlsx", 35, 20)
fit(funzione_2,dati["pulsazione"], dati["risposta in frequenza"],errori_puls, errori_risposta, "1/rad", "($\omega$)","dB", "(dB)", "guadagno", "log")
fit(arctan2,dati["pulsazione"], dati["sfasamento"],errori_puls, errori_sfas, "1/rad", "($\omega$)","rad", "(rad)", "pulsazione di taglio dello sfasamento", "log")
print("array degli errori 2")
print(errori_periodo)
print(errori_risposta)
print(errori_puls)
print(errori_ritardo)
print(errori_sfas)
print(len(errori_periodo))