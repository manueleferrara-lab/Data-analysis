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

def funzione(x,Q,A,k1,k2):
    return (k1/np.sqrt(1+(Q*(x/A-A/x))**2))**k2

def funzione_db(x,Q,A,k1,k2):
    return -k2*10*np.log10(1+(Q*(x/A-A/x))**2)+k1
def arctan(x,Q,A,k):
    return -k*np.arctan(Q*(x/A-A/x))
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
    if Funzione==arctan:
        parameters,covariance = curve_fit(Funzione,valorix, valoriy, sigma=Sigmay, absolute_sigma=True, maxfev=10000, p0=[0.77,18400,0.94])
    else:
        parameters,covariance = curve_fit(Funzione,valorix, valoriy, sigma=Sigmay, absolute_sigma=True, maxfev=10000, p0=[0.77,18400,1,0.5])
    print("il valore di ", titolo, "è:", misura_arr(parameters[1], sqrt(covariance[1][1])) ," ± ", errore_arr(sqrt(covariance[1][1])))
    print("il valore di q in ", titolo, "è:", misura_arr(parameters[0], sqrt(covariance[0][0])) ," ± ", errore_arr(sqrt(covariance[0][0])))
    for i in range(2,len(parameters)):
        print("il valore di k", i-1, "è:", misura_arr(parameters[i], sqrt(covariance[i][i])) ," ± ", errore_arr(sqrt(covariance[i][i])))
    """plt.title(titolo)"""
    x_fit = np.linspace(min(valorix), max(valorix), 10000)
    if len(parameters)==4:
        plt.plot(x_fit, Funzione(x_fit, parameters[0],parameters[1],parameters[2],parameters[3]), "r")
    else:
        plt.plot(x_fit, Funzione(x_fit, parameters[0],parameters[1],parameters[2]), "r")
    plt.errorbar(valorix, valoriy, xerr = Sigmax, yerr = Sigmay, fmt='o', markersize='3')
    plt.xscale(scala_x)
    plt.yscale(scala_y)
    plt.xlabel(unit_x)
    plt.ylabel(unit_y)
    plt.show()
    chiquadro = 0
    if len(parameters)==3:
        for i in range(len(valoriy)):
            chiquadro = chiquadro + (((valoriy[i]-Funzione(valorix[i],parameters[0],parameters[1],parameters[2]))**2)/(Sigmay[i])**2)
    else:
        for i in range(len(valoriy)):
            chiquadro = chiquadro + (((valoriy[i]-Funzione(valorix[i],parameters[0],parameters[1],parameters[2],parameters[3]))**2)/(Sigmay[i])**2)
    dof = len(valoriy)-len(parameters)
    prob = chi2.cdf(chiquadro, df=dof)
    print(f'\u03C72_0 = {np.round(chiquadro, decimals=1)}, e P(\u03C72 > \u03C72_0)= {prob}')
    if prob<0.95:
      print("Test accettato", "\n")
    else:
      print("Test rifiutato","\n")
    chi2in_x = np.arange(0, 100, 0.001)
    """plt.title("\u03C72 "+ titolo)"""
    plt.plot(chi2in_x, chi2.pdf(chi2in_x, df = dof))
    plt.axvline(x=chiquadro, color = "red")
    plt.show()
    return parameters,covariance
def lettura_dati(nome_file, righe, colonne):
    dati=pd.read_excel(nome_file ,usecols=range(colonne), nrows=righe)
    errori_periodo=[]
    errori_risposta_db=[]
    errori_puls=[]
    errori_ritardo=[]
    errori_sfas=[]
    errori_risposta=[]
    for i in range(len(dati["periodo condensatore CH2(s)"])):
        errori_periodo.append(errore_periodo(dati["periodo condensatore CH2(s)"][i], dati["divisioni tempo(s)"][i]))
    for i in range(len(dati["tensione condensatore CH2 (V)"])):
        errori_risposta_db.append(10*(np.sqrt(((errore_pp(dati["tensione condensatore CH2 (V)"][i],dati["divisioni CH2 (V)"][i])/(dati["tensione condensatore CH2 (V)"][i]*log(10,e)))**2)+(errore_pp(dati["tensione pp generatore CH1 (V)"][i],dati["divisioni CH1 (V)"][i])/(dati["tensione pp generatore CH1 (V)"][i]*log(10,e)))**2)))
    for i in range(len(dati["periodo condensatore CH2(s)"])):
        errori_puls.append(2*pi*sqrt(errore_periodo(dati["periodo condensatore CH2(s)"][i], dati["divisioni tempo(s)"][i])/dati["periodo condensatore CH2(s)"][i]**2)**2)
    for i in range(len(dati["ritardo(s)"])):
        errori_ritardo.append(errore_delta_t(dati["ritardo(s)"][i], dati["divisioni ritardo(s)"][i]))
    for i in range(len(dati["ritardo(s)"])):
        errori_sfas.append(2*pi*(errori_ritardo[i]/dati["periodo condensatore CH2(s)"][i]+errori_periodo[i]*dati["ritardo(s)"][i]/(dati["periodo condensatore CH2(s)"][i]**2)))
    for i in range(len(dati["risp non db"])):
        errori_risposta.append(np.sqrt((errore_pp(dati["tensione condensatore CH2 (V)"][i],dati["divisioni CH2 (V)"][i]))**2+(dati["tensione condensatore CH2 (V)"][i]*errore_pp(dati["tensione pp generatore CH1 (V)"][i],dati["divisioni CH1 (V)"][i]/dati["divisioni CH1 (V)"][i])**2))/dati["tensione pp generatore CH1 (V)"][i])
    return dati,errori_periodo, errori_risposta_db, errori_puls, errori_ritardo, errori_sfas,errori_risposta

epsilon=10
R=47800
err_R=err_res(R)
R_gen=50
L=2
err_L=0.1*L

C=1.484*(10**-9)
err_C=err_cond(C)
print("Capacità condensatore 1:",misura_arr(C, err_C)," ± ",errore_arr(err_C))
print("induttore: ", misura_arr(L, err_L)," ± ",errore_arr(err_L))
print("resistenza: ",misura_arr(R, err_R)," ± ",errore_arr(err_R))
w0=1/(L*C)**(1/2)
err_w0=w0*(err_C**2/C**2+err_L**2/L**2)**(1/2)/2
print("Pulsazione di taglio :",misura_arr(w0,err_w0)," ± ",errore_arr(err_w0))
f0=w0/(2*pi)
err_f0=err_w0/(2*pi)
print("Frequenza di taglio :",misura_arr(f0,err_f0)," ± ",errore_arr(err_f0))
Q=(1/R)*sqrt(L/C)
err_Q=Q*sqrt((err_R/R)**2+(err_L/2/L)**2+(err_C/2/C)**2)
print("Fattore di qualità:",misura_arr(Q,err_Q)," ± ",errore_arr(err_Q))
delta_banda=w0/Q
err_delta_banda=sqrt(err_R**2+(R*err_L/L)**2)/L
print("Larghezza della banda:",misura_arr(delta_banda,err_delta_banda)," ± ",errore_arr(err_delta_banda))
delta_f0=f0/Q

dati, errori_periodo, errori_risposta_db, errori_puls, errori_ritardo, errori_sfas,errori_risposta =lettura_dati("esperienza 4.xlsx", 25, 15)
a,b=fit(funzione,dati["pulsazione"], dati["risp non db"],errori_puls, errori_risposta, "pulsazione", "1/rad","risposta in frequenza", "", "guadagno", "log")
a,b=fit(funzione_db,dati["pulsazione"], dati["risposta in frequenza"],errori_puls, errori_risposta_db, "pulsazione", "1/rad","risposta in frequenza", "dB", "guadagno", "log")
c,d=fit(arctan,dati["pulsazione"], dati["sfasamento"],errori_puls, errori_sfas, "pulsazione", "1/rad","sfasamento", "rad", "sfasamento", "log")

K=(a[3]-2.5968*log10(2))/a[2]
k=(10**(K)-1)/a[0]**2
print(K,k)
x1=a[1]*sqrt((k+2+sqrt((k+2)**2-4))/2)
x2=a[1]*sqrt((k+2-sqrt((k+2)**2-4))/2)
print(funzione_db(x1,a[0],a[1],a[2],a[3]))
print("graficamente si ottiene che la lunghezza di banda è:", x1-x2)
print("le intersezioni con il grafico avvengono in:",x1, "e",x2)
x_fit = np.linspace(90, 70000, 10000)
x1_fit = np.linspace(x2, x1, 1000)
plt.plot(x_fit, funzione_db(x_fit, a[0],a[1],a[2],a[3]), "b")
plt.plot(x1_fit,x1_fit-x1_fit-3)
plt.errorbar(dati["pulsazione"], dati["risposta in frequenza"],xerr = errori_puls, yerr = errori_risposta_db, fmt='o', markersize='3')
plt.xscale("log")
plt.xlabel("1/rad")
plt.ylabel("dB")
plt.show()

phi=arctan(c[1],c[0],c[1],c[2])
print("lo sfasamento in corrispondenza della pulsazione di taglio è:",phi)
"($\omega$)"
print("Array degli errori:")
print(errori_periodo)
print(errori_puls)
print(errori_risposta)
print(errori_risposta_db)
print(errori_ritardo)
print(errori_sfas)