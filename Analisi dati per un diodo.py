import numpy
import math
from decimal import Decimal
import statistics
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
arrvalue_Amp = []
arrerror_Amp = []
corrente_D=[2,8,40,220,340,3000,3900,4600,19000,21000,25000]#micro A fino al terzo, mmA in poi
fondoscala_Amp_D=[50,50,50,500,500,5000,5000,5000,50000,50000,50000] # micro A fino al secondo, mmA in poi
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
for i in range(len(corrente_D)):
    strumento_analogico(corrente_D[i], fondoscala_Amp_D[i], 1, arrvalue_Amp, arrerror_Amp)
print(arrvalue_Amp)
print(arrerror_Amp)

voltaggio_D=[1.627,1.680,1.735,1.798,1.812,1.936,1.961,1.980,2.26,2.30,2.37] #V
fondoscala_Volt_D=[2,2,2,2,2,2,2,2,20,20,20] #V
arrvalue_Volt=[]
arrerror_Volt=[]
def strumento_digitale(value, classe_di_precisione, num_digit, arrvalue, arrerror):
    value_str=str(abs(value))
    if ("." not in value_str):
        digit = 1
    else:
        parte_intera, parte_decimale = value_str.split(".")
        digit = 10**(-len(parte_decimale))
    uncertainty=(((classe_di_precisione/100)*abs(value)) + (num_digit*digit))/numpy.sqrt(3)
    unc_rounded = round(uncertainty, -int(math.floor(math.log10(abs(uncertainty)))))
    arrvalue.append(arrotondamento_decimale(value, unc_rounded))
    arrerror.append(unc_rounded)
for i in range(len(voltaggio_D)):
    strumento_digitale(voltaggio_D[i], 0.8, 2, arrvalue_Volt, arrerror_Volt)
print(arrvalue_Volt)
print(arrerror_Volt)
fig, grafico = plt.subplots()
grafico.errorbar(arrvalue_Volt, arrvalue_Amp, xerr = 0, yerr = arrerror_Amp, fmt='o', markersize='3')
plt.xlabel('Volt')
plt.ylabel('microAmpere')
plt.show()
def esponenziale(x, A, B):
    return A*(math.e**(x/B)-1)
parameters, covariance = curve_fit(esponenziale, arrvalue_Volt,arrvalue_Amp, sigma=arrerror_Amp, absolute_sigma=True)
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
for i in range(len(arrvalue_Volt)):
    chiquadro = chiquadro + (((arrvalue_Amp[i]-esponenziale(i,Ap,Bp))**2)/(arrerror_Amp[i])**2)
dof = len(arrvalue_Amp)-len(parameters)
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
arrerror_Amp_final=[]
for i in range(len(arrerror_Volt)):
    uy=numpy.sqrt((arrerror_Amp[i]**2)+((Ap/Bp*(math.e**(arrvalue_Volt[i]/Bp))*arrerror_Volt[i])**2))
    arrerror_Amp_final.append(uy)
fig, grafico = plt.subplots()
grafico.errorbar(arrvalue_Volt, arrvalue_Amp, xerr =arrerror_Volt , yerr = arrerror_Amp_final, fmt='o', markersize='3')
plt.xlabel('V')
plt.ylabel('μA')
plt.show()
parameters, covariance = curve_fit(esponenziale,arrvalue_Volt , arrvalue_Amp, sigma=arrerror_Amp_final, absolute_sigma=True)
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
for i in range(len(arrvalue_Volt)):
    chiquadro2 = chiquadro2 + (((arrvalue_Amp[i]-esponenziale(i,Ap,Bp))**2)/(arrerror_Amp_final[i])**2)
dof = len(arrvalue_Volt)-len(parameters)
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
#GRAFICO FINALE
punti_retta = numpy.arange(1,2.5,0.01)
plt.plot(punti_retta,esponenziale(punti_retta,Ap,Bp),"r-")
plt.errorbar(arrvalue_Volt, arrvalue_Amp, xerr = arrerror_Volt, yerr = arrerror_Amp_final, fmt='o', markersize='3')
plt.xlabel('V')
plt.ylabel('μA')
plt.show()
print(Ap2)
print(arrvalue_Amp)
print(arrvalue_Volt)