import math
import matplotlib.pyplot as plt
import numpy as np
def err_res(R):
    if R>200:
        return (0.008*R+0.001)/3**(1/2)
    else:
        return (0.008*R+0.003)/3**(1/2)
def err_R3(R):
    if R>200:
        return (0.01025*R+0.001)/3**(1/2)
    else:
        return (0.01025*R+0.003)/3**(1/2)
def errore_arr(n):
    exp=math.floor(math.log10(n))
    a=n/10**exp
    if math.floor(a)!=math.floor(a+0.5):
        a+=1
    nn=math.floor(a)*10**exp
    if exp<-1:
        nn=round(math.floor(a)*10**exp,abs(exp))
    return nn
def misura_arr(mis, err):
    dec=math.floor(math.log10(err))
    a=mis/10**dec
    if math.floor(a)!=math.floor(a+0.5):
        a+=1
    nn=math.floor(a)*10**dec
    if dec<-1:
        nn=round(math.floor(a)*10**dec,abs(dec))
    return nn
def  sensibilità(R0,R1,R2,R3,R4,R5,e0):
        G0 = 1/R0
        G1 = 1/R1
        G2 = 1/R2
        G3 = 1/R3
        G4 = 1/R4
        G5 = 1/R5
        det = G0*G1*G3+G0*G1*G4+G0*G2*G3+G0*G2*G4+G0*G5*G1+G0*G5*G2+G0*G5*G3+G0*G5*G4+G1*G2*G3+G1*G2*G4+G1*G5*G2+G4*G5*G1+G2*G5*G3+G1*G3*G4+G2*G3*G4+G3*G4*G5
        sens = G0*G1*G4*G4*e0/det #risulta così perchè all'equilibrio G1G4=G2G3, G4^2 si ottiene dalla derivata di g4 rispetto a r4
        return sens
def stampa(oggetto, misura, errore):
    mis_str=str(misura)
    if "." in mis_str:
      zeri=len(str(errore).split(".")[1])
      if len(str(misura).split(".")[1])<zeri:
        mis_str=mis_str+"0"*(zeri-len(str(misura).split(".")[1]))
    print("il valore di", oggetto ,"è: "+ mis_str, " ± ", errore)
def misura(voltaggio,R1,R2,R3b,R3s,R5,tens_sbil):
    R1_err=err_res(R1)
    R2_err=err_res(R2)
    R3b_err=err_R3(R3b)*1.0025
    R4=R2*R3b/R1
    delta_R4=(R2/R1)*abs(R3s-R3b)*delta_V/tens_sbil
    us=delta_R4/3**1/2
    u4=(R2_err**2)*(R3b/R1)**2+(R1_err**2)*(R2*R3b/R1**2)**2+(R3b_err**2)*(R2/R1)**2+us**2
    u4_round=errore_arr(u4)
    R4_round=misura_arr(R4, u4)
    stampa("R4",R4_round,u4_round)
    sens=sensibilità(R0, R1, R2, R3b, R4, R5, voltaggio)
    stampa("sensibilità", misura_arr(sens, us), errore_arr(us))
def grafici(R0,R1,R2,R3,R5,e0):
    N_punti = 100000
    a_min = 0.5
    a_max = 2.
    a = np.linspace(a_min, a_max, N_punti)
    R2=a*R1
    R4=a*R3

    plt.title(f'Sensibilità (caso ideale: R0 nulla e R5 infinita) con $\epsilon$={e0} V')
    S_ideale = e0*R3/(R3+R4)**2
    us_rel_ideale = delta_V/(S_ideale*R4)
    print('A min =', a[np.argmin(us_rel_ideale)], min(us_rel_ideale))
    plt.grid(axis='x', color='0.9')
    plt.plot(a, us_rel_ideale)
    plt.xlabel('A')
    plt.ylabel('us_rel_ideale')
    plt.show()

    R0_arr = [0.1, 0.5, 1., 2., 4., 10.]
    plt.title(f'Sensibilità (al variare di R0, R5={R5} $\Omega$, $\epsilon$={e0} V)')
    plt.plot(a, us_rel_ideale, label='ideale')
    for R0 in R0_arr:
        S = sensibilità(R0,R1,R2,R3,R4,R5,e0)
        us_rel = delta_V/(S*R4)
        plt.plot(a, us_rel, label='realistica R0={}'.format(R0))
        print('R0=',R0, 'A=', a[np.argmin(us_rel)], min(us_rel))
    plt.grid(axis='x', color='0.9')
    plt.xlabel('A')
    plt.ylabel('us_rel')
    plt.legend()
    plt.show()

    R5_arr = [100., 1000., 10000., 10**5., 10**6., 10**7.]

    plt.title(f'Sensibilità (al variare di R5, R0={R0} $\Omega$, e0={e0} V)')
    plt.plot(a, us_rel_ideale, label='ideale')
    for R5 in R5_arr:
        S = sensibilità(R0,R1,R2,R3,R4,R5,e0)
        us_rel = delta_V/(S*R4)
        plt.plot(a, us_rel, label='realistica R5={}'.format(R5))
        print('R5=',R5, 'A=', a[np.argmin(us_rel)], min(us_rel))
    plt.grid(axis='x', color='0.9')
    plt.xlabel('A')
    plt.ylabel('us_rel')
    plt.legend()
    plt.show()

    e0_arr = [1., 3., 6., 10., 20.]
    plt.title(f'Sensibilità (al variare di $\epsilon$,  R0={R0} $\Omega$, R5={R5} $\Omega$)')
    plt.plot(a, us_rel_ideale, label='ideale')
    for e0 in e0_arr:
        S = sensibilità(R0,R1,R2,R3,R4,R5,e0)
        us_rel = delta_V/(S*R4)
        plt.plot(a, us_rel, label='realistica $\epsilon$={}'.format(e0))
        print('\u03B5=',e0, 'A=', a[np.argmin(us_rel)], min(us_rel))
    plt.grid(axis='x', color='0.9')
    plt.xlabel('A')
    plt.ylabel('us_rel')
    plt.legend()
    plt.show()



delta_V=0.0001
Rvoltmetro=1000000
R0=0.1
R1=199.1
R2=199.0
R2_primo=1194

R3_bil3=265
tensione_sbil3=0.05
R3_sbil3=248

R3_bil6=265
tensione_sbil6=0.0501
R3_sbil6=256

R3_bil9=264
tensione_sbil9=0.05
R3_sbil9=265

grafici(R0,R1,R2,R3_bil3,Rvoltmetro,3)
print("Resistenza R2")
print("\n Misura con 3V")
V3=misura(3,R1,R2,R3_bil3,R3_sbil3,Rvoltmetro,tensione_sbil3)
print("\n Misura con 6V")
V6=misura(6,R1,R2,R3_bil6,R3_sbil6,Rvoltmetro,tensione_sbil6)
print("\n Misura con 9V")
V9=misura(9,R1,R2,R3_bil9,R3_sbil9,Rvoltmetro,tensione_sbil9)

R3_bil3_primo=297
tensione_sbil3_primo=0.05
R3_sbil3_primo=258

R3_bil6_primo=298
tensione_sbil6_primo=0.05
R3_sbil6_primo=278

R3_bil9_primo=298
tensione_sbil9_primo=0.05
R3_sbil9_primo=285

print("\n Resistenza R2'")
print("\n Misura con 3V")
V3_primo=misura(3,R1,R2_primo,R3_bil3_primo,R3_sbil3_primo,Rvoltmetro,tensione_sbil3)
print("\n Misura con 6V")
V6_primo=misura(6,R1,R2_primo,R3_bil6_primo,R3_sbil6_primo,Rvoltmetro,tensione_sbil6)
print("\n Misura con 9V")
V9_primo=misura(9,R1,R2_primo,R3_bil9_primo,R3_sbil9_primo,Rvoltmetro,tensione_sbil9)

# R41 misurata con l'ohmetro R41=264 portata: 2000 OHM
resistenza_1=[199.3,199.1,198.7] #i diversi valori sono misurati con voltmetri diversi per rilevare eventuali correlazioni tra le misurazioni
portata_1=[200]
resistenza_2=[199.4,198.8,198.8]
portata_2=[200]
#3V
resistenza_3_b_3V=[265]
portata_3_b=[2000]
resistenza_3_s_3V=[248,248,248]
tensione_s_3V=[0.05] #V
portata_3_s=[2000]
#6V
resistenza_3_b_6V=[265] #0.0mV
tensione_s_6V=[50.1] #mV
resistenza_3_s_6V=[256]
#9V
resistenza_3_b_9V=[264] #tensione di bilanciamento = 0.3 mV
resistenza_3_s_9V=[265]
tensione_s_9V=[0.05]

#sostituzione di R2 e R41 con R'2 e R42 = 1790

resistenza_2_primo=[1194]
portata_2_primo=[2000] #ohm
#3V
resistenza_3_b_3V_primo=[297]
portata_3_b_primo=[2000] #ohm
resistenza_3_s_3V_primo=[258]
tensione_s_3V_primo=[50.1] #mV
portata_3_s_primo=[2000]
#6V
resistenza_3_b_6V_primo=[298] #00.1 mV
tensione_s_6V_primo=[50.0] #mV
resistenza_3_s_6V_primo=[278]
#9V
resistenza_3_b_9V_primo=[298] #tensione di bilanciamento = 0.2 mV
resistenza_3_s_9V_primo=[285]
tensione_s_9V_primo=[50.1] #mV