from numpy import random
f = 10.1
data = [12.6, 15.6, 18.6, 21.6, 24.6, 27.6, 30.6, 33.6, 36.6, 39.6, 42.6, 45.6]
q = []
for p in data:
    q.append((p*f)/(p-f))
print(q)
corr = []
for i in range(len(data)):
    corr.append(random.randint(10)/10)
print(corr)
final = []
for i in range(len(data)):
    final.append(round(q[i]+corr[i],1))
print(final)