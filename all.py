#%%

#    The code from line 24 on is adapted from
#     "The Invisible Hand Algorithm: Solving the Assignment Problem With Statistical Physics"
#      - by J. J. Kosowsky and A. L. Yuille

#%%

import numpy
import scipy
import matplotlib.pyplot as plt
import time
import munkres

N = 150
A = numpy.random.random((N, N))#3 - numpy.array([[2, 3, 3],
                 #[3, 2, 3],
                 #[3, 3, 2]])
#

m = munkres.Munkres()

tmp = time.time()
indices = m.compute(1.0 - A)
print time.time() - tmp

idxs, indices = zip(*indices)

#%%
tmp = time.time()
beta = 500.0

Q = numpy.zeros(N)
P = numpy.zeros(N)

for i in range(50):
    for r in range(N):
        Q[r] = scipy.misc.logsumexp(2 * beta * (A[r, :] - P)) / (2 * beta)

    for c in range(N):
        P[c] = scipy.misc.logsumexp(2 * beta * (A[:, c] - Q)) / (2 * beta)

S = numpy.array([beta * (A[r, :] - P - Q[r]) for r in range(N)])

#S = numpy.exp(S)
#print S

assignments = []
for i in range(N):
    assignments.append(numpy.argmax(S[i, :]))

print time.time() - tmp
print len(set(assignments))

print zip(indices, assignments)

#%%
P = numpy.random.random(N)#array([1.0, 1.0, 1.0])
dP = numpy.zeros(N)
S = numpy.zeros((N, N))

beta = 1000.0

sums = [scipy.misc.logsumexp(beta * (A[i, :] - P)) for i in range(N)]

for a in range(N):
    for i in range(N):
        S[i, a] = beta * A[i, a]

for i in range(1):
    for r in range(N):
        S[r, :] -= scipy.misc.logsumexp(2 * S[r, :]) / 2.0

    for c in range(N):
        S[:, c] -= scipy.misc.logsumexp(2 * S[:, c]) / 2.0

#S = numpy.exp(S)

assignments = []
for i in range(N):
    assignments.append(numpy.argmax(S[i, :]))

print len(set(assignments))

print zip(indices, assignments)

#%%
Ps = []
tmp = time.time()
for t in range(100):
    sums = [scipy.misc.logsumexp(beta * (A[i, :] - P)) for i in range(N)]

    print sums

    for a in range(N):
        for i in range(N):
            S[i, a] = beta * (A[i, a] - P[a]) - sums[i]

    S = numpy.exp(S)

    dP = numpy.sum(S, axis = 0) - 1.0

    P += dP * .1

    if max(numpy.abs(dP)) < 1e-2:
        break

    Ps.append(P.copy())

    #print dP
print "ODE: ", time.time() - tmp
print P, t

    #print i, numpy.argmax(S[i, :])

Ps = numpy.array(Ps)

plt.plot(Ps[:, 0], Ps[:, 1])
plt.show()

#%%