#%%

#    The code from line 24 on is adapted from
#     "The Invisible Hand Algorithm: Solving the Assignment Problem With Statistical Physics"
#      - by J. J. Kosowsky and A. L. Yuille

#%%

import numpy
import scipy
import time
import munkres

N = 100
A = numpy.random.random((N, N))

m = munkres.Munkres()

tmp = time.time()
indices = m.compute(1.0 - A)
munktime = time.time() - tmp

order, indices = zip(*indices)

#%% Invisible handle algo starts here
tmp = time.time()
beta = 250.0

Q = numpy.zeros(N)
P = numpy.zeros(N)

for i in range(50):
    for r in range(N):
        Q[r] = scipy.misc.logsumexp(2 * beta * (A[r, :] - P)) / (2 * beta)

    for c in range(N):
        P[c] = scipy.misc.logsumexp(2 * beta * (A[:, c] - Q)) / (2 * beta)

S = numpy.array([beta * (A[r, :] - P - Q[r]) for r in range(N)])

assignments = []
for i in range(N):
    assignments.append(numpy.argmax(S[i, :]))

approxTime = time.time() - tmp

print "Munkres took : ", munktime
print "Approximation took : ", approxTime
print "Approximation made {0} errors".format(sum([0 if a == b else 1 for a, b in zip(indices, assignments)]))
