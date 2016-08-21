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
beta = 10.0 / max(numpy.abs(A.flatten()))
maxBeta = 2 * N * numpy.sqrt(N)

Q = numpy.zeros(N)
P = numpy.zeros(N)

for i in range(1000):
    # Normalize over rows of diagonal_matrix(exp(-beta * Q)) * exp(beta * A) * diagonal_matrix(exp(-beta * P))
    for r in range(N):
        Q[r] = scipy.misc.logsumexp(beta * (A[r, :] - P)) / (beta)

    # Normalize over columns
    for c in range(N):
        P[c] = scipy.misc.logsumexp(beta * (A[:, c] - Q)) / (beta)

    # Compute S. Gotta verify this is doubly stochastic
    logS = numpy.array([beta * (A[r, :] - P - Q[r]) for r in range(N)])

    rowSums = numpy.exp([scipy.misc.logsumexp(logS[r, :]) for r in range(N)])

    # Check how different the sum of each row of S is from 1.0
    #  Only gotta check rows -- the last thing we did was normalize sum over
    #  colums so that should be okay

    if max(numpy.abs(rowSums - 1.0)) < 1e-1: # 1e-1 to 1e-2 seem to be the numbers
        print "Tolerance met at iteration: {0}".format(i)

        # DEBUG INFO -- uncomment this if you want to watch how well it's doing

        #S = numpy.exp(logS)
        #assignments = []
        #for j in range(N):
        #    assignments.append(numpy.argmax(S[j, :]))
        #print "There are {0} errors".format(sum([0 if a == b else 1 for a, b in zip(indices, assignments)]))

        if beta < maxBeta:
            beta = beta * 2.0
            print "New beta: {0}".format(beta)
        else:
            break

S = numpy.exp(logS)

assignments = []
for i in range(N):
    assignments.append(numpy.argmax(S[i, :]))

print "There were {0} unique assignments".format(len(set(assignments)))

if len(set(assignments)) > 0:
    print "Greedily fixing assignments"
    # Greedily fix duplicate assignments
    used = set()
    needFixed = []
    for i in range(N):
        if assignments[i] not in used:
            used.add(assignments[i])
        else:
            needFixed.append(i)

    options = list(set(range(N)) - used)

    for i in needFixed:
        print i, options, [A[i, j] for j in options]
        newAssignment = options[numpy.argmax([A[i, j] for j in options])]
        print newAssignment

        print '----'
        options.remove(newAssignment)

        assignments[i] = newAssignment

approxTime = time.time() - tmp

print "Summary ------"
print ""

print "Munkres took : ", munktime
print "Approximation took : ", approxTime
print "Approximation made {0} errors".format(sum([0 if a == b else 1 for a, b in zip(indices, assignments)]))

print "Munkres max: ", sum(A[i, indices[i]] for i in range(N))
print "Approx max (duplicate assignments not fixed): ", sum(A[i, assignments[i]] for i in range(N))