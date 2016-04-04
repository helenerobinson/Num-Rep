import numpy as np 
import pylab as plt
from MaximumLikelihood import Minimiser


A = Minimiser("DecayTimesData.txt")

min_value = A.min()


# for plotting the PDF 
time = np.linspace(0, 7, 500)
PDF = [A.pdf(min_value, t ) for t in time]
print 'Minimum: f = ' , min_value[0], ' tau1 = ', min_value[1], ' tau2 =' , min_value[2]


results = A.error(0.001)
print results

# unpacking the various sigmas 
sigma_f = results['f'][2]
sigma_tau1 = results['tau1'][2]
sigma_tau2 = results['tau2'][2]


total_min = A.max_likelihood(min_value)

# plotting the errors for tau1
vary_tau1 = np.linspace(min_value[1] - 3*sigma_tau1, min_value[1] + 3*sigma_tau1, 50)
error_tau1 = [A.max_likelihood([min_value[0], i, min_value[2]]) - total_min for i in vary_tau1]

plt.plot(vary_tau1, error_tau1)
plt.xlabel('$\\tau_1$')
plt.ylabel('$NLL$')
plt.axvline(x = (min_value[1] - sigma_tau1),linestyle = '-.', label = '1$\sigma$')
plt.axvline(x = (min_value[1] - 2*sigma_tau1),linestyle = '-.')
plt.axvline(x = (min_value[1] - 3*sigma_tau1),linestyle = '-.')
plt.axvline(x = (min_value[1] + sigma_tau1),linestyle = '-.')
plt.axvline(x = (min_value[1]  + 2*sigma_tau1),linestyle = '-.')
plt.axvline(x = (min_value[1]  + 3*sigma_tau1),linestyle = '-.')
plt.savefig('tau1_error.pdf')
plt.show()


# plotting the errors for tau2
vary_tau2 = np.linspace(min_value[2] - 3*sigma_tau2, min_value[2] + 3*sigma_tau2, 50)
error_tau2 = [A.max_likelihood([min_value[0], min_value[1], i]) - total_min  for i in vary_tau2]

plt.plot(vary_tau2, error_tau2)
plt.xlabel('$\\tau_2$')
plt.ylabel('$NLL$')
plt.axvline(x = (min_value[2] - sigma_tau2),linestyle = '-.')
plt.axvline(x = (min_value[2] - 2*sigma_tau2),linestyle = '-.')
plt.axvline(x = (min_value[2] - 3*sigma_tau2),linestyle = '-.')
plt.axvline(x = (min_value[2] + sigma_tau2),linestyle = '-.')
plt.axvline(x = (min_value[2]  + 2*sigma_tau2),linestyle = '-.')
plt.axvline(x = (min_value[2]  + 3*sigma_tau2),linestyle = '-.')
plt.savefig('tau2_error.pdf')
plt.show()


# plotting the errors for f
vary_f  = np.linspace(min_value[0] - 3*sigma_f, min_value[0] + 3*sigma_f, 50)
error_f = [A.max_likelihood([i, min_value[1], min_value[2]]) - total_min  for i in vary_f]

plt.plot(vary_f, error_f)
plt.xlabel('$f$')
plt.ylabel('$NLL$')
plt.axvline(x = (min_value[0] - sigma_f),linestyle = '-.')
plt.axvline(x = (min_value[0] - 2*sigma_f),linestyle = '-.')
plt.axvline(x = (min_value[0] - 3*sigma_f),linestyle = '-.')
plt.axvline(x = (min_value[0] + sigma_f),linestyle = '-.')
plt.axvline(x = (min_value[0] + 2*sigma_f),linestyle = '-.')
plt.axvline(x = (min_value[0] + 3*sigma_f),linestyle = '-.')
plt.savefig('f_error.pdf')
plt.show()





#produces a histogram of the decay times
plt.hist(A.data, bins=100, normed = True)
plt.plot(time, PDF, color = 'c', label = 'PDF', linewidth = 3)
plt.legend()
plt.title('Decay Times')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.savefig('outputHistogram.pdf')
plt.show()

#produces a histogram of the decay times
plt.hist(A.data, bins=100, normed= 1) 
plt.plot(time, PDF,  label = 'PDF')
plt.legend()
plt.title('Decay Times - log scale')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.savefig('outputHistogram_logplot.pdf')
plt.show()