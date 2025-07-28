import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("dj1_deuterated_Combined_IR.csv", skip_header=1, delimiter=',')
x_freq = data[:,0]
y_intens_raw = data[:,1]
fig = plt.figure(figsize=[8,5],dpi=300,facecolor='white')
ax = fig.add_subplot(1,1,1)
## fingerprint region
ax.axvspan(500,1500,color='grey',alpha=0.25)
ax.text(1000,max(y_intens_raw)*0.95,'Fingerprint Region',ha='center',va='center',fontsize=8)
ax.plot(x_freq,y_intens_raw,color='r',linestyle='--',lw=0.5)
ax.set_xlim(4000,0)
ax.set_yticks([])
ax.set_ylim(0,max(y_intens_raw)*1.05)
ax.set_xlabel(r'Wavenumbers (cm$^{-1}$)')
fig.tight_layout()
fig.savefig("dj1_deuterated_Combined_IR.png",dpi=300,facecolor='white')


