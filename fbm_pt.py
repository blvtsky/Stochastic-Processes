import numpy as np
import math
import mpmath as sb
from scipy.special import hyp1f1,hyp2f1
import sys 
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe


import matplotlib.pyplot as plt
import matplotlib as mp
mp.rc('font', family='DejaVu Sans')
mp.rcParams['legend.fontsize'] = 8
mp.rcParams['font.size'] = 10
mp.rcParams['figure.figsize'] = 12.0, 12.0 # A4 inches, landscape
# mp.pyplot.subplots_adjust(hspace=0.25, wspace=0.5)
# mp.pyplot.subplots_adjust(hspace=0.25, wspace=0.25)

ax01 = plt.subplot(2,2, 1)
ax02 = plt.subplot(2,2, 2)
ax03 = plt.subplot(2,2, 3)
ax04 = plt.subplot(2,2, 4)
aaa = [ax01, ax02, ax03, ax04]


M =  50 # number of cells
N = 200 # number of time points
rng = np.random.default_rng(20250828)  
Z=rng.standard_normal((M, N, 2))
dt = 1
h=0.7
vd = 0.4
gamma=math.gamma(2*h+1)
Dh=10
gamma_ou=0.15
sigma=np.sqrt(2)
vm2=(sigma**2)/(2*(gamma_ou))
Deff=(sigma**2)/(2*(gamma_ou**2))
x1m_pt = np.zeros(N)
x2m_pt = np.zeros(N)
x1m_et = np.zeros(N)
x2m_et = np.zeros(N)
x1m_fbm = np.zeros(N)
x2m_fbm = np.zeros(N)
x1m_ou = np.zeros(N)
x2m_ou = np.zeros(N)
c=np.zeros((N,N))
c1=np.zeros((N,N))
c2=np.zeros((N,N))
c3=np.zeros((N,N))
tau=5
tex=0.7
#np.random.seed(42)
for k in range(M):
    x0 = 0
    tt = np.zeros(N)
    xx = np.zeros(N)
    yy=np.zeros(N)
    xx[0] = x0
    for i in range(N):
        ti = (i+1)*dt
        tt[i] = ti
        for j in range(N):
            tj = (j+1)*dt
            F1=hyp1f1(2*h-1, 2*h+1, -ti/tau)
            F2=hyp1f1(2*h-1, 2*h+1, -tj/tau)
            F3=hyp1f1(2*h-1, 2*h+1, -abs(ti-tj)/tau)
            F4=hyp2f1(tex,2*h-1, 2*h+1, -ti/tau)
            F5=hyp2f1(tex,2*h-1, 2*h+1, -tj/tau)
            F6=hyp2f1(tex,2*h-1, 2*h+1, -abs(ti-tj)/tau)
            c[i][j]=(ti**(2*h)*F1+tj**(2*h)*F2-abs(ti-tj)**(2*h)*F3)*Dh/gamma #et
            c1[i][j]=(ti**(2*h)*F4+tj**(2*h)*F5-abs(ti-tj)**(2*h)*F6)*Dh/gamma #pt
            c2[i][j]=(ti**(2*h)+tj**(2*h)-abs(ti-tj)**(2*h))*Dh/gamma #fbm 
            c3[i][j]=vm2/(gamma_ou**2)*(2*gamma_ou*min(ti,tj)-1-sb.e**(-gamma_ou*abs(ti-tj))+sb.e**(-gamma_ou*ti)+sb.e**(-gamma_ou*tj))
    #print(c)
    A = np.linalg.cholesky(c) #et
    A1 = np.linalg.cholesky(c1) #pt
    A2=np.linalg.cholesky(c2)
    A3=np.linalg.cholesky(c3)
    #print(np.shape(A))
    #print(len(tt))
    #print(np.allclose(c, A @ A.T))
    eta=Z[k,:,0]
    eta1=Z[k,:,1]
    x=A1@eta
    y=A1@eta1
    x1m_pt+=x/M
    x2m_pt+=x**2/M
    ax01.plot(tt, x, "-",color="grey",label='pt') #pt plot

    x=A@eta
    x1m_et+=x/M
    x2m_et+=x**2/M
    ax01.plot(tt,x,"-",color="red",label='et') #et plot
    x=A2@eta
    ax01.plot(tt,x,"-",color="blue",label='fbm') #fbm plot
    #ax02.plot(yy,xx,"-")
    x1m_fbm += x/M
    x2m_fbm += x**2/M
    x=A3@eta
    x1m_ou+=x/M
    x2m_ou+=x**2/M
    ax01.plot(tt, x, "-",color="green",label='pt') #ou plot

x2m_pt = (x2m_pt-x1m_pt**2)
x2m_et = (x2m_et-x1m_et**2)
x2m_fbm = (x2m_fbm-x1m_fbm**2)
x2m_ou=(x2m_ou-x1m_ou**2)
ax01.yaxis.set_major_locator(MultipleLocator(30))
ax01.grid(True, which="both", axis="y")

ymin, ymax = ax01.get_ylim()
ax01.set_ylim(10*int(np.floor(ymin/10)), 10*int(np.ceil(ymax/10)))
msd_fbm_th = 2*Dh*tt**(2*h)/gamma
F_t = np.array([hyp1f1(2*h - 1, 2*h + 1, -t_ / tau) for t_ in tt], dtype=float)
F_t2 = np.array([hyp2f1(tex,2*h - 1, 2*h + 1, -t_ / tau) for t_ in tt], dtype=float)
msd_et_th=2*Dh*tt**(2*h)*F_t/gamma
msd_pt_th=2*Dh*tt**(2*h)*F_t2/gamma
msd_ou_th=2*Deff*(tt-(1-np.e**(-gamma_ou*tt))/gamma_ou)
ax02.plot(tt, x2m_pt, "-", color="red")
ax02.plot(tt,msd_pt_th,"-",color="green")
ax02.plot(tt, x2m_et, "-", color="red")
ax02.plot(tt,msd_et_th,"-",color="blue")
ax02.plot(tt, x2m_fbm, "-", color="red")
ax02.plot(tt,msd_fbm_th,"-",color="yellow")
ax02.plot(tt, x2m_ou, "-", color="red")
ax02.plot(tt,msd_ou_th,"-",color="purple")
line_pt, = ax02.plot(tt, msd_pt_th, "-", color="green")  # your green plot
xdata, ydata = line_pt.get_xdata(), line_pt.get_ydata()
j = int(0.8*(len(xdata)-1))
ax02.text(xdata[j]-20, ydata[j]-1200, "pt",
          color="black", fontsize=12, va="bottom", ha="center",
          path_effects=[pe.withStroke(linewidth=3, foreground="white")],
          zorder=5)
ax02.text(xdata[j]-20, ydata[j]-2500, "et",
          color="black", fontsize=12, va="bottom", ha="center",
          path_effects=[pe.withStroke(linewidth=3, foreground="white")],
          zorder=5)
ax02.text(xdata[j]-20, ydata[j]+400, "fbm",
          color="black", fontsize=12, va="bottom", ha="center",
          path_effects=[pe.withStroke(linewidth=3, foreground="white")],
          zorder=5)

#ax03.plot(tt, x1m)
#ax03.plot(tt, vd*tt)

#ax04.plot(tt, x2m, "-")
#ax04.plot(tt, 2*Dh*tt, "-", color="grey")
ax04.plot(tt, 2*Dh*tt**(2*h)/gamma, "-", color="blue")
ax04.plot(tt, 2*Dh*tt**(2*h)*F_t/gamma, "-", color="red")
ax04.plot(tt, 2*Dh*tt**(2*h)*F_t2/gamma, "-", color="green")
ax04.plot(tt, 2*Deff*(tt-(1-np.e**(-gamma_ou*tt))/gamma_ou), "-")
ax04.set_xscale('log')
ax04.set_yscale('log')


labels = ["A", "B", "C", "D"]; i=0
for aa in aaa:
    aa.grid(True)
    # aa.set_xlabel("time")
    # aa.set_ylabel("$Ca^{2+}$ response")
    # plt.text(-0.25, 1.05, labels[i], fontweight="bold", fontsize=16, transform=aa.transAxes)
    plt.text(-0.10, 1.05, labels[i], fontweight="bold", fontsize=16, transform=aa.transAxes)
    i += 1

# plt.suptitle(sheetname+"  $p_{min}$ = "+str(pmin))    
plt.savefig("test.png", dpi=660, bbox_inches='tight')
