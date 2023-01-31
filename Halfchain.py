'''
Dynamical matrix propagation of driven fermionic problem
'''

from threading import local
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import eigvals
from scipy.linalg import eig
from matplotlib.ticker import FormatStrFormatter
# from jax.experimental.ode import odeint
import numpy as np
import matplotlib.pyplot as plt
# from celluloid import Camera

import qutip as q
from qutip import tensor

si, sx, sy, sz=q.qeye(2), q.sigmax(), q.sigmay(), q.sigmaz()
sp, sm=q.sigmap(), q.sigmam()
state_z_plus=q.basis(2,0)
state_z_minus=q.basis(2,1)
state_x_plus=1.0/np.sqrt(2)* (q.basis(2,0) +  q.basis(2,1))
state_x_minus=1.0/np.sqrt(2)* (q.basis(2,0) -  q.basis(2,1))

sza = tensor(sz,si); szb = tensor(si,sz)
sxa = tensor(sx,si); sxb = tensor(si,sx)
sya = tensor(sy,si); syb = tensor(si,sy)
spa = tensor(sp,si); spb = tensor(si,sp)
sma = tensor(sm,si); smb = tensor(si,sm)
siab = tensor(si,si)
Purity = lambda rho: (rho*rho).tr()



def X_maj(J,omega,J1,N=5):
    W = np.array([[0,0,2*omega],[0,-1.0/2,0],[-2*omega,0,-1.0/2]])
    A = np.zeros((3,2*N-2))
    A[1,1]=J1; A[2,0]=-J1
    R = np.array([[0,J],[-J,0]])
    T = np.kron(np.diag([1 for _ in range(N-2)], k=1),R)+\
            np.kron(np.diag([1 for _ in range(N-2)], k=-1),-R.T)
    return np.block([[W,A],[-A.T,T]])

def ODE(x,t,Xmat,Bmat):
    N = Xmat.shape[0]//2 # dim(X) = 2N+1
    xmat = x.reshape(2*N+1,2*N+1)
    return (Xmat@xmat+xmat@Xmat.T + Bmat).reshape((2*N+1)**2,)

def cov_ss(l,left,right,Bmat):
    overlap = np.diag(left.T.conj()@right)
    n1 = len(l)
    P = np.zeros((n1,n1))
    for i in range(n1):
        for j in range(n1):
            P = P- 1/(l[i]+l[j].conj())*(left[:,i].T.conj()@Bmat@left[:,j])/(overlap[i]*overlap[j].conj())*np.einsum('i,j->ij',right[:,i],right[:,j].conj())
    return P

def density(J,omega,N,Bi):
    Utransform = np.kron(np.diag([1 for _ in range(N)]),1/np.sqrt(2)*np.array([[1,1],[1j,-1j]])).T.conj()
    X = X_maj(J,omega,J,N=N)
    energy, left, right = eig(X,left=True)
    Gamma = cov_ss(energy,left,right,Bi)
    sys_mat = Gamma[1:,1:]
    ada_mat = Utransform@(np.eye(2*N)/2-1j*sys_mat)@Utransform.T
    return [np.real(ada_mat)[2*i,2*i+1] for i in range(N)]

def Bi(N):
    B = np.zeros((2*N+1,2*N+1)); B[1,2] = -1.0/2 ; B[2,1] = 1.0/2
    return B

def Time_evolve(t_stamp,N,omega,J,i):
    '''
    Return Time dependent local density
    '''
    smaller_mat = np.array([[1/2,1j/2],[-1j/2,1/2]])

    init_maj = np.block([[np.array([1/2]),np.zeros((1,2*N))],\
                    [np.zeros((2*N,1)),np.kron(np.eye(N),smaller_mat)]])
    
    init_m = 1j*init_maj-1j*np.eye(2*N+1)/2
    X = X_maj(J,omega,J,N=N)
    B = Bi(N) # Incoherent pumping
    time_clip = odeint(ODE,init_m.reshape((2*N+1)**2,),t_stamp,X,B,rtol=1.4e-10) #return y value for each value on t_stamp

    Utransform = np.kron(np.diag([1 for _ in range(N)]),1/np.sqrt(2)*np.array([[1j,-1j],[1,1]])).T.conj()
    res = []
    for each in time_clip:
        Gamma = each.reshape(2*N+1,2*N+1)
        sys_mat = Gamma[1:,1:]
        ada_mat = Utransform@(np.eye(2*N)/2-1j*sys_mat)@Utransform.T
        res.append(np.real(ada_mat[2*i+1,2*i]))
    return res

def G_r0N(omega,l,left,right):
    overlap = np.diag(left.T.conj()@right)
    n1 = len(l)
    G = 0
    u0 = np.zeros(n1,dtype='complex'); u0[1]= 1/np.sqrt(2)*1j; u0[2] = 1/np.sqrt(2)*1
    u1 = np.zeros(n1,dtype='complex'); u1[-2]= -1/np.sqrt(2)*1j; u1[-1] = 1/np.sqrt(2)*1
    for i in range(n1):
        G = G+ 1/(omega-1j*l[i])*1/(overlap[i])*(u1.T.conj()@right[:,i])*(left[:,i].T.conj()@u0)
    return G

def G_rii(site,omega,l,left,right):
    overlap = np.diag(left.T.conj()@right)
    n1 = len(l)
    G = 0
    u0 = np.zeros(n1,dtype='complex'); u0[1+2*site]= 1/np.sqrt(2)*1j; u0[2+2*site] = 1/np.sqrt(2)*1
    for i in range(n1):
        G = G+ 1/(omega-1j*l[i])*1/(overlap[i])*(u0.T.conj()@right[:,i])*(left[:,i].T.conj()@u0)
    return G

def gap(J,omega,N):
    '''
    gap as a function of coupling strength
    '''
    X = X_maj(J,omega,J,N=N)
    energy, left, right = eig(X,left=True)
    rate = np.imag(1j*energy)
    rate.sort()
    return np.abs(rate[-2])

def gap_plot():
    '''
    Plot the gap as a function of coupling strength
    '''
    N=21 ;J = 1; J1 = 1; omega= 10
    # Bi = np.zeros((2*N+1,2*N+1)); Bi[1,2] = -1.0/2 ; Bi[2,1] = 1.0/2
    # Utransform = np.kron(np.diag([1 for _ in range(N)]),1/np.sqrt(2)*np.array([[1,1],[1j,-1j]])).T.conj()
    
    Nspace = [5*x+10 for x in range(20)]
    omega_list = [0.1,0.2,0.5,1,2,10,100]
    gap = [[] for _ in omega_list]
    for i,omega in enumerate(omega_list):
        J1range = np.linspace(0.1,10,100)
        for J1 in J1range:
        # for N in Nspace:
            # X = X_maj(10*omega,omega,10*omega,N=N)
            X = X_maj(J1*omega,omega,J1*omega,N=N)
            energy, left, right = eig(X,left=True)
            rate = np.imag(1j*energy)
            rate.sort()
            gap[i].append(np.abs(rate[-2]))
            # gap.append(rate[-2])
        plt.plot(J1range,gap[i],'-o',label=rf'$\Omega/\kappa =$ {omega}')
    
    # plt.plot(Nspace,gap,'-o')
    # plt.xlabel('Total number N')
    plt.xlabel(r'$J/\Omega$')
    plt.ylabel('dissipative gap')
    plt.legend()
    plt.yscale('log')
    plt.show()
    return 

def diffusion_relation():
    '''
    Plot the gap as a function of total particle number
    '''

    # Bi = np.zeros((2*N+1,2*N+1)); Bi[1,2] = -1.0/2 ; Bi[2,1] = 1.0/2
    # Utransform = np.kron(np.diag([1 for _ in range(N)]),1/np.sqrt(2)*np.array([[1,1],[1j,-1j]])).T.conj()
    
    # Nspace = np.array([10*x+10 for x in range(20)])
    Nspace = np.logspace(1,3,10,dtype=int)
    omega_list = [0.1,1,5,10]
    gap = [[] for _ in omega_list]
    for i,omega in enumerate(omega_list):
        for N in Nspace:
            X = X_maj(10,omega,10,N=N)
            energy, left, right = eig(X,left=True)
            rate = np.imag(1j*energy)
            rate.sort()
            gap[i].append(np.abs(rate[-2]))
        plt.plot(Nspace,gap[i],'-o',label=rf'$\Omega/\kappa =$ {omega}')
    plt.plot(Nspace,gap[0][0]*Nspace[0]**3/Nspace**3,'--',label=r'$\propto N^{-3}$')
    plt.plot(Nspace,gap[0][0]*Nspace[0]**2/Nspace**2,'--',label=r'$\propto N^{-2}$')
    plt.xlabel('Total number N')
    plt.ylabel(r'Dissipative gap[$\kappa$]')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(
        r'Dissipative gap scaling as total length N ($J=10$ and $\kappa =1$)'
    )
    plt.show()
    return 

def scatter_G_eigenvalues(N=20, J = 1,omega= 10):
    # N=20; J = 1; J1 = 1; omega= 10
    # scattering plot of Eigenvalues
    X = X_maj(J,omega,J,N=N)
    energy, left, right = eig(X,left=True)
    plt.scatter(np.real(1j*energy),np.imag(1j*energy))
    plt.xlabel(r'Re($\lambda$)')
    plt.ylabel(r'Im($\lambda$)')
    plt.title(r'Eigenvalues of $H_{eff}$')
    return 

def Movie_scattering(omega_list):
    N=40; J = 10
    fig = plt.figure()
    cam = Camera(fig)
    for omega in omega_list:
        X = X_maj(J,omega,J,N=N)
        energy, left, right = eig(X,left=True)
        plt.scatter(np.real(1j*energy),np.imag(1j*energy),label=rf"$\Omega/\kappa=$ {omega}")
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.title(fr'Eigenspectrum, $J/\kappa =$ {J}')
        plt.legend(loc='upper right')
        cam.snap()
    return cam

def Time_evolve_qutip(t_space,omega,J):
    H2 = lambda omega,J: omega*sxa+J*(tensor(sp,sm)+tensor(sm,sp))
    c2 = lambda kappa: np.sqrt(kappa)*sma
    res = q.mesolve(H2(omega,J), q.ket2dm(tensor(state_z_minus,state_z_minus)), t_space, [c2(1)])
    return res

def VonNeumann_entropy(J,omega,N):
    '''
    Use the free fermion correlation matrix to calculate the entanglement entropy
    S = tr (rho_a log rho_a)
    '''
    ### Construct the covariance matrix
    # print(f"J={J}, omega={omega}")
    X = X_maj(J,omega,J,N)
    energy, left, right = eig(X,left=True)
    B = Bi(N)
    Gamma = cov_ss(energy,left,right,B)
    sys_mat = 2*Gamma[1:,1:]/1j # There is a factor of two because of the definition of majorana operators
    Cov_eigenvals = np.real(eig(sys_mat)[0])
    Cov_eigenvals = Cov_eigenvals[Cov_eigenvals>0] # Take only the positive value
    # print(Cov_eigenvals)
    S = lambda v: -(1+v)/2*np.log((1+v)/2)-(1-v)/2*np.log((1-v)/2)
    return sum([S(v) for v in Cov_eigenvals])

from matplotlib.colors import LogNorm

def Entropy_sweep(N,N_p=40):
    '''
    Sweep the parameter space to get omega and J dependence
    '''
    omega_space = np.logspace(-1,1,N_p)
    J_space = np.logspace(-1,1,N_p)

    Entropy_list = [[VonNeumann_entropy(J,omega,N)/(N*np.log(2)) for omega in omega_space] for J in J_space]

    plt.imshow(np.abs(Entropy_list),extent=(omega_space.min(),omega_space.max(),J_space.min(),J_space.max()),
                    norm=LogNorm(vmin=0.1,vmax=1),cmap='seismic',origin='lower')

    plt.xlabel(r'$\Omega/\kappa$')
    plt.ylabel(r'$J/\kappa$')
    plt.title(rf'Entanglement Entropy ($S(N)/S_B(N)$)(N={N})')
    plt.xticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.yticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.colorbar()
    plt.savefig(
        './figs/sweep.png'
    )
    plt.show()

def gap_sweep(N,N_p=40):
    '''
    Sweep the parameter space to get omega and J dependence on gap
    '''
    omega_space = np.logspace(-1,1,N_p)
    J_space = np.logspace(-1,1,N_p)

    gap_list = [[gap(J,omega,N) for omega in omega_space] for J in J_space]

    plt.imshow(np.abs(gap_list),extent=(omega_space.min(),omega_space.max(),J_space.min(),J_space.max()),
                    norm=LogNorm(vmin=1e-6,vmax=1e-1),cmap='seismic',origin='lower')

    plt.xlabel(r'$\Omega/\kappa$')
    plt.ylabel(r'$J/\kappa$')
    plt.title(rf'slow rate (N={N})')
    plt.xticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.yticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.colorbar()
    plt.savefig(
        './figs/slowrate.png'
    )
    # plt.show()

def bell_pair_rate(N,N_p=40):
    omega_space = np.logspace(-1,1,N_p)
    J_space = np.logspace(-1,1,N_p)

    Entropy_list = np.array([[VonNeumann_entropy(J,omega,N)/(np.log(2)) for omega in omega_space] for J in J_space])
    gap_list = np.array([[gap(J,omega,N) for omega in omega_space] for J in J_space])
    plt.imshow(np.multiply(Entropy_list,gap_list),extent=(omega_space.min(),omega_space.max(),J_space.min(),J_space.max()),
                    norm=LogNorm(vmin=1e-5,vmax=1),cmap='seismic',origin='lower')

    plt.xlabel(r'$\Omega/\kappa$')
    plt.ylabel(r'$J/\kappa$')
    plt.title(rf'Entropy generation rate (ebits*$\kappa$) N = {N}')
    plt.xticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.yticks([0,5,10],[r'$10^{-1}$',r'$10^0$',r'$10^1$'])
    plt.colorbar()
    plt.savefig(
        './figs/rate.png'
    )
    plt.show()

def linecut(J,N,N_p=40):
    '''
    Sweep the parameter space to get omega dependence for a given J
    '''
    omega_space = np.logspace(-2,2,N_p)

    Entropy_list = [VonNeumann_entropy(J,omega,N)/(N*np.log(2)) for omega in omega_space]
    plt.plot(omega_space,Entropy_list,'--')
    plt.xlabel(r'$\Omega/\kappa$')
    plt.ylabel(rf'Relative Entropy')
    plt.xscale('log')
    plt.title(rf'Entanglement Entropy ($S(N)/S_B(N)$)(N={N} J={J})')
    plt.savefig(
        './figs/line.png'
    )

def Entropy_N(J,omega):
    Nspace = np.logspace(1,2,10,dtype=int)
    Entropy_list = [VonNeumann_entropy(J,omega,N) for N in Nspace ]
    plt.plot(Nspace,Entropy_list,'-o',label=rf'$\Omega/\kappa =$ {omega},$J/\kappa={J}$')
    p = np.polyfit(np.log(Nspace),np.log(Entropy_list),1)
    plt.plot(Nspace,p[1]*Nspace**p[0],label=f'a = {p[0]}')
    # plt.plot(Nspace,gap[0][0]*Nspace[0]**3/Nspace**3,'--',label=r'$\propto N^{-3}$')
    # plt.plot(Nspace,gap[0][0]*Nspace[0]**2/Nspace**2,'--',label=r'$\propto N^{-2}$')
    return 

def powerlaw_fit(J,axis):
    omega_list = np.logspace(-2,1,10)
    Nspace = np.logspace(1,2,5,dtype=int)
    power_list = []
    for omega in omega_list:
        entropy = np.array([VonNeumann_entropy(J,omega,N) for N in Nspace ])
        p = np.polyfit(np.log(Nspace),np.log(entropy),1)
        power_list.append(p[0]) # Add the linear part to
    axis.plot(omega_list,power_list,label=f'J={J}')
    return

if __name__ == '__main__':
    # fig, ax = plt.subplots()
    # Jlist = np.logspace(-1,1,3)
    # for J in Jlist:
    #     powerlaw_fit(J,ax)
    # ax.set_ylabel(r'$\alpha$')
    # ax.set_xlabel(r'$\omega/\kappa$')
    # ax.set_xscale('log')
    # ax.set_title(r'Power law scaling of entanglement $S(N)=S_0 N^\alpha$')
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.legend()
    # fig.savefig(
    #     './figs/scaling.png'
    # )
    # plt.show()

    ### Entropy sweep
    # Entropy_sweep(10,N_p=40)

    ### Entropy line cut
    J = 1; N=10
    # linecut(J,N)

    ### Entanglement Rate
    # bell_pair_rate(N)

    ###slow rate
    gap_sweep(10,N_p=40)
    ### Entropy Calculation
    # print(VonNeumann_entropy(1,0.1,10)/(10*np.log(2)))

    # Entropy_N(1,1)
    # Entropy_N(10,0.1)
    # Entropy_N(0.1,10)
    # plt.xlabel('Total number N')
    # plt.ylabel(r'Entanglement Entropy $-\mathrm{Tr} [\rho_A \log \rho_A ]$')
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(
    #     rf'Entanglement entropy scaling'
    # )
    # plt.show()

    # t_space = np.linspace(0,5,100)
    # clips = Time_evolve(t_space,N,omega,J,0)
    # clips_q = Time_evolve_qutip(t_space,omega,J)
    # plt.plot(t_space,clips,'r--',label='')
    # plt.plot(t_space,np.real([q.expect(spa*sma,state) for state in clips_q.states]),'k',label='')
    # plt.xlabel(r'time t ($\kappa^{-1}$)')
    # plt.ylabel(r'$\langle \gamma_1^\dagger \gamma_1 \rangle$')
    # plt.title(r'Time evolution of local density $\langle \gamma_1^\dagger \gamma_1 \rangle$')
    # plt.legend()
    # plt.show()
    



    #Eigenvalues of Non hermitian matrix
    # omega = 0.1
    # J =2*omega
    # X = X_maj(J,omega,J,N=N)
    # energy, left, right = eig(X,left=True)
    # plt.scatter(np.real(1j*energy),np.imag(1j*energy),label="With Rabi")  

    # diffusion_relation()
    # X = X_maj(J,0,J,N=N)
    # energy, left, right = eig(X,left=True)
    # plt.scatter(np.real(1j*energy),np.imag(1j*energy),label="w/o Rabi")  
    # plt.xlabel(r'Re($\lambda$)')
    # plt.ylabel(r'Im($\lambda$)')
    # plt.title(fr'Eigenspectrum $\Omega/\kappa=${omega}, $J/\Omega =$ {J/omega}')
    # plt.legend(loc='upper right')
    # plt.show()

    # omega_list = np.linspace(0,5,30)
    # gap_list = []
    # N=6; J = 10
    # fig, ax1 = plt.subplots()
    # for omega in omega_list:
    #     X = X_maj(J,omega,J,N=N)
    #     energy, left, right = eig(X,left=True)
    #     sortedrate = sorted([[np.abs(np.real(1j*x)),np.imag(1j*x)]  for x in energy])[3]
    #     gap_list.append(sortedrate[-1])
    # ax1.plot(omega_list,gap_list)
    # ax1.set_ylabel('Dissipative Gap')
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax1.set_xlabel(r'Rabi drive $\Omega [\kappa]$')
    # plt.title(f'Im(E) for smallest energy mode (J={J}, N={N})')
    # plt.show()



    #Movie writer
    # omega_list = np.linspace(0,5,30)
    # N=2; J = 10
    # fig = plt.figure()
    # ax1=fig.add_subplot(111)
    # cam = Camera(fig)
    # for i,omega in enumerate(omega_list):
    #     X = X_maj(J,omega,J,N=N)
    #     energy, left, right = eig(X,left=True)
    #     artist = ax1.scatter(np.real(1j*energy),np.imag(1j*energy),color='blue')
    #     plt.xlabel(r'Re($\lambda$)')
    #     plt.ylabel(r'Im($\lambda$)')
    #     plt.title(fr'Eigenspectrum, $J/\kappa =$ {J}')
    #     plt.legend([artist],[rf"$\Omega/\kappa=$ {np.round(omega,decimals=2)}"],loc='upper right')
    #     cam.snap()
    # animation = cam.animate(interval = 200, repeat = True,
    #                        repeat_delay = 500)
    # animation.save('animation_2qubits.gif', writer='imagemagick')

    # Bi = np.zeros((2*N+1,2*N+1)); Bi[1,2] = -1.0/2 ; Bi[2,1] = 1.0/2
    # Utransform = np.kron(np.diag([1 for _ in range(N)]),1/np.sqrt(2)*np.array([[1,1],[1j,-1j]])).T.conj()

    # X = X_maj(J,omega,J1,N=N)
    # energy, left, right = eig(X,left=True)
    # idx = energy.argsort()[::-1]   
    # energy = energy[idx]
    # left = left[:,idx]
    # right = right[:,idx]
    # print(np.round(np.einsum('i,j->ij',right[:,0],right[:,0].conj()),decimals=3))

    # wspace = np.linspace(0.001,20,2000)
    # G_list = [G_r0N(w,energy,left,right) for w in wspace]
    # # print(np.real(G_list[0]))
    # plt.plot(wspace,np.abs(G_list)**2)
    # plt.xlabel(r'$\omega$')
    # plt.ylabel(r'$G^r_{0N} [\omega]$')
    # plt.title(f'Retarded Greens function (N={N}, J/k={J}, \omega/k = {omega})')
    # plt.show()

    # Plot out the last site density as a function of length
    # N_list = []
    # N_mid = []
    # N_init = []
    # J= 10
    # for x in range(50):
    #     chain = density(J,omega,x+2,Bi(x+2))
    #     N_list.append(chain[-1])
    #     N_mid.append(chain[x//2])
    #     N_init.append(chain[0])
    # plt.plot(N_list,'-o',label=r'$\langle c^\dagger_N c_N\rangle$')
    # plt.plot(N_mid,'-o',label=r'$\langle c^\dagger_{N/2} c_{N/2}\rangle$')
    # plt.plot(N_init,'-o',label=r'$\langle c^\dagger_{0} c_{0}\rangle$')
    # plt.ylabel('Local density')
    # plt.xlabel('site number (i)')
    # plt.title(rf'Length dependence ($J/k$={J}, $\Omega$/k ={omega})')
    # plt.legend()
    # plt.show()

    
