import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

def MAKE_POS_MATRIX(pos):
    x,y,z = pos[0], pos[1], pos[2]
    X = np.zeros((N,N))
    Y = np.zeros((N,N))
    Z = np.zeros((N,N))
    for i in range(N):
        X[i] = np.roll(x,-i)
        Y[i] = np.roll(y,-i)
        Z[i] = np.roll(z,-i)
    return np.vstack((np.expand_dims(X,axis=0),np.expand_dims(Y,axis=0),np.expand_dims(Z,axis=0)))

def MAKE_MASS_MATRIX(mass):
    M = np.zeros((N,N))
    for i in range(N):
        M[i] = np.roll(mass,-i)
    return M
        
def GET_ACCELERATION(X,M):
    pos = np.expand_dims(X[:,:,0],axis=2)
    return ((((X[:,:,1:]-pos)**2).sum(axis=0))**-1.5*M[:,1:]*(X[:,:,1:]-pos)).sum(axis=2)

def DKD_SCHEME(pos,vel,M,dt):
    pos += vel*0.5*dt
    X = MAKE_POS_MATRIX(pos)
    A = GET_ACCELERATION(X,M)
    vel += A*dt
    pos += vel*0.5*dt
    X = MAKE_POS_MATRIX(pos)
    energy = GET_ENERGY(X,M,vel,mass)
    return pos, vel, energy

def KDK_SCHEME(pos,vel,M,A,dt):
    vel += A*0.5*dt
    pos += vel*dt
    X = MAKE_POS_MATRIX(pos)
    A = GET_ACCELERATION(X,M)
    vel += A*0.5*dt
    energy = GET_ENERGY(X,M,vel,mass)
    return pos, vel, A, energy

def RK4_SCHEME(pos,vel,M,X,A,dt):
    k_1_vel = A
    k_1_pos = vel
    X = MAKE_POS_MATRIX(pos+k_1_pos*dt/2.)
    k_2_vel = GET_ACCELERATION(X,M)
    k_2_pos = vel + k_1_vel*dt/2.
    X = MAKE_POS_MATRIX(pos+k_2_pos*dt/2.)
    k_3_vel = GET_ACCELERATION(X,M)
    k_3_pos = vel + k_2_vel*dt/2.
    X = MAKE_POS_MATRIX(pos+k_3_pos*dt)
    k_4_vel = GET_ACCELERATION(X,M)
    k_4_pos = vel + k_3_vel*dt
    pos += dt/6.*(k_1_pos+2.*k_2_pos+2.*k_3_pos+k_4_pos)
    vel += dt/6.*(k_1_vel+2.*k_2_vel+2.*k_3_vel+k_4_vel)
    X = MAKE_POS_MATRIX(pos)
    A = GET_ACCELERATION(X,M)
    energy = GET_ENERGY(X,M,vel,mass)
    return pos, vel, X, A, energy

def GET_ENERGY(X,M,vel,mass):
    kinetic = (mass*vel**2).sum()
    pos = np.expand_dims(X[:,:,0],axis=2)
    potential = -(((M[:,1:]*(((X[:,:,1:]-pos)**2).sum(axis=0))**-0.5).sum(axis=1))*mass).sum()
    return 0.5*(kinetic + potential)

def GET_ERROR(pos,pos_analytic):
    error_pos = (((pos-pos_analytic)**2).mean()/(pos_analytic**2).mean())**0.5
    return error_pos

def GET_ANALYTIC_RESULT(t,A):
    if A>=1.:
        pos_analytic = INTERPOLATE(theta,T,t)
    else:
        pos_analytic = INTERPOLATE(theta,T,t-T_half*2*(t//(T_half*2)))
    return pos_analytic

def init():
	ball_1.set_data(pos[0][0],pos[1][0])
	ball_2.set_data(pos[0][1],pos[1][1])
	ball_1_analytic.set_data(pos[0][0],pos[1][0])
	ball_2_analytic.set_data(pos[0][1],pos[1][1])
	title.set(text="Time = %.3f $2\pi/\omega$\nError of Energy = %.4e"%(t/period, error_energy))
	return ball_1, ball_2, ball_1_analytic, ball_2_analytic, title

def anime(i):
	global t, pos, vel, pos_analytic, acc, X
	if mode == 'KDK':
		if t==0:
			X = MAKE_POS_MATRIX(pos)
			acc = GET_ACCELERATION(X,M)
		for step in range(drawing_resolution):
			pos, vel, acc,energy = KDK_SCHEME(pos,vel,M,acc,dt)
			t += dt
			t_record.append(t)
			if energy_analytic!=0:
			    error_energy = (energy-energy_analytic)/energy_analytic
			else:
			    error_energy = energy-energy_analytic
			
			energy_record.append(error_energy)
			if t > end_t:
				print("End time reached!")
				break

	elif mode == 'DKD':
		for step in range(drawing_resolution):
			pos, vel,energy = DKD_SCHEME(pos,vel,M,dt)
			t += dt
			t_record.append(t)
			if energy_analytic!=0:
				error_energy = (energy-energy_analytic)/energy_analytic
			else:
				error_energy = energy-energy_analytic
			energy_record.append(error_energy)
			if t > end_t:
				print("End time reached!")
				break

	elif mode == 'RK4':
		if t==0:
			X = MAKE_POS_MATRIX(pos)
			acc = GET_ACCELERATION(X,M)
		for step in range(drawing_resolution):
			pos, vel, X, acc, energy = RK4_SCHEME(pos,vel,M,X,acc,dt)
			t += dt
			t_record.append(t)
			if energy_analytic!=0:
			    error_energy = (energy-energy_analytic)/energy_analytic
			else:
			    error_energy = energy-energy_analytic
			energy_record.append(error_energy)
			if t > end_t:
				print("End time reached!")
				break
			
	pos_analytic = GET_ANALYTIC_RESULT(t,A)
	ball_1.set_data( pos[0][0], pos[1][0] )
	ball_2.set_data( pos[0][1], pos[1][1] )
	ball_1_analytic.set_data( pos_analytic[0][0], pos_analytic[1][0] )
	ball_2_analytic.set_data( pos_analytic[0][1], pos_analytic[1][1] )
	title.set(text="Time = %.3f $2\pi/\omega$\nError of Energy = %.4e"%(t/period, error_energy))
	return ball_1, ball_2, ball_1_analytic, ball_2_analytic, title

def F_PARABOLIC(x):
    return np.sin(x)*(np.cos(x)+2.)/3./(np.cos(x)+1.)**2

def F_HYPERBOLIC(x,A):
    return A*np.sin(x)/(A**2-1.)/(A*np.cos(x)+1.)-2.*np.arctanh((A-1.)*np.tan(x/2.)/(A**2-1)**0.5)/(A**2-1.)**1.5

def F_ELLIPSE(x,A):
    return (A*np.sin(x)/(A**2-1.)/(A*np.cos(x)+1.)-2.*np.arctanh((A-1.)*np.tan(x/2.)/(A**2-1.)**0.5)/(A**2-1.)**1.5).real

def INTERPOLATE(theta, t, t_selected):
    theta_to_t = interp1d(t, theta, kind='cubic')
    theta_selected = np.float(theta_to_t(t_selected))
    r = L**2/M_total/M_reduce**2/(1.+A*np.cos(theta_selected))
    x_analytic = r/M_total*np.array([np.cos(theta_selected)*m2, -np.cos(theta_selected)*m1])
    y_analytic = r/M_total*np.array([np.sin(theta_selected)*m2, -np.sin(theta_selected)*m1])
    pos_analytic = np.vstack((x_analytic,y_analytic, np.zeros(2)))
    return pos_analytic
###################################################################################################################

N = 2
r = 1.0
m1, m2 = 4., 1.0
M_total = m1+m2
M_reduce = m1*m2/(m1+m2)
A = 0.4
L = ((1.+A)*r*M_total*M_reduce**2)**0.5
omega = L/M_reduce/r**2
period = 2.*np.pi/omega

x = r/M_total*np.array([m2, -m1])
y = np.zeros(2)
z = np.zeros(2)
pos = np.vstack((x,y,z))
X = MAKE_POS_MATRIX(pos)

vx = np.zeros(2)
vy = omega*x
vz = np.zeros(2)
vel = np.vstack((vx,vy,vz))

mass = np.array([m1,m2])
M = MAKE_MASS_MATRIX(mass)

if A==1.:    
    c = L**2/2./M_total/M_reduce**2
    x_f = c
    x_prime = x_f + np.linspace(-50*c,0,1000)
    y_prime = (abs(4.*c*(x_prime-x_f)))**0.5
    x_prime = np.hstack((x_prime,np.flip(x_prime)))
    y_prime = np.hstack((y_prime,np.flip(-y_prime)))
    theta = np.linspace(0,np.pi*0.95,400)
    T = L**3/M_total**2/M_reduce**3*F_PARABOLIC(theta)
    
elif A<1.0:
    x_f = -A*L**2/M_total/M_reduce**2/(1.-A**2)
    a = L**2/(1.-A**2)/M_total/M_reduce**2
    b = L**2/(1.-A**2)**0.5/M_total/M_reduce**2
    x_prime = x_f + np.linspace(-a,a,1000)
    y_prime = b*(abs(1.-((x_prime-x_f)/a)**2))**0.5
    x_prime = np.hstack((x_prime,np.flip(x_prime)))
    y_prime = np.hstack((y_prime,np.flip(-y_prime)))
    theta = np.linspace(0.,2.*np.pi,200)
    T = np.zeros_like(theta)
    T[:len(theta)//2] = L**3/M_total**2/M_reduce**3*F_ELLIPSE(theta[:len(theta)//2], A)
    T_half = 2.*np.pi*a*b*M_reduce/L*0.5
    T[len(theta)//2:] = 2.*T_half-np.flip(T[:len(theta)//2])
    
else:
    x_f = A*L**2/M_total/M_reduce**2/(A**2-1.)
    a = L**2/(A**2-1.)/M_total/M_reduce**2
    b = L**2/(A**2-1.)**0.5/M_total/M_reduce**2
    x_prime = x_f + np.linspace(-50*a,-a,200)
    y_prime = b*(abs(1.-((x_prime-x_f)/a)**2))**0.5
    x_prime = np.hstack((x_prime,np.flip(x_prime)))
    y_prime = np.hstack((y_prime,np.flip(-y_prime)))
    theta = np.linspace(-np.arccos(-1./A)*0.9999,np.arccos(-1./A)*0.9999,1000)
    T = L**3/M_total**2/M_reduce**3*F_HYPERBOLIC(theta,A)
    
x_1_analytic = x_prime*m2/M_total
y_1_analytic = y_prime*m2/M_total
x_2_analytic = -x_prime*m1/M_total
y_2_analytic = -y_prime*m1/M_total

x_min = min(x_1_analytic.min(), x_2_analytic.min())
x_max = max(x_1_analytic.max(), x_2_analytic.max())
y_min = min(y_1_analytic.min(), y_2_analytic.min())
y_max = max(y_1_analytic.max(), y_2_analytic.max())

dt = 1e-2
t = 0.0
error_energy = 0.0
index, drawing_resolution = 0, 100
end_t = 80.0*period
mode = 'RK4'
t_record = []
energy_record = []
energy_analytic = GET_ENERGY(X,M,vel,mass)

fig = plt.figure(figsize=(5,4), dpi=200)
ax = plt.axes(xlim=(1.1*x_min,1.1*x_max), ylim=(1.3*y_min,1.3*y_max))
ax.set_aspect('equal')
ax.tick_params(top=True, right=True, labeltop=True, labelright=True, labelsize=6.0)
ball_1, = ax.plot(pos[0][0],pos[1][0], color='blue', lw = 0.0, markersize=6., marker='o', label='%s Pos. of $m_1$'%mode)
ball_2, = ax.plot(pos[0][1],pos[1][1], color='lime', lw = 0.0, markersize=6., marker='o', label='%s Pos. of $m_2$'%mode)
ball_1_analytic, = ax.plot(pos[0][0],pos[1][0], color='gold', lw = 0.0, markersize=6., marker='*', label='Analytic Pos. of $m_1$')
ball_2_analytic, = ax.plot(pos[0][1],pos[1][1], color='pink', lw = 0.0, markersize=6., marker='*', label='Analytic Pos. of $m_2$')
ax.plot(x_1_analytic, y_1_analytic, lw = 1.25, color='red', ls='--', alpha=0.7, label='Traj. of $m_1$')
ax.plot(x_2_analytic, y_2_analytic, lw = 1.25, color='k', ls='--', alpha=0.7, label='Traj. of $m_2$')
ax.set_xticks(np.linspace(x_min,x_max,5))
ax.set_yticks(np.linspace(y_min,y_max,5))
title = ax.text(0.5, 1.+0.1*(x_max-x_min)/(y_max-y_min), "Time = %.3f $2\pi/\omega$\nError of Energy = %.4e"%(t/period, error_energy),fontsize=6., transform = ax.transAxes, ha='center',va='center')
lgnd = ax.legend(bbox_to_anchor=(1.0, 1.0+0.1*(x_max-x_min)/(y_max-y_min)), prop={'size':3.5}, markerscale=0.5, bbox_transform=ax.transAxes)
#lgnd = ax.legend(loc='lower right', prop={'size':3.5}, markerscale=0.5)

nframe = int( np.ceil( end_t/(drawing_resolution*dt) ) )
anim   = animation.FuncAnimation( fig, func=anime, init_func=init, frames=nframe, interval=10, repeat=False )
plt.show()
t_record = np.array(t_record)
t_record /= period
energy_record = np.array(energy_record)
error_pos = GET_ERROR(pos, pos_analytic)
print("Error of position is %.8e; Error of energy is %.8e."%(error_pos,energy_record[-1]))
