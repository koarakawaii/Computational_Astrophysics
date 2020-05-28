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

def GET_ERROR(pos,pos_analytic,vel,vel_analytic):
    error_pos = (((pos-pos_analytic)**2).mean()/(pos_analytic**2).mean())**0.5
    error_vel = (((vel-vel_analytic)**2).mean()/(vel_analytic**2).mean())**0.5
    return error_pos, error_vel

def GET_ANALYTIC_RESULT(t, theta_0):
    theta = theta_0+v_mag*t
    x_analytic = np.cos(theta)
    y_analytic = np.sin(theta)
    z_analytic = np.zeros(N)
    pos_analytic = np.vstack((x_analytic,y_analytic,z_analytic))
    vx_analytic = v_mag*np.cos(theta+np.pi/2)
    vy_analytic = v_mag*np.sin(theta+np.pi/2)
    vz_analytic = np.zeros(N)
    vel_analytic = np.vstack((vx_analytic, vy_analytic, vz_analytic))
    return pos_analytic, vel_analytic

def init():
	balls.set_data(pos[0],pos[1])
#	arrows.set_XY(pos[0],pos[1])
	arrows.set_UVC(vel[0],vel[1])
	balls_analytic.set_data(pos[0],pos[1])
#	arrows_analytic.set_XY(pos[0],pos[1])
	arrows_analytic.set_UVC(vel[0],vel[1])
	title.set(text="Time = %.3f $2\pi/\omega$\nError of Energy = %.4e"%(t/period, error_energy))
	return balls, arrows, balls_analytic, arrows_analytic, title

def anime(i):
	global t, pos, vel, pos_analytic, vel_analytic, acc, X
	if mode == 'KDK':
		if t==0:
			X = MAKE_POS_MATRIX(pos)
			acc = GET_ACCELERATION(X,M)
		for step in range(drawing_resolution):
			pos, vel, acc,energy = KDK_SCHEME(pos,vel,M,acc,dt)
			t += dt
			t_record.append(t)
			error_energy = (energy-energy_analytic)/energy_analytic
			energy_record.append(error_energy)
			if t > end_t:
				print("End time reached!")
				break

	elif mode == 'DKD':
		for step in range(drawing_resolution):
			pos, vel,energy = DKD_SCHEME(pos,vel,M,dt)
			t += dt
			t_record.append(t)
			error_energy = (energy-energy_analytic)/energy_analytic
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
			error_energy = (energy-energy_analytic)/energy_analytic
			energy_record.append(error_energy)
			if t > end_t:
				print("End time reached!")
				break
			
	pos_analytic,vel_analytic = GET_ANALYTIC_RESULT(t,theta)
	balls.set_data( pos[0], pos[1] )
	arrows.set_offsets(np.vstack((pos[0],pos[1])).T)
	arrows.set_UVC( vel[0], vel[1] )
	balls_analytic.set_data( pos_analytic[0], pos_analytic[1] )
	arrows_analytic.set_offsets(np.vstack((pos[0],pos[1])).T)
	arrows_analytic.set_UVC( vel_analytic[0], vel_analytic[1] )
	title.set(text="Time = %.3f $2\pi/\omega$\nError of Energy = %.4e"%(t/period, error_energy))
	return balls, arrows, balls_analytic, arrows_analytic, title

###################################################################################################################
N = 3
theta = 2.*np.pi/N*np.arange(0,N)
x = np.cos(theta)
y = np.sin(theta)
z = np.zeros(N)
# x = np.random.normal(loc=5.0, scale=16., size=N)
# y = np.random.normal(loc=-4.0, scale=8., size=N)
# z = np.random.normal(loc=2.2, scale=4., size=N)
pos = np.vstack((x,y,z))
angle = np.arange(0,2.*np.pi,0.01*np.pi)
XX, YY = np.cos(angle), np.sin(angle)

if N==3:
    v_mag = 3**-0.25
elif N==4:
    v_mag = (1/2**0.5+0.25)**0.5
vx = v_mag*np.cos(theta+np.pi/2.)
vy = v_mag*np.sin(theta+np.pi/2.)
vz = np.zeros(N)
# vx = np.random.normal(loc=0.3, scale=2., size=N)
# vy = np.random.normal(loc=-0.1, scale=1.5, size=N)
# vz = np.random.normal(loc=-0.45, scale=1.6, size=N)
mass = np.ones(N, dtype=np.float64)
# mass = 10.*np.random.uniform(size=N)
vel = np.vstack((vx,vy,vz))
X = MAKE_POS_MATRIX(pos)
M = MAKE_MASS_MATRIX(mass)
    
dt = 2.5e-2
t = 0.0
error_energy = 0.0
index, drawing_resolution = 0, 20
period = 2.0*np.pi/v_mag
end_t = 5.*period
mode = 'DKD'
t_record = []
energy_record = []
energy_analytic = GET_ENERGY(X,M,vel,mass)

fig = plt.figure(figsize=(5,4), dpi=200)
ax = plt.axes(xlim=(-1.35,1.35), ylim=(-1.35,1.35))
ax.set_aspect('equal')
ax.tick_params(top=True, right=True, labeltop=True, labelright=True, labelsize=6.0)
balls, = ax.plot(pos[0],pos[1], color='blue', lw = 0.0, markersize=8., marker='o', label='%s Pos.'%mode)
arrows = ax.quiver(pos[0],pos[1],vel[0],vel[1], color='lime', scale=10.0)
balls_analytic, = ax.plot(pos[0],pos[1], color='gold', lw = 0.0, markersize=8., marker='*', label='Analytic Pos.', alpha=0.6)
arrows_analytic = ax.quiver(pos[0],pos[1],vel[0],vel[1], color='k', scale=10.0)
ax.plot(XX, YY, lw = 1.25, color='red', ls='--', alpha=0.7, label='Analytic Traj.')
ax.set_xticks(np.linspace(-1.,1.,5))
ax.set_yticks(np.linspace(-1.,1.,5))
ax.quiverkey(arrows, 0.87,0.15,1.,'%s Vel.'%mode, fontproperties={'size':6.0}, labelsep=0.05)
ax.quiverkey(arrows_analytic, 0.87,0.05,1.,'Analytic Vel.', fontproperties={'size':6.0}, labelsep=0.05)
title = ax.text(0.5, 1.1, "Time = %.3f Period\nError of Energy = %.4e"%(t/period, error_energy),fontsize=6., transform = ax.transAxes, ha='center',va='center')
lgnd = ax.legend(loc='upper right', prop={'size':5.0}, markerscale=0.5)

nframe = int( np.ceil( end_t/(drawing_resolution*dt) ) )
anim   = animation.FuncAnimation( fig, func=anime, init_func=init, frames=nframe, interval=10, repeat=False )
plt.show()
t_record = np.array(t_record)
t_record /= period
energy_record = np.array(energy_record)
error_pos, error_vel = GET_ERROR(pos, pos_analytic, vel, vel_analytic)
print("Error of position is %.8e; error of velocity is %.8e; error of energy is %.8e."%(error_pos,error_vel,energy_record[-1]))
