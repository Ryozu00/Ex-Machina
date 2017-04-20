from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# Make data.
#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
X, Y = np.meshgrid(X, Y)
# https://en.wikipedia.org/wiki/Rosenbrock_function
Z = -((1+np.cos(12*np.sqrt(X**2+Y**2)))/(0.5*(X**2+Y**2)+2))

num_swarm = 100
num_func_params = 2
position = -3 + 6 * np.random.rand(num_swarm, num_func_params)
velocity = np.zeros([num_swarm, num_func_params])
personal_best_position = np.copy(position)
personal_best_value = np.zeros(num_swarm)

predator_swarm = 100
predator_params = 2
predator = -3 + 6 * np.random.rand(predator_swarm, predator_params)
predator_velocity = np.zeros([predator_swarm, predator_params])
predator_best_position = np.copy(predator)
predator_best_value = np.zeros(predator_swarm)

for i in range(num_swarm):
    # Z = (1-X)**2 + 1 *(Y-X**2)**2
    personal_best_value[i] = -((1+np.cos(12*np.sqrt(position[i][0]**2+position[i][1]**2)))/(0.5*(position[i][0]**2+position[i][1]**2)+2))

for i in range(predator_swarm):
    predator_best_value[i] = -((1+np.cos(12*np.sqrt(predator[i][0]**2+predator[i][1]**2)))/(0.5*(predator[i][0]**2+predator[i][1]**2)+2))

# #3D
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.1, linewidth=0, antialiased=False)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_autoscalex_on(False)
# ax.set_xlim([-3, 3])
# ax.set_autoscaley_on(False)
# ax.set_ylim([-3, 3])
# ax.set_autoscalez_on(False)
# ax.set_zlim([-1, 1])

#dot size
size = 0.09

#Predators var
pc1 = 0.002
pc2 = 0.003
boolp = []

#Foods var
tmax = 200
c1 = 0.001
c2 = 0.002
levels = np.linspace(-1, 35, 100)


for i in range(num_swarm):
    boolp.append(1)

global_best = np.min(personal_best_value)
global_best_position = np.copy(personal_best_position[np.argmin(personal_best_value)])

predator_global_best = np.min(predator_best_value)
predator_global_best_position = np.copy(predator_best_position[np.argmin(predator_best_value)])

for t in range(tmax):
    #3D
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.1, linewidth=0, antialiased=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-3, 3])
    ax.set_autoscaley_on(False)
    ax.set_ylim([-3, 3])
    ax.set_autoscalez_on(False)
    ax.set_zlim([-1, 1])

    for i in range(num_swarm):
        error = -((1+np.cos(12*np.sqrt(position[i][0]**2+position[i][1]**2)))/(0.5*(position[i][0]**2+position[i][1]**2)+2))
        if personal_best_value[i] > error:
            personal_best_value[i] = error
            personal_best_position[i] = position[i]

    for i in range(predator_swarm):
        error = -((1+np.cos(12*np.sqrt(predator[i][0]**2+predator[i][1]**2)))/(0.5*(predator[i][0]**2+predator[i][1]**2)+2))
        if predator_best_value[i] > error:
            predator_best_value[i] = error
            predator_best_position[i] = predator[i]

    best = np.min(personal_best_value)
    best_index = np.argmin(personal_best_value)
    if global_best > best:
        global_best = best
        global_best_position = np.copy(personal_best_position[best_index])

    predator_best = np.min(predator_best_value)
    predator_best_index = np.argmin(predator_best_value)
    if predator_global_best > predator_best:
        predator_global_best = best
        predator_global_best_position = np.copy(predator_best_position[predator_best_index])
#Foods
    # for k in range(predator_swarm):
    for i in range(num_swarm):
        # update velocity
        velocity[i] += c1 * np.random.rand() * (personal_best_position[i] - position[i]) \
                   + c2 * np.random.rand() * (predator_global_best_position - position[i])
        if (position[i][0] - velocity[i][0] >= 3) or (position[i][0] - velocity[i][0] <= -3):
            velocity[i][0] *= -1
            
        if (position[i][1] - velocity[i][1] >= 3) or (position[i][1] - velocity[i][1] <= -3):
            velocity[i][1] *= -1
        # position[i] -= velocity[i]
        position[i][0] -= velocity[i][0]    
        position[i][1] -= velocity[i][1]
#Predators
    for i in range(predator_swarm):
        temp_x = position[0][0] - predator[i][0]
        temp_y = position[0][1] - predator[i][1]
        length = np.sqrt(np.power(temp_x, 2.0) + np.power(temp_y, 2.0))
        # temp_length = length
        # if length < 0:
        #     temp_length = temp_length * -1

        for k in range(num_swarm):
            if boolp[k] == 1:
                ttemp_x = position[k][0] - predator[i][0]
                ttemp_y = position[k][1] - predator[i][1]
                tlength = np.sqrt(np.power(ttemp_x, 2.0) + np.power(ttemp_y, 2.0))
                # ttemp_length = tlength
                # if tlength < 0:
                #     ttemp_length = ttemp_length * -1
                # if temp_length > ttemp_length:
                #     length = tlength
                if tlength < length:
                    length = tlength
                    temp_x = ttemp_x
                    temp_y = ttemp_y
                # print("Predator", i, " : ", length)

        predator_velocity[i][0] += pc1 * np.random.rand() * temp_x \
            + pc2 * np.random.rand() * temp_x
        predator_velocity[i][1] += pc1 * np.random.rand() * temp_y \
            + pc2 * np.random.rand() * temp_y
        if (predator[i][0] + predator_velocity[i][0] >= 3) or (predator[i][0] + predator_velocity[i][0] <= -3):
            predator_velocity[i][0] *= -1
        if (predator[i][1] + predator_velocity[i][1] >= 3) or (predator[i][1] + predator_velocity[i][1] <= -3):
            predator_velocity[i][1] *= -1
        predator[i][0] += predator_velocity[i][0]    
        predator[i][1] += predator_velocity[i][1]
# #2D
#     fig = plt.figure()
#     CS = plt.contour(X, Y, Z, levels=levels, cmap=cm.gist_stern)
#     plt.gca().set_xlim([-3, 3])
#     plt.gca().set_ylim([-3, 3])
#     #foods
#     for i in range(num_swarm):
#         if boolp[i] == 1:
#             plt.plot(position[i][0], position[i][1], 'go')
#         # for k in range(predator_swarm):
#         #     if position[i].any == predator[k].any:
#         #         plt.plot(position[i][0], position[i][1], 'bo')
                
#     # plt.plot(global_best_position[0], global_best_position[1], 'ro')
#     #Predators
#     for i in range(predator_swarm):
#         plt.plot(predator[i][0], predator[i][1], 'ro')

#3D
    for i in range(num_swarm):
        if boolp[i] == 1:
            ax.scatter(position[i][0], position[i][1], 0 ,color='blue')
    for i in range(predator_swarm):
        ax.scatter(predator[i][0], predator[i][1], 0 ,color='red')

    for i in range(predator_swarm):
        for j in range(num_swarm):
            if predator[i][0] <= position[j][0] + size and predator[i][0] >= position[j][0] - size and predator[i][1] <= position[j][1] + size and predator[i][1] >= position[j][1] - size:
                boolp[j] = 0 

    # plt.title('{0:03d}'.format(t))
    filename = 'img{0:03d}.png'.format(t)
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()
plt.close(fig)