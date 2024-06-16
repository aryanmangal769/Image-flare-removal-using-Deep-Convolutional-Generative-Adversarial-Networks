import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

f = open("combined_result.txt", "r")

ff = f.readlines()

gen_loss = []
disc_loss = []

for values in ff:
    vv = values.split(",")
    gen_loss.append(float(vv[0]))
    disc_loss.append(float(vv[1]))


gen_loss = [float('%.3f'%(i/3207)) for i in gen_loss]
disc_loss = [float('%.3f'%(round(i, 3)/3207)) for i in disc_loss]

fig, ax = plt.subplots(1, 2, figsize=(30, 30))
ax[0].plot(np.array(gen_loss), label = "Generator Loss", color = "green")
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0].set_xlabel("Epoch")
ax[1].plot(np.array(disc_loss), label = "Discriminator Loss", color = "orange")
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1].set_xlabel("Epoch")
ax[0].grid(color = 'grey', linestyle = '--', linewidth = 0.5)
ax[1].grid(color = 'grey', linestyle = '--', linewidth = 0.5)
ax[0].legend()
ax[1].legend()
plt.show()
# fig, ax = plt.subplots(1, 2)

# ax[0].plot(gen_loss)
# ax[1].plot(disc_loss)
# plt.show()