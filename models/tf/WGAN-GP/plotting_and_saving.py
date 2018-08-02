import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from scipy.stats import entropy

path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"

class Plot_and_save(object):

    def __init__(self):

        data = np.loadtxt(path+"data/data_v2/baseline/baseline.csv", delimiter=',')
        self.data = data/data.max()

        self.E_l0 = self.data[:,:1000].sum(axis=1)
        self.E_l1 = self.data[:,1000:2000].sum(axis=1)
        self.E_l2 = self.data[:,2000:3000].sum(axis=1)
        self.E_l3 = self.data[:,3000:4000].sum(axis=1)
        self.E_l12 = self.data[:,4000:5000].sum(axis=1)
        self.E_ltot = self.data.sum(axis=1)

        self.chi2_l0_list = []
        self.chi2_l1_list = []
        self.chi2_l2_list = []
        self.chi2_l3_list = []
        self.chi2_l12_list = []
        self.chi2_tot_list = []

    def make_2D_plots(self, samples, epoch):

        gen_imgs = samples[np.random.choice(samples.shape[0], 4, replace=False)]
        gen_imgs = np.log(gen_imgs+10e-5)

        r, c = 2, 2
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):

                img = np.reshape(gen_imgs[cnt], newshape=(10,23), order='F')

                num_levels = 20
                vmin, vmax = img.min(), img.max()
                midpoint = 0
                levels = np.linspace(vmin, vmax, num_levels)
                midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
                vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
                colors = plt.cm.seismic(vals)
                cmap, norm = from_levels_and_colors(levels, colors)

                im = axs[i,j].imshow(img, cmap=cmap, norm=norm, interpolation='none')
                fig.colorbar(im, ax=axs[i,j])
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig("out/sample_%d.png" % epoch)
        plt.close()

    def sample_images(self, samples, epoch, is_last_epoch):

        l0 = samples[:,:1000].sum(axis=1)
        l1 = samples[:,1000:2000].sum(axis=1)
        l2 = samples[:,2000:3000].sum(axis=1)
        l3 = samples[:,3000:4000].sum(axis=1)
        l12 = samples[:,4000:5000].sum(axis=1)
        ltot = samples.sum(axis=1)

        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(231)
        ax1.hist(self.E_l0, bins=50, histtype=u'step',label='Truth')
        ax1.hist(l0, bins=50, histtype=u'step',label='Processed')
        ax1.set_title('Layer 0')
        ax1.legend()

        ax2 = fig.add_subplot(232)
        ax2.hist(self.E_l1, bins=50, histtype=u'step',label='Truth')
        ax2.hist(l1, bins=50, histtype=u'step',label='Processed')
        ax2.set_title('Layer 1')
        ax2.legend()

        ax3 = fig.add_subplot(233)
        ax3.hist(self.E_l2, bins=50, histtype=u'step',label='Truth')
        ax3.hist(l2, bins=50, histtype=u'step',label='Processed')
        ax3.set_title('Layer 2')
        ax3.legend()

        ax4 = fig.add_subplot(234)
        ax4.hist(self.E_l3, bins=50, histtype=u'step',label='Truth')
        ax4.hist(l3, bins=50, histtype=u'step',label='Processed')
        ax4.set_title('Layer 3')
        ax4.legend()

        ax5 = fig.add_subplot(235)
        ax5.hist(self.E_l12, bins=50, histtype=u'step',label='Truth')
        ax5.hist(l12, bins=50, histtype=u'step',label='Processed')
        ax5.set_title('Layer 12')
        ax5.legend()

        ax6 = fig.add_subplot(236)
        ax6.hist(self.E_ltot, bins=50, histtype=u'step',label='Truth')
        ax6.hist(ltot, bins=50, histtype=u'step',label='Processed')
        ax6.set_title('All layers')
        ax6.legend()
        fig.savefig("out/evolution_hist/sample_hist_%d.png" % epoch)
        plt.close()

        chi2_l0 = entropy(l0, self.E_l0)
        chi2_l1 = entropy(l1, self.E_l1)
        chi2_l2 = entropy(l2, self.E_l2)
        chi2_l3 = entropy(l3, self.E_l3)
        chi2_l12 = entropy(l12, self.E_l12)
        chi2_tot = entropy(ltot, self.E_ltot)

        print('EPOCH: {}; L0: {:.3}; L1: {:.3}; L2: {:.3}; L3: {:.3}; L12: {:.3}; TOT: {:.3}'
              .format(epoch, chi2_l0, chi2_l1, chi2_l2, chi2_l3, chi2_l12, chi2_tot))

        self.chi2_l0_list.append(chi2_l0)
        self.chi2_l1_list.append(chi2_l1)
        self.chi2_l2_list.append(chi2_l2)
        self.chi2_l3_list.append(chi2_l3)
        self.chi2_l12_list.append(chi2_l12)
        self.chi2_tot_list.append(chi2_tot)

        if is_last_epoch:

            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(231)
            ax1.plot(self.chi2_l0_list)
            ax1.set_title('Layer 0')

            ax2 = fig.add_subplot(232)
            ax2.plot(self.chi2_l1_list)
            ax2.set_title('Layer 1')

            ax3 = fig.add_subplot(233)
            ax3.plot(self.chi2_l2_list)
            ax3.set_title('Layer 2')

            ax4 = fig.add_subplot(234)
            ax4.plot(self.chi2_l3_list)
            ax4.set_title('Layer 3')

            ax5 = fig.add_subplot(235)
            ax5.plot(self.chi2_l12_list)
            ax5.set_title('Layer 12')

            ax6 = fig.add_subplot(236)
            ax6.plot(self.chi2_tot_list)
            ax6.set_title('All layers')
            fig.savefig("out/chi2/chi2.png")
            plt.close()

            chi2s = np.vstack([self.chi2_l0_list,
                               self.chi2_l1_list,
                               self.chi2_l2_list,
                               self.chi2_l3_list,
                               self.chi2_l12_list,
                               self.chi2_tot_list])

            np.savetxt("out/chi2/chi2.csv", chi2s, delimiter=',')



