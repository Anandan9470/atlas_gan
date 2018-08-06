import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.colors import from_levels_and_colors
from stats import wasserstein_distance


#path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"
path = "/afs/inf.ed.ac.uk/user/s17/s1749290/Code/"

class Plot_and_save(object):

    def __init__(self):

        data = np.loadtxt(path+"data/data_v2/65/coordinate_transformation/coordinate_transformation.csv", delimiter=',')
        self.data = data/data.max()

        self.E_l0_65 = self.data[:,:10].sum(axis=1)
        self.E_l1_65 = self.data[:,10:110].sum(axis=1)
        self.E_l2_65 = self.data[:,110:210].sum(axis=1)
        self.E_l3_65 = self.data[:,210:220].sum(axis=1)
        self.E_l12_65 = self.data[:,220:230].sum(axis=1)
        self.E_ltot_65 = self.data.sum(axis=1)

        self.dst_l0_65_list = []
        self.dst_l1_65_list = []
        self.dst_l2_65_list = []
        self.dst_l3_65_list = []
        self.dst_l12_65_list = []
        self.dst_tot_65_list = []
        
        data = np.loadtxt(path+"data/data_v2/524/coordinate_transformation/coordinate_transformation.csv", delimiter=',')
        self.data = data/data.max()

        self.E_l0_524 = self.data[:,:10].sum(axis=1)
        self.E_l1_524 = self.data[:,10:110].sum(axis=1)
        self.E_l2_524 = self.data[:,110:210].sum(axis=1)
        self.E_l3_524 = self.data[:,210:220].sum(axis=1)
        self.E_l12_524 = self.data[:,220:230].sum(axis=1)
        self.E_ltot_524 = self.data.sum(axis=1)

        self.dst_l0_524_list = []
        self.dst_l1_524_list = []
        self.dst_l2_524_list = []
        self.dst_l3_524_list = []
        self.dst_l12_524_list = []
        self.dst_tot_524_list = []

    def sample_images(self, samples, epoch, energy_point, is_last_epoch):

        l0 = samples[:,:10].sum(axis=1)
        l1 = samples[:,10:110].sum(axis=1)
        l2 = samples[:,110:210].sum(axis=1)
        l3 = samples[:,210:220].sum(axis=1)
        l12 = samples[:,220:230].sum(axis=1)
        ltot = samples.sum(axis=1)
        
        if energy_point == '65':
            E_l0 = self.E_l0_65
            E_l1 = self.E_l1_65
            E_l2 = self.E_l2_65
            E_l3 = self.E_l3_65
            E_l12 = self.E_l12_65
            E_tot = self.E_ltot_65
        elif energy_point == '524':
            E_l0 = self.E_l0_524
            E_l1 = self.E_l1_524
            E_l2 = self.E_l2_524
            E_l3 = self.E_l3_524
            E_l12 = self.E_l12_524
            E_tot = self.E_ltot_524
            
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(231)
        ax1.hist(E_l0, bins=50, histtype=u'step',label='Original')
        ax1.hist(l0, bins=50, histtype=u'step',label='Generated')
        ax1.set_title('Layer 0')
        ax1.legend()

        ax2 = fig.add_subplot(232)
        ax2.hist(E_l1, bins=50, histtype=u'step',label='Original')
        ax2.hist(l1, bins=50, histtype=u'step',label='Generated')
        ax2.set_title('Layer 1')
        ax2.legend()

        ax3 = fig.add_subplot(233)
        ax3.hist(E_l2, bins=50, histtype=u'step',label='Original')
        ax3.hist(l2, bins=50, histtype=u'step',label='Generated')
        ax3.set_title('Layer 2')
        ax3.legend(bbox_to_anchor=(0.01, 1), loc=2)

        ax4 = fig.add_subplot(234)
        ax4.hist(E_l3, bins=50, histtype=u'step',label='Original')
        ax4.hist(l3, bins=50, histtype=u'step',label='Generated')
        ax4.set_title('Layer 3')
        ax4.legend()

        ax5 = fig.add_subplot(235)
        ax5.hist(E_l12, bins=50, histtype=u'step',label='Original')
        ax5.hist(l12, bins=50, histtype=u'step',label='Generated')
        ax5.set_title('Layer 12')
        ax5.legend()

        ax6 = fig.add_subplot(236)
        ax6.hist(E_tot, bins=50, histtype=u'step',label='Original')
        ax6.hist(ltot, bins=50, histtype=u'step',label='Generated')
        ax6.set_title('All layers')
        ax6.legend(bbox_to_anchor=(0.01, 1), loc=2)
        if energy_point == '65':
            fig.savefig("out/evolution_hist_65/sample_hist_%d.png" % epoch)
        elif energy_point == '524':
            fig.savefig("out/evolution_hist_524/sample_hist_%d.png" % epoch)
        plt.close()

        dst_l0 = wasserstein_distance(l0, E_l0)
        dst_l1 = wasserstein_distance(l1, E_l1)
        dst_l2 = wasserstein_distance(l2, E_l2)
        dst_l3 = wasserstein_distance(l3, E_l3)
        dst_l12 = wasserstein_distance(l12, E_l12)
        dst_tot = wasserstein_distance(ltot, E_tot)

        print('EPOCH: {}; L0: {:.3}; L1: {:.3}; L2: {:.3}; L3: {:.3}; L12: {:.3}; TOT: {:.3}'
              .format(epoch, dst_l0, dst_l1, dst_l2, dst_l3, dst_l12, dst_tot))

        if energy_point == '65':

            self.dst_l0_65_list.append(dst_l0)
            self.dst_l1_65_list.append(dst_l1)
            self.dst_l2_65_list.append(dst_l2)
            self.dst_l3_65_list.append(dst_l3)
            self.dst_l12_65_list.append(dst_l12)
            self.dst_tot_65_list.append(dst_tot)
            
        elif energy_point == '524':
            
            self.dst_l0_524_list.append(dst_l0)
            self.dst_l1_524_list.append(dst_l1)
            self.dst_l2_524_list.append(dst_l2)
            self.dst_l3_524_list.append(dst_l3)
            self.dst_l12_524_list.append(dst_l12)
            self.dst_tot_524_list.append(dst_tot)           

        if is_last_epoch:

            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(231)
            ax1.plot(self.dst_l0_65_list)
            ax1.set_title('Layer 0')

            ax2 = fig.add_subplot(232)
            ax2.plot(self.dst_l1_65_list)
            ax2.set_title('Layer 1')

            ax3 = fig.add_subplot(233)
            ax3.plot(self.dst_l2_65_list)
            ax3.set_title('Layer 2')

            ax4 = fig.add_subplot(234)
            ax4.plot(self.dst_l3_65_list)
            ax4.set_title('Layer 3')

            ax5 = fig.add_subplot(235)
            ax5.plot(self.dst_l12_65_list)
            ax5.set_title('Layer 12')

            ax6 = fig.add_subplot(236)
            ax6.plot(self.dst_tot_65_list)
            ax6.set_title('All layers')
            fig.savefig("out/metric/metric65.png")
            plt.close()

            dsts = np.vstack([self.dst_l0_65_list,
                               self.dst_l1_65_list,
                               self.dst_l2_65_list,
                               self.dst_l3_65_list,
                               self.dst_l12_65_list,
                               self.dst_tot_65_list])

            np.savetxt("out/metric/metric65.csv", dsts, delimiter=',')
            
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(231)
            ax1.plot(self.dst_l0_524_list)
            ax1.set_title('Layer 0')

            ax2 = fig.add_subplot(232)
            ax2.plot(self.dst_l1_524_list)
            ax2.set_title('Layer 1')

            ax3 = fig.add_subplot(233)
            ax3.plot(self.dst_l2_524_list)
            ax3.set_title('Layer 2')

            ax4 = fig.add_subplot(234)
            ax4.plot(self.dst_l3_524_list)
            ax4.set_title('Layer 3')

            ax5 = fig.add_subplot(235)
            ax5.plot(self.dst_l12_524_list)
            ax5.set_title('Layer 12')

            ax6 = fig.add_subplot(236)
            ax6.plot(self.dst_tot_524_list)
            ax6.set_title('All layers')
            fig.savefig("out/metric/metric524.png")
            plt.close()

            dsts = np.vstack([self.dst_l0_524_list,
                               self.dst_l1_524_list,
                               self.dst_l2_524_list,
                               self.dst_l3_524_list,
                               self.dst_l12_524_list,
                               self.dst_tot_524_list])

            np.savetxt("out/metric/metric524.csv", dsts, delimiter=',')


