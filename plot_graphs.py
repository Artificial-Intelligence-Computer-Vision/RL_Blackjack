from header_import import *


class Value_Policy_Graph_Plot(object):
    def __init__(self):
        self.path = "graphs_charts/"

    def Plot_Graphs_Value(self, q_value, policy, graph_title, value = False):
    
        dealer_sum = list(range(1, 11))
        player_sum = list(range(12, 22))
        X_axis, Y_axis = np.meshgrid(dealer_sum, player_sum)
        stable_ace = np.zeros([10,10])
        unstable_ace = np.zeros([10,10])
    
        if value == False:
            for player in range(12, 22):     
                for dealer in range(1, 11):
                    unstable_ace[player-12, dealer-1] = q_value[((False,player,dealer), policy[(False,player,dealer)])]
                    stable_ace[player-12, dealer-1] = q_value[((True,player,dealer), policy[(True, player,dealer)])]
    
            self.Plot(X_axis, Y_axis, stable_ace, "{} (Stable Ace)".format(graph_title))
            self.Plot(X_axis, Y_axis, unstable_ace, "{} (Usable Ace)".format(graph_title))    
        
        else:
            for player in range(12, 22):     
                for dealer in range(1, 11):
                    unstable_ace[player-12, dealer-1] = q_value[(False, player, dealer)]
                    stable_ace[player-12, dealer-1] = q_value[(True, player, dealer)]
        
            self.Plot(X_axis, Y_axis, stable_ace, "{} (Stable Ace)".format(graph_title))
            self.Plot(X_axis, Y_axis, unstable_ace, "{} (Usable Ace)".format(graph_title))
        
    
    def Plot(self, X_axis, Y_axis, value, title):
        figure = plt.figure(figsize = (20, 10))
        axis = figure.add_subplot(projection = '3d')
        surface = axis.plot_surface(X_axis, Y_axis, value, rstride = 1, cstride = 1, cmap = matplotlib.cm.coolwarm, vmin = -1, vmax = 1)
        axis.set_ylabel('Player Sum')
        axis.set_xlabel('Dealer Showing')
        axis.set_zlabel('Value')
        axis.set_title(r'$v_*$ '+ title, fontsize=24)
        axis.view_init(axis.elev, -30)
        figure.colorbar(surface)
        plt.savefig((str(self.path) + title + "_value_graphspng"), dpi =500)



    def Plot_Graphs_policy(self, policy, graph_title):
    
        stable_ace = np.zeros([10,10])
        unstable_ace = np.zeros([10,10])
    
        for player in range(12, 22):     
            for dealer in range(1, 11):
                unstable_ace[player-12, dealer-1] =  policy[(False,player, dealer)]
                stable_ace[player-12, dealer-1] = policy[(True,player, dealer)]
    
        self.Plot_policy(stable_ace, "{} (Stable Ace)".format(graph_title))
        self.Plot_policy(unstable_ace, "{} (Usable Ace)".format(graph_title))
        
        
    def Plot_policy(self, policy, title):
        figure = plt.figure(figsize=[30,15])
        axis = plt.subplot2grid([2,3], [0, 0], fig=figure)
        axis.imshow(policy, origin='lower', cmap = matplotlib.cm.coolwarm, alpha=0.3, extent=[0.5,10.5,11.5,21.5], vmin=-1, vmax=1, interpolation='none')
        axis.set_xticks(np.arange(1,11, 1))
        axis.set_yticks(np.arange(12,22, 1))
        axis.set_title(r'$\pi_*$  ' + title , fontsize=24)
        axis.text(8, 20, 'STICK')
        axis.text(8, 13, 'HIT')
        plt.savefig((str(self.path) + title + "_policy_graphs.png"), dpi =500)

