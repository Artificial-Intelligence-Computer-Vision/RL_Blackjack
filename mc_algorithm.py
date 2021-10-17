from header_import import *

class BlackJack_First_Visit_MC_Prediction_Value(object):
    def __init__(self, number_of_episode, epsilon = 0.1, gamma = 1):
        self.number_of_episode = number_of_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = []
        self.number_of_state = 20
        
        self.Returns = defaultdict(float)
        self.return_count = defaultdict(float)
        self.value = dict()
        
    def policy_player_(self, player_sum, dealer_card, unstable_ace):
        return 0 if player_sum > 19 else 1
    
    def First_Visit_MC_Prediction_Value(self, bjack):
        
        for i in range(self.number_of_episode):
            if i % 1000 == 0:
                print("\rNumber of episode {}/{}.".format(i, self.number_of_episode), end="")
            
            # Generate Episode
            self.episode = []
            state = (False, 0, 0)
            
            for i in range(self.number_of_state):
                next_state, reward, trajectory = bjack.play(self.policy_player_)
                done, action = map(list, zip(*trajectory))
                
                # Recalibrate
                next_state = tuple(next_state)
                action = int(str(action)[1:-1][0])
                done = done[0][0]
                state = next_state
                self.episode.append((state, action, reward))
                if done:
                    break
                    
            G = 0
            for state, action, reward in reversed(self.episode):
                G = self.gamma * G + reward

                if not state in [self.episode[i][0] for i in range((len(self.episode))-1)]:
                    self.return_count[state]  += 1
                    self.Returns[state] += G
    
        for state, returns in self.Returns.items():
            self.value[state] = returns / self.return_count[state]
        
        self.q_value = defaultdict(float, self.value)
        return self.q_value



class BlackJack_MC_Prediction_Value_With_Exploring(object):
    def __init__(self, number_of_episode, epsilon = 0.1, gamma = 1):
        self.number_of_episode = number_of_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = []
        self.number_of_state = 20
        self.q_value = defaultdict(float)
        self.policy = defaultdict(int)
        self.policy_value = defaultdict(int) 
        self.Returns = defaultdict(list)
        
    def policy_player_(self, player_sum, dealer_card, unstable_ace):
        return 0 if player_sum > 19 else 1
                
    def MC_Prediction_Value_With_Exploring(self, bjack):
        
        for i in range(self.number_of_episode):
            if i % 1000 == 0:
                print("\rNumber of episode {}/{}.".format(i, self.number_of_episode), end="")
            
            # Generate Episode
            self.episode = []
            state = (False, 0, 0)
            
            for i in range(self.number_of_state):
                next_state, reward, trajectory = bjack.play(self.policy_player_)
                done, action = map(list, zip(*trajectory))
                
                # Recalibrate
                next_state = tuple(next_state)
                action = int(str(action)[1:-1][0])
                done = done[0][0]
                state = next_state
                self.episode.append((state, action, reward))
                if done:
                    break
                
            G = 0
            for state, action, reward in reversed(self.episode):
                
                G = self.gamma * G + reward
                
                if not (state, action) in [(self.episode[i][0], self.episode[i][1]) for i in range((len(self.episode))-1)]:
                    self.Returns[(state, action)].append(G)
                    self.q_value[(state, action)] = np.average(self.Returns[(state, action)])
                    self.policy[state] = np.argmax([self.q_value[state,a] for a in range(2)])
                    if np.argmax([self.q_value[state,a] for a in range(2)]) == 1:
                        self.policy_value[state] = np.argmax([self.q_value[state,a] for a in range(2)])
                    else:
                        self.policy_value[state] = np.argmin([self.q_value[state,a] for a in range(2)])
                            
        return self.q_value, self.policy_value, self.policy



class BlackJack_MC_Prediction_Value_With_Importance_Sampling(object):
    def __init__(self, number_of_episode, epsilon = 0.1, gamma = 1):
        self.number_of_episode = number_of_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = []
        self.number_of_state = 2
        self.behavior_policy = 0.5
        self.state_sum = defaultdict(int)
        self.state_count = defaultdict(float)
        self.q_value = defaultdict(lambda: np.zeros(2))
        self.value = defaultdict(float)
        self.policy = defaultdict(int)
        self.inportance_sampling = defaultdict(lambda: np.zeros(2))
        
    def target_policy(self, state):
        action = np.zeros_like(self.q_value[state], dtype=float)
        best_action = np.argmax(self.q_value[state])
        action[best_action] = 1.0
        return action
        
    def policy_player_(self, player_sum, dealer_card, unstable_ace):
        return 0 if player_sum > 19 else 1
    
    def MC_Prediction_Value_With_Importance_Sampling(self, bjack):
                
        for i in range(self.number_of_episode):
            if i % 1000 == 0:
                print("\rNumber of episode {}/{}.".format(i, self.number_of_episode), end="")
            
            # Generate Episode
            self.episode = []
            state = (False, 0, 0)
            
            for i in range(self.number_of_state):
                next_state, reward, trajectory = bjack.play(self.policy_player_)
                done, action = map(list, zip(*trajectory))
                
                # Recalibrate
                next_state = tuple(next_state)
                action = int(str(action)[1:-1][0])
                done = done[0][0]
                state = next_state
                self.episode.append((state, action, reward))
                if done:
                    break
                
            G = 0
            self.weights = 1.0
            for state, action, reward in reversed(self.episode):
                G = self.gamma * G + reward
                self.inportance_sampling[state][action] += self.weights
                self.q_value[state][action] += (self.weights / self.inportance_sampling[state][action]) * (G - self.q_value[state][action])
                if action !=  np.argmax(self.target_policy(state)):
                    break
                    
                self.weights = (self.weights * (1.0/ self.behavior_policy))
                                           
        self.policy = dict((state, np.argmax(action)) for state,action  in self.q_value.items())
        self.policy = defaultdict(float, self.policy)
        
        for state, action in self.q_value.items():
            if np.max((action)) == 0:
                self.value[state] = np.min((action))
            else:
                self.value[state] = np.max((action))
            
        return self.value, self.policy
