import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3:
    """The DDPG Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.      
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma

        #Initialize d, the updating delay constant.
        self.d = 2

        # Initialize Critics networks and targets network.s Should be named self.Critic1 and self.Critic2
        # Now we have twin networks
        self.Critic_1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic_target_1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic_2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic_target_2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)

        #Copy the weights of the Critics to the target Critics
        copy_target(self.Critic_target_1,self.Critic_1)
        copy_target(self.Critic_target_2,self.Critic_2)

        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)

        copy_target( self.Actor_target,self.Actor) # Copy the weights of the Actor to the target Actor


        # Define the optimizers for the actor and critic networks as proposed in the paper
        self.optim_critic_1 = optim.Adam(self.Critic_1.parameters(), lr=0.001)
        self.optim_critic_2 = optim.Adam(self.Critic_2.parameters(), lr=0.001)
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.001)


    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps

        # We use here Noise instead of Gaussian to add some exploration to the agent. OU noise is a stochastic process
        # that generates a random sample from a Gaussian distribution whose value at time t depends on the previous value
        # x(t) and the time elapsed since the previous value y(t). It helps to explore the environment better than Gaussian noise.
        # This line initializes the noise with mean 0 and sigma 0.15 (see Noise.py file)

        NNoise =  NormalActionNoise(mean=np.zeros(self.act_dim),sigma=0.1) #TD3 paper uses 0.1 sigma

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.

            epsilon= np.clip(NNoise.sample(), a_min=-0.5, a_max=0.5) #We clip the noise to be between -0.5 and 0.5
            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)
                batch = self.replay_buffer.get(self.batch_size)
                # Get the batch data
                state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch

                # Compute the losses for the critics and update the critics network
                critic_loss_1,critic_loss_2 = self.compute_critic_loss(batch)

                self.optim_critic_1.zero_grad() # Zero the gradients
                critic_loss_1.backward() # Compute the gradients
                self.optim_critic_1.step() # Update the weights

                #Same for the second critic
                self.optim_critic_2.zero_grad()
                critic_loss_2.backward()
                self.optim_critic_2.step()

                # Update the actor network and update targets every d steps
                if timestep % self.d == 0:
                    # Compute the loss for the actor and update the actor network
                    actor_loss = self.compute_actor_loss(batch)

                    self.optim_actor.zero_grad()
                    actor_loss.backward()
                    self.optim_actor.step()




                    #Now we update two critic targets
                    soft_update(self.Actor_target,self.Actor, tau=0.005)
                    soft_update(self.Critic_target_1, self.Critic_1,  tau=0.005)
                    soft_update(self.Critic_target_2, self.Critic_2, tau=0.005)



            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-15:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        # by the OrnsteinUhlenbeckActionNoise in the main loop.

        # Convert the state to a tensor
        self.Actor.eval() # Set the actor network to evaluation mode
        with torch.no_grad(): # No need to compute gradients
            s = torch.tensor(s, dtype=torch.float32).to(device)

            # Get the action from the actor network
            a = self.Actor(s).detach().cpu().numpy()

        self.Actor.train() # Set the actor network to training mode
        return a


    def compute_critic_loss(self, batch):
        """
        The function computes the critic loss using the Mean Squared Bellman Error (MSBE) calculation.
        
        :param batch: The `batch` parameter is a tuple containing the data for computing the loss.
        :return: the critic loss, which is calculated using the mean squared error (MSE) loss between
        the expected Q-values (q_expected) and the target Q-values (target).
        """
        
        # similar to the DQN loss.

        # Implement MSBE calculation (need to sample from replay buffer first)
        # Get the data. Should be 6 numpy arrays of size batch_size
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch

        # Move data to Tensor and also to device to take profit of GPU if available
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        terminated_batch = torch.FloatTensor(terminated_batch).to(dtype=torch.long).to(device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(dtype=torch.long).to(device).unsqueeze(1)

        with torch.no_grad(): # No need to compute gradients
            a_next_batch = self.Actor_target(next_state_batch)
            q_targets_next_1   = self.Critic_target_1(next_state_batch,a_next_batch)
            q_targets_next_2   = self.Critic_target_2(next_state_batch,a_next_batch)

            #Now we choose the pessimistic target critic network
            target = reward_batch + ((1-terminated_batch) *self.gamma* torch.min(q_targets_next_1,q_targets_next_2))

        # We calculate the loss for both critic networks
        q_expected_1 = self.Critic_1(state_batch,action_batch)
        q_expected_2 = self.Critic_2(state_batch,action_batch)

        criterion = nn.MSELoss()

        loss_1 = criterion(q_expected_1, target)
        loss_2 = criterion(q_expected_2,target)#MSE

        return loss_1,loss_2
    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # The loss is the negative mean of the expect ed Q-values.

        # Get the data. Should be 6 numpy arrays of size batch_size
        state_batch, action_batch, _, _, _, _ = batch

        # Move data to Tensor and also to device to      take profit of GPU if available
        state_batch = torch.FloatTensor(state_batch).to(device)

        for param in self.Critic_1.parameters(): # We don't want to compute gradients for the critic network
            param.requires_grad = False

        action_next_batch = self.Actor(state_batch) # Get the action from the actor network

        # Compute the Q-values for the state_batch according to the Critic network
        q_values = self.Critic_1(state_batch,action_next_batch)

        # Compute the loss for the actor network
        loss = -q_values.mean()

        for param in self.Critic_1.parameters(): # We want to compute gradients for the critic network onwards
            param.requires_grad = True


        return loss



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)




if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array')

    ddpg = TD3(env,replay_size=1000000, batch_size=100, gamma=0.99)

    ddpg.learn(1000000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video', episode_trigger=lambda x: True)

    video_agent(env, ddpg,n_episodes=5)
    pass
