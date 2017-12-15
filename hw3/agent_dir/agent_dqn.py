from agent_dir.agent import Agent
from collections import deque
import os
import json
import random
import numpy as np

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        self.state_size = [1,84,84,4]
        self.action_size = env.get_action_space().n
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount rate
        self.target_freq = 100
        self.nn_freq = 1
        self.learning_rate = 0.001
        self.model_name = args.model_name or "breakout"
        self.death = args.death_time or 50000
        self.record = []
        self.nb_step = 0
        self.batch_size = 64
        
        self.args = args
        self.make_action_f = np.ones((1,self.action_size),dtype="float64")


        if args.test_dqn:
            self.epsilon = 0.01  # exploration rate
            self.epsilon_range = 0.0
            self.epsilon_ratio = 0.0
            #you can load your model here
            print('loading trained model')
            self.load()
            self.model.summary()
            self.target_model = None

        ##################
        # YOUR CODE HERE #
        ##################
        elif args.train_dqn:
            self.epsilon = 1.0  # exploration rate
            self.epsilon_range = 0.95
            self.epsilon_ratio = 0.02

            print('Building new model')
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.model.summary()
            with open("model/%s.json" % self.model_name, 'w') as f:
                json.dump(self.model.to_json(), f)
            if not (os.path.exists("img/%s/" % self.model_name)):
                os.makedirs("img/%s/" % self.model_name)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass
        ##################
        # YOUR CODE HERE #
        ##################
        

    def build_model(self):
        import keras.models as kmodel
        from keras.layers import Dense, Reshape, Flatten, Input, Dot, LeakyReLU, Multiply, MaxPool2D
        from keras.optimizers import Adam,RMSprop
        from keras.layers.convolutional import Convolution2D

        state_input = Input([84,84,4])
        action_input = Input([self.action_size])

        cnn = Convolution2D(32, (3,3), activation='relu', padding='same', input_shape=(84, 84, 4))(state_input)
        cnn = MaxPool2D((2,2))(cnn)
        cnn = Convolution2D(64, (3,3), activation='relu', padding='same')(cnn)
        cnn = MaxPool2D((2,2))(cnn)

        dnn = Flatten()(cnn)
        dnn = Dense(512, activation='relu')(dnn)
        dnn = Dense(self.action_size, activation='linear')(dnn)

        result = Multiply()([dnn,action_input])

        model = kmodel.Model([state_input,action_input],result)

        opt = RMSprop(lr=self.learning_rate,rho=0.99)
        model.compile(loss='mse', optimizer=opt)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def fit(self):
        minibatch = random.sample(self.memory, self.batch_size)

        t_stack = []
        s_stack = []
        a_stack = []

        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, self.state_size)
            next_state = np.reshape(next_state, self.state_size)
            target = reward + (1-done)* self.gamma * np.amax(self.target_model.predict([next_state,self.make_action_f])[0])

            action_f = np.zeros((1,self.action_size),dtype="float64")
            target_f = np.zeros((1,self.action_size),dtype="float64")
            action_f[0][action] = 1.0
            target_f[0][action] = target
            t_stack.append(target_f)
            s_stack.append(state)
            a_stack.append(action_f)

        t_stack = np.vstack(t_stack)
        a_stack = np.vstack(a_stack)
        s_stack = np.vstack(s_stack)
        self.model.train_on_batch([s_stack,a_stack], t_stack)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        self.model.save_weights(os.path.join('model',"%s_model_weight.hdf5" % self.model_name))

    def load(self):
        from keras.models import load_model,model_from_json
        from keras.utils.vis_utils import plot_model

        self.model = model_from_json(json.load(open("model/%s.json" % self.model_name)))
        #plot_model(self.model,to_file=os.path.join("img","breakout.png"),show_shapes = True)
        self.model.load_weights(os.path.join('model',"%s_model_weight.hdf5" % self.model_name))

    def save_record(self):
        import csv

        x = []
        y = []

        for k in range(0,len(self.record)):
            x.append(self.record[k][0])
            size = min(self.record[k][0],30)
            ma_score = np.array([ self.record[t][1] for t in range(self.record[k][0] - size,self.record[k][0]) ])
            y.append(ma_score.mean() )

        with open(os.path.join("img",self.model_name,"score.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(x)
            writer.writerow(y)

    def plot_img(self):
        import matplotlib.pyplot as plt

        x = []
        y = []

        for k in range(0,len(self.record)):
            x.append(self.record[k][0])
            size = min(self.record[k][0],30)
            ma_score = np.array([ self.record[t][1] for t in range(self.record[k][0] - size,self.record[k][0]) ])
            y.append(ma_score.mean() )

        plt.plot(x,y)
        plt.savefig(os.path.join("img",self.model_name,"score.png"))
        # save img


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        #return self.env.get_random_action()
        ##################
        # YOUR CODE HERE #
        ##################
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        observation = np.reshape(observation, self.state_size)
        act_values = self.model.predict([observation,self.make_action_f])[0]
        return np.argmax(act_values)

    def real_test(self):
        from test import test
        from environment import Environment

        e_bp = self.epsilon
        self.epsilon = 0.01
        test_env = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
        test(self, test_env, total_episodes=100)
        self.epsilon = e_bp

    def train(self):
        """
        Implement your training algorithm here
        """
        #pass
        ##################
        # YOUR CODE HERE #
        ##################
        done = False
        score = 0
        episode = 0
        state = self.env.reset()
        
        while True:
            action = self.make_action(state,test=False)
            next_state, reward, done, info = self.env.step(action)
            self.nb_step += 1
            score += reward
            self.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                episode += 1
                print('Step: %d - Episode: %d - Score: %f - E : %f. ' % (self.nb_step,episode, score, self.epsilon))
                self.record.append([episode, score])
                score = 0
                state = self.env.reset()
                decay = float(episode)/(self.epsilon_ratio*self.death) * self.epsilon_range
                self.epsilon = max(1.0 - decay, 1.0 - self.epsilon_range)
                if episode > 1 and episode % self.nn_freq == 0 and len(self.memory) > self.batch_size:
                    self.fit()
                if episode > 1 and episode % self.target_freq == 0:
                    self.update_target()
                if episode > 1 and episode % 10 == 0:
                    self.save()
                    self.save_record()
                # if episode > 1 and episode % 1000 == 0:
                #     self.real_test()
                # if self.nb_step >= self.death :
                if episode >= self.death :
                    self.save()
                    self.save_record()
                    self.plot_img()
                    return