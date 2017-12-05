from agent_dir.agent import Agent
import os
import json
import numpy as np

# def preprocess(arr):
#     from scipy.misc import imresize
#     bg = (144, 72, 17)

#     arr = arr[35:195,:,:]
#     arr = arr[:,20:-16,:]
#     arr = imresize(arr, (80,62), interp="nearest")

#     grayscale = np.zeros((arr.shape[0],arr.shape[1]))
#     for i in range(0,arr.shape[0]):
#         for j in range(0,arr.shape[1]):
#             # Precision 7
#             #if arr[i][j].shape[0] == 3 :
#             cond = (arr[i][j][0],arr[i][j][1],arr[i][j][2]) == bg
#             grayscale[i][j] = 0 if cond else 1
#             #elif arr[i][j].shape[0] == 1 :
#                 #grayscale[i][j] = arr[i][j][0]

#     #print(grayscale.sum(axis=0))
#     #grayscale = (grayscale - grayscale.min())/(grayscale.max()+0.00001)
#     return np.expand_dims(grayscale,axis=2)

def preprocess(I):
    I = I[35:195]
    #I = I[:,20:-16]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    I = I.astype(np.float).ravel()
    return I.reshape((80,80,1))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.state_size = 80*62
        #self.action_size = env.get_action_space().n
        self.action_size = 3
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.death = args.death_time
        self.model_name = args.model_name or "pong"
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.prev_x = None
        self.record = []

        if args.test_pg:
            print('Loading trained model')
            #you can load your model here
            self.load()
            self.model.summary()
            #model_dir = os.path.join('model',model_name)
            

        ##################
        # YOUR CODE HERE #
        ##################
        if args.train_pg:
            print('Building new model')
            self.model = self.build_model()
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
        #pass
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None


    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense, Reshape, Flatten
        from keras.optimizers import Adam
        from keras.layers.convolutional import Convolution2D

        model = Sequential()
        #model.add(Reshape((1, 80, 80), ))
        model.add(Convolution2D(32, (6, 6), activation="relu", padding="same", input_shape=(80, 80, 1), strides=(3, 3), kernel_initializer="he_uniform"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model


    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        #return self.env.get_random_action()

        ##################
        # YOUR CODE HERE #
        ##################
        state = preprocess(observation)
        x = state - self.prev_x if self.prev_x is not None else np.zeros(state.shape)
        self.prev_x = state

        x = np.expand_dims(x,axis=0)
        aprob = self.model.predict(x, batch_size=1).flatten()
        self.probs.append(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        if test:
            return action+1
        else :
            return action,aprob,x


    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    def fit(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards-np.mean(rewards)) / np.std(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]),axis = 1)
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        #Y[np.where(Y < 0.0)] = 0.0
        #Y = softmax(Y)
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []


    def save(self):
        self.model.save_weights(os.path.join('model',"%s_model_weight.hdf5" % self.model_name))

    def load(self):
        from keras.models import load_model,model_from_json

        self.model = model_from_json(json.load(open("model/%s.json" % self.model_name)))
        self.model.load_weights("model/%s_model_weight.hdf5" % self.model_name)

    def plot_img(self):
        import matplotlib.pyplot as plt
        #r = np.array(self.record)

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

    def train(self):
        """
        Implement your training algorithm here
        """
        #pass
        ##################
        # YOUR CODE HERE #
        ##################
        score = 0
        episode = 0
        state = self.env.reset()
        #state = preprocess(state)
        prev_x = None

        while True:
            #self.env.render()

            # cur_x = state
            # x = cur_x - prev_x if prev_x is not None else np.zeros(state.shape)
            # prev_x = cur_x
            # print(x.shape)
            try:
                action, prob, x = self.make_action(state,test=False)
                state, reward, done, info = self.env.step(action+1) #+1 is important
                score += reward
                # state = preprocess(state)
                self.remember(x, action, prob, reward)

                if done:
                    episode += 1
                    print('Episode: %d - Score: %f.' % (episode, score))
                    self.record.append([episode, score])
                    score = 0
                    state = self.env.reset()
                    self.prev_x = None
                    # if episode > 1 and episode % 5 == 0:
                    self.fit()
                    if episode > 1 and episode % 10 == 0:
                        self.save()
                        self.save_record()
                        #self.plot_img()
                    if episode == self.death :
                        #self.fit()
                        self.save()
                        self.save_record()
                        self.plot_img()
                        return
            except KeyboardInterrupt:
                self.fit()
                self.save()
                self.save_record()
                self.plot_img()
                return

    

