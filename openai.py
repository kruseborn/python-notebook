import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

class NetworkData(object):
    __slots__ = ['stateSize', 'actionSize', 'gamma', 'memory', 'epsilon', 'epsilonMin', 'epsilonDecay', 'learningRate']

def createNetworkData():
    networkData = NetworkData()

    networkData.stateSize = 4 
    networkData.actionSize = 2
    networkData.memory = deque(maxlen=2000)
    networkData.gamma = 0.95
    networkData.epsilon = 1.0
    networkData.epsilonMin = 0.01
    networkData.epsilonDecay = 0.995
    networkData.learningRate = 1e-3
    
    return networkData;

def createNetwork(data):
    network = Sequential()
    network.add(Dense(24, input_dim=data.stateSize, activation='relu'))
    network.add(Dense(24, activation='relu'))
    network.add(Dense(networkData.actionSize, activation='linear'))
    network.compile(loss='mse', optimizer=Adam(lr=data.learningRate))

    return network

def createAction(network, data, state):
    if np.random.rand() < data.epsilon:
        return random.randrange(data.actionSize)
    predictedAction = network.predict(state)
    return np.argmax(predictedAction[0])

def trainNetwork(network, data, state, size):
    trainingSample = random.sample(data.memory, size)
    for state, nextState, action, reward, done in trainingSample:
        target = reward
        if not done:
            target = reward + data.gamma * np.amax(network.predict(nextState)[0])
            targetF = network.predict(state)
            targetF[0][action] = target
            network.fit(state, targetF, epochs=1, verbose=0)
        if data.epsilon > data.epsilonMin:
            data.epsilon *= data.epsilonDecay


def storeState(memory, state, nextState, action, reward, done):
    memory.append((state, nextState, action, reward, done))

data = createNetworkData()
network = createNetwork(data)

env = gym.make('CartPole-v1')
episodes = 1
frames =  1
trainingSize = 50
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, data.stateSize])
    for frame in range(frames):
        #env.render()
        action = createAction(network, data, state)
        nextState, reward, done, info = env.step(action)
        nextState = np.reshape(nextState, [1, data.stateSize])

        storeState(data.memory, state, nextState, action, reward, done)
        state = nextState
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(episode, episodes, frame, data.epsilon))
            break

    if len(data.memory) > trainingSize:
        trainNetwork(network, data, state, trainingSize)

