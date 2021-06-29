# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:50:33 2021

@author: Matheus Ribeiro Fernandes
"""

#imports
import time
import numpy as np
import random as rnd
import pyautogui # click buttons and automation
from PIL import ImageGrab #get game state and resize
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#hyperparameters
alpha = 0.1
gamma = 1
epsilon = 0.99
epsilon_decay = 0.99

#screen information and possible moves
screen_region = [0, 0, 1024, 768]

class DQN:
    def __init__(self, pixels, possible_moves):
        #hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        #experience replay to avoid outliers
        self.memory = []
        self.batch_size = 32
        
        #AI model
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(pixels,), activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(possible_moves, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        mem_sample = rnd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in mem_sample:
            q = self.model.predict(state)
            q_update = self.alpha * (reward + self.gamma * np.max(self.model.predict(next_state)))
            q[0][action] = q_update
            self.model.fit(state, q, verbose=0)
        #decaying epsilon when the AI gets updated
        self.epsilon *= epsilon_decay
        
    def memory_add(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def choose_action(self, state):
        #exploration
        if rnd.random() < self.epsilon:
            return rnd.choice([0,1,2,3,4])
        #exploitation
        return np.argmax(self.model.predict(state))

#             0:center      1:up         2:down      3:right      4:left
moves_pos = [[512, 384], [512, 100], [512, 500], [650, 384], [400, 384]]
def do_move(move):
    pyautogui.moveTo(moves_pos[0])
    pyautogui.mouseDown(button='left')
    pyautogui.moveTo(moves_pos[move])
    pyautogui.mouseUp(button='left')
    
def image_located_on_screen(image):
    image = pyautogui.locateOnScreen(image, region=(0,0,1024,768),
                                     grayscale=True, confidence = 0.8)
    return image

def get_state():
    img = ImageGrab.grab(bbox=(screen_region))
    img = img.resize((60,60))
    img = img.convert('P')
    img = np.array(img)
    img = img.reshape(1, 3600)
    return img
    
def subway_surfers():
    pixels = 3600
    possible_moves = 5 # 0, 1, 2, 3, 4
    dqn = DQN(pixels, possible_moves)
    print('Ai initialized...')
    
    start_screen = image_located_on_screen('play_screen.png')
    if bool(start_screen) == True:
        pyautogui.click(moves_pos[0])
        time.sleep(2)
        
        #start game loop
        while True:
            is_alive = True
            while is_alive:
                #get game state and predict move
                state = get_state()
                action = dqn.choose_action(state)
                do_move(action)
                
                #verify if he still alive after the move and set reward
                is_alive = bool(image_located_on_screen('alive.png'))
                
                if is_alive == True:
                    reward = 1
                else:
                    reward = -10
                    
                #get the next state and add epoch to AI memory
                next_state = get_state()
                dqn.memory_add(state, action, reward, next_state)
            
            #train the AI
            dqn.update()
            
            #restart game
            time.sleep(2)
            play_button = image_located_on_screen('play_button.png')
            pyautogui.click(play_button.left, play_button.top)
            time.sleep(2)
            
            
subway_surfers()
