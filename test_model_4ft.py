import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet_4ft import AlexNet
import torch
from getkeys import key_check

import random

WIDTH = 200
HEIGHT = 150
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'pygta5-{}-{}-{}-epochs.model_with_4'.format(LR, 'alexnet', EPOCHS)

t_time = 0.09


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    time.sleep(t_time)
    ReleaseKey(D)


def slow():
    PressKey(S)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(S)


model = AlexNet().cuda()
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()


for name, param in model.named_parameters():
    if param.requires_grad:
        pass#print(name, param)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False

    while(True):
        
        if not paused:

            # 800x600 windowed mode

            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))

            screen = grab_screen(region=(0,40,800,640))

            print('loop took {} seconds'.format(time.time()-last_time))

            last_time = time.time()

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            inp = np.reshape(screen, (1,1,WIDTH,HEIGHT))
            print(inp.shape)
            #inp = [screen.reshape(WIDTH,HEIGHT,1)]
            prediction = model.forward(torch.cuda.FloatTensor(inp))[0]

            print(prediction)

            turn_thresh = .75

            fwd_thresh = 0.70

            slow_tresh = 0.8

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            elif prediction[3] > slow_tresh:
                slow()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.

        if 'T' in keys:

            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
