#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/13  22:28
# @File:  ont.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import gym
import time

env = gym.make("LunarLander-v2", render_mode="human")
if True:
    state = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        print("state = {0}; reward={1}".format(state, reward))
        if done:
            print("game over")
            break
        time.sleep(0.01)
env.close()
