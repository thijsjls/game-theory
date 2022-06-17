"""
FICTITIOUS PLAY SIMULATOR
Author: Thijs Sluijter, Universiteit van Amsterdam, 11050721

This is a program to simulate fictitious play in normal-form games.

The game input format is the following:

    game = [a(H, H), a(H, T), a(T, H), a(T, T)]
    
an actionprofile like a(H, H) is represented as a tuple of utilities:

    a(H, H) = (u_rowena, u_colin)
    
To run the program enter a game in the main function. An example of the prisoners
dilemma game is provided there as well.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

"""
Helper function to create graphic representation of a 2x2 normalform game
input: a game
output: prints a 2x2 normalform game
"""
def print_game(game):
    print()
    print(' '*17+"Colin")
    print(' '*14+'H'+' '*8+'T')
    print(' '*11+'-'*17)
    print(' '*10+'| '+"{:>6}".format(game[0][1])+' | '+"{:>6}".format(game[1][1])+' |')
    print(' '*8+'H | '+"{:<6}".format(game[0][0])+' | '+"{:<6}".format(game[1][0])+' |')
    print('Rowina'+' '*5+'-'*17)
    print(' '*10+'| '+"{:>6}".format(game[2][1])+' | '+"{:>6}".format(game[3][1])+' |')
    print(' '*8+'T | '+"{:<6}".format(game[2][0])+' | '+"{:<6}".format(game[3][0])+' |')
    print(' '*11+'-'*17)
    print()


"""
Helper function to turn game into numpy arrays of utilities per player (useful for processing)
input: a game
output: 

    tuple of numpy arrays of form:

        [[u(T,L), u(T,R)], [u(B,L), u(B,R)]]

    where u(T,L) is the utility for the player in a(Top, Left)
    
    first array in tuple represents Rowena, second array represents Colin
"""
def u(game):
    return np.array([[game[0][0], game[1][0]], [game[2][0], game[3][0]]]), np.array([[game[0][1], game[1][1]], [game[2][1], game[3][1]]])  

"""
Function that generates constant-sum game
input: minimal and maximal utility
output: 
        a constant sum game
"""    
def constant_sum_game(mi, ma):
    while True:
        constant_sum = True
        # generate a random game
        game = [(random.randint(mi, ma), random.randint(mi, ma)) for i in range(4)]
        c = game[0][0] + game[0][1]
        # check if its constant sum
        for a in game[1:]:
            if a[0] + a[1] != c:
                constant_sum = False
                break
        if constant_sum:
            return game

"""
Function that generates zero-sum game
input: minimal and maximal utility
output: 
        a zero-sum game
"""      
def zero_sum_game(mi, ma):
    while True:
        constant_sum = True
        # generate a random game
        game = [(random.randint(mi, ma), random.randint(mi, ma)) for i in range(4)]
        c = 0
        # check if its zero sum
        for a in game:
            if a[0] + a[1] != c:
                constant_sum = False
                break
        if constant_sum:
            return game

"""
Function that generates zero-sum equivalent of constant-sum game
input:
        a constant-sum game
output: 
        a zero-sum game
"""      
def zero_sum_equivalent(cs_game):
    zs_game = []
    c = cs_game[0][0] + cs_game[0][1]
    for a in cs_game:
        a_new = (a[0]-(c/2), a[1]-(c/2))
        zs_game.append(a_new)
    return zs_game

"""
Function that computes the empirical strategy given a history of actions
input: history of actions
output: 
        the empirical strategy
""" 
def emp_strat(H):
    # H = [1/0, 1/0, ..., 1/0]
    # s(L) = cnt(L) / l
    # s(R) = 1 - s(L)
    return sum(H) / len(H)

"""
Function that generates the best response for a player in a game give a strategy
input: 
        game: game
        player: player
        strategy: s
output: 
        the best response for player in game against strategy s
""" 
def best_response(game, player, s):
    ut = u(game)
    if player == 'R':
        # NOTE: if s = 0.5 we round down, so s <= 0.5 --> a_c = R
        a_c = ut[0].T[(round(s)+1)%2]
        # if both utils are equally good return 0 (T)
        return (a_c.argmax()+1)%2
    else:
        # NOTE: if s = 0.5 we round down, so s <= 0.5 --> a_r = B
        a_r = ut[1][(round(s)+1)%2]
        # if both utils are equally good return 0 (L)
        return (a_r.argmax()+1)%2

"""
Main function that simulates fictitioys play
input: a game
output: strategies played by both players
"""       
def fict_play(zs_game):
    map_r = {1: 'H', 0: 'T'}
    map_c = {1: 'H', 0: 'T'}
    converged = False
    eps = 1e-4
    a = (1,1)
    H_r = [a[0]]
    H_c = [a[1]]
    strats_r = [a[0]]
    strats_c = [a[1]]
    i = 0
    while not converged:
        # calculate empirical strategy for both
        s_r = emp_strat(H_r)
        s_c = emp_strat(H_c)
        # calculate best response for both
        a_r = best_response(zs_game, 'R', s_c)
        a_c = best_response(zs_game, 'C', s_r)
        # print what happened this round
        print(f"a_{len(H_r)-1}: {map_r[a[0]]}{map_c[a[1]]}({s_r},{s_c})")
        # check convergence
        if i > 0 and (abs(strats_r[-1] - s_r) < eps) and (abs(strats_c[-1] - s_c) < eps):
            converged = True
        # save statistics
        a = (a_r,a_c)
        H_r.append(a_r)
        H_c.append(a_c)
        strats_r.append(s_r)
        strats_c.append(s_c)
        i+=1
    return np.array(strats_r), np.array(strats_c)



"""
Main function. Takes a game, prints it, simulates fictious play and plots the strategy convergence.
"""  
def main():
    
    # game = zero_sum_equivalent(constant_sum_game(0,10)) # generate a random game
    game = [(1,-1),(-1,1),(-1,1),(1,-1)]                  # equal coins example

    # print the game
    print_game(game)

    # run fictitious play (and print simulation to terminal)
    s_r, s_c = fict_play(game)

    # plot the convergence pattern
    fig, axs = plt.subplots(1,1)
    axs.plot(np.arange(0,len(s_r)), s_r, color='m', label='Strategy of Rowena')
    axs.plot(np.arange(0,len(s_c)), s_c, color='c', label='Strategy of Colin')
    axs.set_ylabel('Probability of playing H')
    axs.set_xlabel('Number of rounds')
    axs.title.set_text('Strategies for Rowena and Colin during fictitious play')
    axs.legend()
    axs.grid()
    plt.show()

if __name__ == "__main__": 
    main()
        