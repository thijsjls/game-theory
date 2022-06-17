"""
NASH EQUILIBRIUM SOLVER
Author: Thijs Sluijter, Universiteit van Amsterdam, 11050721

This is a program to find Nash equilibria (pure and mixed) in normalform games
with 2 players and 2 actions each.

The game input format is the following:

    game = [a(Top, Left), a(Top, Right), a(Bottom, Left), a(Bottom, Right)]
    
an actionprofile like a(Top, Left) is represented as a tuple of utilities:

    a(Top, Left) = (u_rowena, u_colin)
    
To run the program enter a game in the main function. An example of the prisoners
dilemma game is provided there as well.
"""

import numpy as np
import random

"""
Helper function to create graphic representation of a 2x2 normalform game
input: a game
output: prints a 2x2 normalform game
"""
def print_game(game):
    print()
    print(' '*17+"Colin")
    print(' '*14+'L'+' '*8+'R')
    print(' '*11+'-'*17)
    print(' '*10+'| '+"{:>6}".format(game[0][1])+' | '+"{:>6}".format(game[1][1])+' |')
    print(' '*8+'T | '+"{:<6}".format(game[0][0])+' | '+"{:<6}".format(game[1][0])+' |')
    print('Rowena'+' '*5+'-'*17)
    print(' '*10+'| '+"{:>6}".format(game[2][1])+' | '+"{:>6}".format(game[3][1])+' |')
    print(' '*8+'B | '+"{:<6}".format(game[2][0])+' | '+"{:<6}".format(game[3][0])+' |')
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
Helper function to determine wether player p has a dominating strategy in a game
Note that '>=' is used, therefor these strategies are not strictly dominating
input: a game
output: 
        1 if player p always plays Left or Top
        2 if player p always plays Bottom or Right
        None if no dominating strategy exists for player p
"""      
def dom_strat(game, p):
    u1,u2 = u(game)
    if p == "R":
        if u1[0,0] >= u1[1,0] and u1[0,1] >= u1[1,1]:
            return 1
        elif u1[1,0] >= u1[0,0] and u1[1,1] >= u1[0,1]:
            return 2
    elif p == "C":
        if u2[0,0] >= u2[0,1] and u2[1,0] >= u2[1,1]:
            return 1
        elif u2[0,1] >= u2[0,0] and u2[1,1] >= u2[1,0]:
            return 2
    else:
        return None


"""
Function to calculate pure Nash equilibria of a game
input: a game
output: list of PNE's represented by a tuple containing the strategies
        of Colin and Rowena e.g. a PNE in a(Top, Left) is represented as (1, 1)
"""  
def pure_NE(game):
    u1,u2 = u(game)
    PNE = []
    if u1[0,0] >= u1[1,0] and u2[0,0] >= u2[0,1]:
        PNE.append((1,1))
    if u1[0,1] >= u1[1,1] and u2[0,1] >= u2[0,0]:
        PNE.append((1,0))
    if u1[1,1] >= u1[0,1] and u2[1,1] >= u2[1,0]:
        PNE.append((0,0))
    if u1[1,0] >= u1[0,0] and u2[1,0] >= u2[1,1]:
        PNE.append((0,1))
    return PNE


"""
Function to calculate mixed Nash equilibria of a game
input: a game
output:
        list of equilibria represented by the strategies of Rowena and Colin
        e.g. (0.5, 0.75) represents Rowena playing Top or Bottom both with 50% probability
            and Colin playing Left with 75% probability and Right with 25% probability
        when the output contains a probability of the form: [0,1] this represents all infinitely
        many probabilities between zero and one (including zero and one)
        when the output contains a probability of the form: p + eps this represents all infinitely
        many probabilities between p and one (including one)
        when the output contains a probability of the form: p - eps this represents all infinitely
        many probabilities between p and zero (including zero)
"""
def mixed_NE(game):
    u1,u2 = u(game)
    equilibria = []
    # try solving the strategy equations assuming indiference
    try:
        p = np.linalg.solve(np.array([[((u2[0,0]-u2[1,0])-(u2[0,1]-u2[1,1]))]]),np.array([[-u2[1,0]+u2[1,1]]]))[0,0]
        # if the result is not a probability between 0 and 1, treat it like it's unsolvable
        if not(p <= 1 and p >= 0):
            s_c = dom_strat(game, 'C')
            # if a dominating strategy exists for colin
            if s_c:
                p = s_c%2
                i = p+1%2
                # check wether Rowena has a dominating strategy on that side of the game
                if u1[0,i] > u1[1,i]:
                    return [(1,p)]
                elif u1[0,i] < u1[1,i]:
                    return [(0,p)]
                else:
                # if her utilities are the same she can play any strategy
                    return [('[0,1]',p)]
            else:
                p = None
    except:
        p = None
    try:
        q = np.linalg.solve(np.array([[((u1[0,0]-u1[0,1])-(u1[1,0]-u1[1,1]))]]),np.array([[-u1[0,1]+u1[1,1]]]))[0,0]
        # if the result is not a probability between 0 and 1, treat it like it's unsolvable
        if not(q <= 1 and q >= 0):
            s_r = dom_strat(game, 'R')
            if s_r:
                q = s_r%2
                i = q+1%2
                # check wether Colin has a dominating strategy on that side of the game
                if u2[0,i] > u2[1,i]:
                    return [(1,p)]
                elif u2[0,i] < u2[1,i]:
                    return [(0,p)]
                else:
                # if his utilities are the same he can play any strategy
                    return [('[0,1]',p)]
            else:
                q = None
    except:
        q = None
    # if Rowena is unsolvable check Colin
    if not p:
        if not q:
            # if both are unsolvable check wether there is a dominating strategy
            s_r = dom_strat(game,'R')
            s_c = dom_strat(game,'C')
            # if Rowena has a dominating strategy return this as her pure strategy
            if s_r:
                 # check Colin and do the same
                if s_c:
                    return [(s_r%2,s_c%2)]
                else:
                    return [(s_r%2,'[0,1]')]
            elif s_c:
                return [('[0,1]',s_c%2)]
            else:
                return [('[0,1]','[0,1]')]
        # if only Rowena is unsolvable, there are 3 types of equilibrium
        else:
            # equlibria of type ([0,1], q)
            equilibria.append(('[0,1]',q))
            # equilibria of type (1 or 0, q + eps)
            if q < 1: #check edge case
                if u1[0,0] > u1[1,0]:
                    equilibria.append((1,str(q)+'+eps'))
                else:
                    equilibria.append((0,str(q)+'+eps'))
            #equilibria of type (1 or 0, p - eps)
            if u1[0,1] > u1[1,1]:
                equilibria.append((1,str(q)+'-eps'))
            else:
                equilibria.append((0,str(q)+'-eps'))
            return equilibria
    # if only Colin is unsolvable, there are again 3 types of equilibrium
    elif not q:
        # equlibria of type (p, [0,1])
        equilibria.append((p, '[0,1]'))
        # equilibria of type (p + eps, 1 or 0)
        if p < 1: #check edge case
            if u2[0,0] > u2[0,1]:
                equilibria.append((str(p)+'+eps',1))
            else:
                equilibria.append((str(p)+'+eps',0))
        #equilibria of type (p - eps, 1 or 0)
        if u1[1,0] > u1[1,1]:
            equilibria.append((str(p)+'-eps',1))
        else:
            equilibria.append((str(p)+'-eps',0))
        return equilibria
    # if both are solvable return the results
    else:
        equilibria.append((p, q))
        return equilibria


"""
Main function. Takes a game, prints it, calculates the Nash equilibria
and prints those as a set of strategies.
"""  
def main():
    
    # Example games:
    # game = [(random.randint(-50, 50), random.randint(-50, 50)) for i in range(4)] # random game
    game = [(-10,-10),(-25,0),(0,-25),(-20,-20)]                                  # prisoners dilemma (as defined in class)
    # game = [(3,5),(2,7),(1,6),(8,4)]                                              # hw1 2a
    # game = [(2,7),(2,5),(2,1),(2,7)]                                              # hw1 2b
    # game = [(1,-1),(-1,1),(-1,1),(1,-1)]                                          # equal coins
    # game = [(2,2),(1,0),(0,1),(0,0)]                                              # omerta prisoners dilemma (hw2 1b)
    
    print_game(game)

    PNE = pure_NE(game)
    NE = mixed_NE(game)

    print("Nash equilibria (pure and mixed):")
    print(set(PNE+NE))

if __name__ == "__main__": 
    main()
        