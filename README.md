# The Pacman AI Project

*Special acknowledgements to UC Berkeley intially developing this project and providing all supporting features.*

<span style="color: red; font-weight: bolder;">Warning: I do guarantee the correctness of the solution since it's graded by the autograder, and I know this course does not change its assignments across terms. However, please respect the honor code  regulations, and I don't take any responsibility if you choose to make any violation.</span>

# Introduction

This project consists of four parts which correspond to four programming assignments that I have completed or will complete throughout the course.

## Project I

### Search

- search.py
    - Depth First Search
    - Breadth First Search
    - Uniform Cost Search
    - A* Search
- searchAgents.py
    - Defining Corners Search Problem
    - Designing a Heuristic for Using A* in Corners Search Problem
    - Designing a Heuristic for Solving Eating All Dots Problem Using A*
    - Suboptimal Search to Eat All Dots by Greedy Method
    - Mini Contest: Approximate a Short Path to Eat All Dots
    
**Score: 23/20** full credit including extra credit! I believe no one has ever earned full-credit for the mini-contest question!

## Project II

### Markov Decision Process

#### Value Iteration & Q-learning

- valueIterationAgents.py
    - implementing value iteration algorithm to solve fully observable MDPs
- qlearningAgents.py
    - implementing q-learning algorithm to solve partially observable MDPs
- analysis.py
    - completing a few analysis questions by filling up parameters for value iteration and q-learning agents

*A large part of this project involves interactions of the agents in the grid world in which the actions they take are stochastic. Afterwards, the MDP model solved using q-learning is applied in the real pacman game to solve tiny maze problems with one ghost.*

**Score: 22/20** full credit including extra credit!

## Projecrt III

### Tracking by Reasoning Uncertainty

#### Files Editted
- inference.py
- busterAgents.py

#### Topics
- Calculating Exact Probabilities
- Approximating Using Particle Filters
- Joint Particle Filter

#### Questions
- Exact Inference Observation
- Exact Inference with Time Elapse
- Exact Inference Full Test
- Approximate Inference Observation
- Approximate Inference with Time Elapse
- Joint Particle Filter Observation
- Joint Particle Filter Elapse Time

**Score: 25/17** full credit including extra credit!
  
## Project IV

### Optimization with Machine Learning

### Details
- see pdfs

**Score: 23/20** not a technical problem, but lost 1 point due to unclear description in the instruction pdf :(
