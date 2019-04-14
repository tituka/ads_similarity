import random
import unittest
import argparse
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

"""Sign in to plotly"""
py.sign_in('ttkall', 'aFp6ewOL9Y45R4t9403i')

"""Parse command line argument data"""
ap = argparse.ArgumentParser()
ap.add_argument('-vectors', type=str, default='vectors/gensim_glove_vectors25.txt')
ap.add_argument('-word', type=str, default="vectors")
ap.add_argument('-color', type=str, default="blue")
ap.add_argument('-d', action='store_true',
                    default=False,
                    dest='double_cycle',
                    help='Use double cycle')
ap.add_argument('-man', type=str, default=None)
ap.add_argument('-woman', type=str, default=None)
ap.add_argument('-king', type=str, default=None)

args = vars(ap.parse_args())

color = args['color']
word = args['word']
man = args['man']
woman = args['woman']
king = args['king']
double_cycle = args['double_cycle']
word_vectors = KeyedVectors.load_word2vec_format(args['vectors'], binary=False, unicode_errors='ignore')
print("vectors loaded")


def solid_shape(dim:int, color:str, word:str):
    """Draws a solid shape based on word for given vector . Better with 50 dimensions or less."""
    se = word_vectors[word]
    r_s = []
    thetas = []
    for x in range(0, dim):
        thetas.append(2 * np.pi * x / dim)
        r_s.append(se[x])
    thetas.append(2 * np.pi * 0 / dim)
    r_s.append(se[0])

    data = [go.Scatterpolar(r=r_s, theta=thetas, thetaunit="radians", mode='lines', marker=dict(color='peru'),
        line=dict(color="black"), fill="toself", fillcolor=color, )

    ]
    layout = go.Layout(showlegend=False, polar=dict(radialaxis=dict(visible=True)))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='polar-basic')

def two_cycles(dim:int, color:str, word:str):
    """Draws a double cycle shape based on word for given vector , for uneven dimensional vectors gives less precise
    results"""
    se = word_vectors[word]
    """First cycle"""
    r_s = []
    thetas = []
    half_dim = int(dim/2)
    for x in range(0, half_dim):
        thetas.append(2 * np.pi * x / half_dim)
        r_s.append(se[x])
    thetas.append(2 * np.pi * 0 / half_dim)
    r_s.append(se[0])
    """Second cycle (thetas are the same, as both are half of dim)"""
    r_s2 = []
    thetas2 = []
    for x in range(half_dim+1, dim):
        thetas2.append(2 * np.pi * (x - half_dim) / half_dim)
        r_s2.append(1 * se[x])
    thetas2.append(2 * np.pi * 1 / half_dim)
    r_s2.append(1 * se[half_dim+1])

    data = [go.Scatterpolar(r=r_s, theta=thetas, thetaunit="radians", mode='lines', marker=dict(color='peru'),
        line=dict(color="black"), fill=None, fillcolor=color, ),

        go.Scatterpolar(r=r_s2, theta=thetas2, thetaunit="radians", mode='lines', marker=dict(color='peru'),
            line=dict(color="blue"), fill='tonext', fillcolor=color,

        ),

    ]

    layout = go.Layout(showlegend=False, polar=dict(radialaxis=dict(visible=True)))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='polar-basic')

def man_king(dim, color, word, man, woman, king):
    """Displays the calculated vector for the analogy like man:king, woman:queen, as well as the vector for the
    analogous word, where the variable word is in the place of queen"""
    se = word_vectors[word]
    man_vec=word_vectors[man]
    woman_vec=word_vectors[woman]
    king_vec=word_vectors[king]

    """The polar coordinates for the cycle for the variable word"""
    r_s = []
    thetas = []
    for x in range(0, dim):
        thetas.append(2 * np.pi * x / dim)
        r_s.append(se[x])
    thetas.append(2 * np.pi * 0 / dim)
    r_s.append(se[0])

    """The polar coordinates for the vector calculated from the other parts of the analogy"""
    r_s_calc = []
    for x in range(0, dim):
        r_s_calc.append(woman_vec[x]+king_vec[x]-man_vec[x])
    r_s_calc.append(woman_vec[0]+king_vec[0]-man_vec[0])

    data = [go.Scatterpolar(r=r_s, theta=thetas, thetaunit="radians", mode='lines', marker=dict(color='peru'),
        line=dict(color="black"), fill=None, fillcolor=color, ),

        go.Scatterpolar(r=r_s_calc, theta=thetas, thetaunit="radians", mode='lines', marker=dict(color='peru'),
            line=dict(color="blue"), fill='tonext', fillcolor=color,

        ),

    ]

    layout = go.Layout(showlegend=False, polar=dict(radialaxis=dict(visible=True)))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='polar-basic')


def visualize(double_cycle):
    dim = word_vectors.vector_size
    if man==None:
        dim=word_vectors.vector_size
        if double_cycle:
            two_cycles(dim, color, word)
        else:
            solid_shape(dim, color, word)
    else:
        man_king(dim, color, word, man, woman, king)

visualize(double_cycle)