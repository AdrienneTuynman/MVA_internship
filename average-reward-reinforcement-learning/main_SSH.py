from warnings import filterwarnings

import environments.RegisterEnvironments as bW
import learners.discreteMDPs.IRL_BPI as IRL_BPI
import learners.discreteMDPs.EBTCI as EBTCI
import learners.discreteMDPs.Uniform as Uniform
import learners.discreteMDPs.UCRL2 as UCRL2
import learners.discreteMDPs.sr_sensitive as sr
import learners.discreteMDPs.srs_1 as sr1

# import learners.discreteMDPs.IRL as IRL
# import learners.discreteMDPs.PSRL as psrl
# import learners.discreteMDPs.UCRL3 as le
# import learners.Generic.Qlearning as ql
# import learners.Generic.Random as random
from experiments.fullExperiment_BPI import *


nbReplicates = 32
# filterwarnings("ignore")
# To get the list of registered environments:
# print("List of registered environments: ")
# [print(k) for k in bW.registerWorlds.keys()]

#######################
# Pick an environment
#######################

# env = bW.makeWorld(bW.registerWorld("river-swim-6"))
# env = bW.makeWorld(bW.registerWorld("ergo-river-swim-6"))
# eps = 0.05
# printevery = 100
# testevery = 10
# timeHorizon = 301

# env = bW.makeWorld(bW.registerWorld("grid-2-room"))
# eps = 1
# printevery = 2000
# testevery = 2000
# timeHorizon = 50001

env = bW.makeWorld(bW.registerWorld("grid-4-room"))
eps = 3e-2
printevery = 500000
testevery = 100000
timeHorizon = 50000001  # 00000


# env = bW.makeWorld(bW.registerWorld("river-swim-25"))
# epsilon = 0.001

# env = bW.makeWorld(bW.registerWorld("ergo-river-swim-25"))

# env = bW.makeWorld(bW.registerWorld("grid-random-88"))
# epsilon = 0.01

# env = bW.makeWorld(bW.registerWorld('grid-random-1212'))
# env = bW.makeWorld(bW.registerWorld('grid-random-1616'))
# env = bW.makeWorld(bW.registerWorld('random-rich'))


# env = bW.makeWorld(bW.registerWorld("ergodic-random-rich"))
# eps = 1e-2
# printevery = 10000
# testevery = 10000
# timeHorizon = 4000001


# env = bW.makeWorld(bW.registerWorld("nasty"))
# epsilon = 1e-1

nbr_s = env.observation_space.n
nbr_a = env.action_space.n
delta = 0.8

#######################
# Select tested agents
#######################

agents = []

# Stopping interval based agents
agents.append(
    [
        sr.Srs2tt,
        {"nbr_actions": nbr_a, "nbr_states": nbr_s, "delta": delta, "epsstop": eps},
    ]
)
agents.append(
    [
        sr1.Srs_1,
        {"nbr_actions": nbr_a, "nbr_states": nbr_s, "delta": delta, "epsstop": eps},
    ]
)
# agents.append(
#     [
#         sr.Srs2,
#         {"nbr_actions": nbr_a, "nbr_states": nbr_s, "delta": delta, "epsstop": eps},
#     ]
# )
agents.append(
    [
        sr.Srs1,
        {"nbr_actions": nbr_a, "nbr_states": nbr_s, "delta": delta, "epsstop": eps},
    ]
)
agents.append(
    [
        sr.Armdpgape,
        {"nbr_actions": nbr_a, "nbr_states": nbr_s, "delta": delta, "epsstop": eps},
    ]
)

agents.append(
    [
        EBTCI.EBTCI,
        {"nbr_states": nbr_s, "nbr_actions": nbr_a, "epsilon": eps * 1e-2},
    ]
)

# # # # Baseline agents
# agents.append(
#     [
#         IRL_BPI.IRL_BPI,
#         {"nbr_states": nbr_s, "nbr_actions": nbr_a, "epsilon": eps * 1e-2},
#     ]
# )
agents.append(
    [
        UCRL2.UCRL2,
        {
            "nbr_actions": nbr_a,
            "nbr_states": nbr_s,
            "delta": delta,
            "epsilon": eps * 1e-2,
        },
    ]
)
agents.append(
    [
        Uniform.Uniform,
        {"nbr_states": nbr_s, "nbr_actions": nbr_a, "epsilon": eps * 1e-2},
    ]
)


#######################
# Run a full experiment
#######################
runLargeMulticoreExperiment(
    env,
    agents,
    delta,
    eps,
    timeHorizon,
    nbReplicates,
    testevery,
    printevery,
    verbose=True,
    compare=False,
)
