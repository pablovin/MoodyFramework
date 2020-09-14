from KEF import RenderManager
from KEF import PlotManager
from KEF.PlotManager import plots
from MoodyFramework.Utils import GenerateMoodFromDataset
from MoodyFramework.Mood.Intrinsic import Intrinsic, CONFIDENCE_PHENOMENOLOGICAL

from Agents import AgentRandom, AgentDQL, AgentA2C, AgentPPO

import cv2
import numpy

#Experiment control variables
dataSetLocation = "dataset.pkl" #location of the dataset.PKL file

saveMoodDataset = "" #Location where the Mood dataset will be saved
saveMoodPlot = "" #Location where the Mood Plots will be saved

gameToGenerateMood = 0 # Game from which to generate the mood.

#Agents
agent1 = AgentDQL.AgentDQL([False, 1.0, "DQL"]) #training agent
agent2 = AgentPPO.AgentPPO([False, 1.0, "PPO"]) #training agent
agent3 = AgentA2C.AgentA2C([False, 1.0, "A2C"])  # training agent
agent4 = AgentRandom.AgentRandom(AgentRandom.DUMMY_RANDOM)

agents = [agent1,agent2,agent3,agent4]

DQLModel = "dql.dh5" # Location of the trained DQL model

A2cActor = "a2cActor.dh5" # Location of the trained A2C Actor model
A2cCritic = "a2cCritic.dh5" # Location of the trained A2C Critic model

PPOActor = "ppoActor.dh5" # Location of the trained PPO Actor model
PPOCritic = "ppoCritic.dh5" # Location of the trained PPO Critic model

loadModelAgent1 = DQLModel
loadModelAgent2 = [PPOActor, PPOCritic]
loadModelAgent3 = [A2cActor,
                   A2cCritic]
loadModelAgent4 = ""

loadModel = [loadModelAgent1,loadModelAgent2, loadModelAgent3, loadModelAgent4]


#Faces

agent1 = cv2.imread("/home/pablo/Documents/Datasets/ChefsHat_ReinforcementLearning/MoodTest/000007.jpg")

agentFaces = [agent1,"","",""]

#Mood Controlers
intrinsicWithMoodAgent1 = Intrinsic(selfConfidenceType=CONFIDENCE_PHENOMENOLOGICAL, isUsingSelfMood=True,isUsingOponentMood=True)
intrinsicWithMoodAgent2 = Intrinsic(selfConfidenceType=CONFIDENCE_PHENOMENOLOGICAL, isUsingSelfMood=True,isUsingOponentMood=True)
intrinsicWithMoodAgent3 = Intrinsic(selfConfidenceType=CONFIDENCE_PHENOMENOLOGICAL, isUsingSelfMood=True,isUsingOponentMood=True)

intrinsicMoods = [intrinsicWithMoodAgent1, intrinsicWithMoodAgent2, intrinsicWithMoodAgent3]

#
# Generate MoodDataset
GenerateMoodFromDataset.generateMoodFromDataset(intrinsicModels=intrinsicMoods, dataset=dataSetLocation, agents=agents, agentFaces=agentFaces, loadModels=loadModel,saveDirectory = saveMoodDataset)

# Generate Plots
moodDataset = saveMoodDataset + "/IntrinsicDataset.pkl"
plotsToGenerate = [plots["Experiment_Mood"], plots["Experiment_MoodNeurons"],
                   plots["Experiment_SelfProbabilitySuccess"]]

plot = PlotManager.generateIntrinsicPlotsFromDataset(plotsToGenerate=plotsToGenerate, IntrinsicDataset=moodDataset, gameNumber=2, saveDirectory=saveMoodPlot)
PlotManager.generateExperimentPlotsFromDataset(plotsToGenerate=plotsToGenerate,dataset=dataSetLocation,saveDirectory=saveMoodPlot)
