import numpy
import pandas as pd
from KEF.DataSetManager import actionPass, actionFinish, actionDiscard, actionPizzaReady
from MoodyFramework.FaceEditing import FaceEditor
import cv2

from keras.models import load_model



from PIL import Image

def moodReadingToCategorical(reading):
    modelDirectory = "/home/pablo/Documents/Datasets/ChefsHat_ReinforcementLearning/TrainedModels/arousalValence_to_categorical/arousalToCategorical.h5"
    model = load_model(modelDirectory)
    reading = numpy.array(reading)

    reading = numpy.expand_dims(reading, 0)
    reading = numpy.expand_dims(reading, 0)
    # print("Shape:" + str(reading.shape))
    category = model.predict([reading])[0][0]

    # print("Output:" + str(category))
    # print("Output:" + str(numpy.argmax(category)))

    return numpy.argmax(category)



def readIntrinsicMeasures(model):

    probs = []
    moodReadings = []
    moodNeurons = []
    for a in range(4):
        probs.append([])
        moodReadings.append([])
        moodNeurons.append([])

    for p in range(len(model.probabilities)):
            if len(model.probabilities[p]) > 0:
                probs[p].append(model.probabilities[p][-1])
                moodReadings[p].append(model.moodReadings[p][[-1,-1]])
                moodNeurons[p].append(model.moodNeurons[p][[-1,-1]])

    return probs,moodReadings,moodNeurons


def generateMoodFromDataset(intrinsicModels, dataset, saveDirectory = "", qModels = [], agents=[], loadModels = []):

    #Load trained agents
    if len(qModels)==0:
        qModels = []
        for indexAgent, agent in enumerate(agents):
            if len(intrinsicModels) > indexAgent and not intrinsicModels[indexAgent] == "":
                agent.startAgent((11, 17, 200, loadModels[indexAgent], ""))
                qModels.append(agent.actor)

    columns = ["Time", "Agent Names", "Game Number", "Round Number", "Player", "Action Type",
               "P1 probability", "P2 probability", "P3 probability", "P4 probability",
               "P1 Mood Reading","P1 Mood Category" , "P2 Mood Reading", "P2 Mood Category", "P3 Mood Reading","P3 Mood Category", "P4 Mood Reading","P4 Mood Category",
               "P1 Mood Neuron", "P2 Mood Neuron", "P3 Mood Neuron", "P4 Mood Neuron", "P1 AVStimuli", "P2 AVStimuli", "P3 AVStimuli", "P4 AVStimuli",
               "P1 Expectation", "P2 Expectation", "P3 Expectation", "P4 Expectation"]

    moodDataFrame = pd.DataFrame(columns=columns)

    readFile = pd.read_pickle(dataset)

    agentsNames = []
    previousBoard = [13,0,0,0,0,0,0,0,0,0,0]
    pizzaReady = False
    currentRound = 0

    playerFinished = []
    currentGame = 0

    totalPoints = [0, 0 , 0 , 0]

    for lineCounter, row in readFile.iterrows():

        # if lineCounter == 115:
        #     break

        print ("Parsing line:" + str(lineCounter))

        game = row["Game Number"]

        if len(agentsNames) == 0:
            agentsNames = row["Agent Names"]

        player = row["Player"]
        actionType = row["Action Type"]

        if actionType == actionFinish:
            finishingPosition = len(playerFinished)
            points = 3 - finishingPosition
            totalPoints[player] += points
            playerFinished.append(player)

        if game > currentGame:
            currentGame = game
            playerFinished = []
            currentRound = 0

        if not player =="":
            time = row["Time"]
            possibleActions = row["Possible Actions"]
            roundNumber = row["Round Number"]
            playerHand = row["Player Hand"]
            board = row["Board"]
            qValues = row["Qvalues"]
            cardsAction = row["Cards Action"]

            scores = row["Scores"].tolist()

            probabilities = []
            moods = []
            moodNeurons = []

            avStimuli = []

            selfExpectation = []
            for a in range(4):
                probabilities.append(-1)
                moods.append([-1,-1])
                moodNeurons.append([-1,-1])
                avStimuli.append([])
                selfExpectation.append(-1)


            for playerIndex, intrinsicModel in enumerate(intrinsicModels):
                a, v = -1, -1
                if not intrinsicModel == "":
                    """ Calculate Event1: Action """
                    # #Calculate the impact of the action on each of the agents' mood
                    a,v = intrinsicModels[playerIndex].event1_Action(qValues, playerHand[playerIndex],player==playerIndex )
                    avStimuli[playerIndex].append([a,v])

                    """ Calculate Event2: Pizza """
                    # # If there is a pizza, update everyone accordingly.
                    # # if I finished the game, I do not care about the pizza.
                    a, v = -1, -1
                    if pizzaReady and not(playerIndex in playerFinished):
                        a,v = intrinsicModels[playerIndex].event2_Pizza(player==playerIndex)
                    avStimuli[playerIndex].append([a, v])

                    # # if playerIndex==0:
                    # #     print ("Adding pizza action:" + str(a)+","+str(v))
                    #
                    #
                    """ Calculate Event3: Finish Game """
                    # # If I finish the game, I update my mood based on my position
                    a, v = -1, -1
                    if actionType == actionFinish and player == playerIndex:
                        score = scores.index(playerIndex)
                        a,v = intrinsicModels[playerIndex].event3_Finish(score)
                    avStimuli[playerIndex].append([a, v])
                    #
                    """ Calculate Event4: Long Term Game Score """
                    # # Starting from the second game, every time an action is mage
                    # # the agent will take into consideration its global position
                    a, v = -1, -1
                    if currentGame > 0 and player == playerIndex:
                        myPoints = totalPoints[player]
                        orderedPoints = numpy.sort(totalPoints).tolist()
                        myPosition = 3-orderedPoints.index(myPoints)
                        a, v = intrinsicModels[playerIndex].event4_LongTermScore(myPoints, myPosition, player==playerIndex)
                    avStimuli[playerIndex].append([a, v])

                    selfMood, selfNeurons = intrinsicModels[playerIndex].readMood()
                    moods[playerIndex] = selfMood
                    moodNeurons[playerIndex] = selfNeurons

                    selfExpectation[playerIndex] = intrinsicModels[playerIndex].selfExpectation
                else:
                    for a in range(4):
                        avStimuli[playerIndex].append([-1, -1])

            "P1 Expectation", "P2 Expectation", "P3 Expectation", "P4 Expectation"

            if pizzaReady:
                """ Update the dataframe with the pizza information! """
                oldValues = []
                pizzaPValues = []
                for pIndex in range(4):
                    pizzaPValues.append([])
                    thisAVValues = avStimuli[pIndex]
                    # print ("Shape:" + str(thisAVValues.shape))
                    pizzaPValues[pIndex].append([-1,-1])
                    pizzaPValues[pIndex].append(thisAVValues[1])
                    pizzaPValues[pIndex].append(thisAVValues[2])
                    pizzaPValues[pIndex].append(thisAVValues[3])
                    #
                    # oldValues.append(avStimuli[pIndex][0])
                    # avStimuli[pIndex][0] = [-1,-1]

                dataframe = [time, agentsNames, game, currentRound, player, actionPizzaReady,
                             probabilities[0], probabilities[1], probabilities[2], probabilities[3],
                             moods[0], "", moods[1], "", moods[2], "", moods[3],
                             "",
                             moodNeurons[0], moodNeurons[1], moodNeurons[2], moodNeurons[3], pizzaPValues[0],
                             pizzaPValues[1],pizzaPValues[2], pizzaPValues[3],
                             selfExpectation[0], selfExpectation[1], selfExpectation[2],
                             selfExpectation[3]]


                moodDataFrame.loc[-1] = dataframe
                moodDataFrame.index = moodDataFrame.index + 1
                pizzaReady = False

                for pIndex in range(4):
                    avStimuli[pIndex][1] = [-1,-1]

            dataframe = [time, agentsNames, game, currentRound, player, actionType,
                         probabilities[0], probabilities[1], probabilities[2], probabilities[3],
                         moods[0], "", moods[1], "", moods[2], "", moods[3],
                         "",
                         moodNeurons[0], moodNeurons[1], moodNeurons[2], moodNeurons[3], avStimuli[0],
                             avStimuli[1],avStimuli[2], avStimuli[3],
                         selfExpectation[0], selfExpectation[1],selfExpectation[2],
                         selfExpectation[3]]

            moodDataFrame.loc[-1] = dataframe
            moodDataFrame.index = moodDataFrame.index + 1

        elif actionType == actionPizzaReady:
            pizzaReady = True
            currentRound = currentRound+1




    moodDataFrame.to_pickle(saveDirectory + "/" + "IntrinsicDataset.pkl")
    moodDataFrame.to_csv(saveDirectory + "/" + "IntrinsicDataset.csv", index=False, header=True)

