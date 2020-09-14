import numpy
import pandas as pd
from KEF.DataSetManager import actionPass, actionFinish, actionDiscard
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


def generateMoodFromDataset(intrinsicModels, dataset, saveDirectory = "", agentFaces=[], qModels = [], agents=[], loadModels = []):
    from MoodyFramework.FaceEditing.Editor import editFace

    faceSequence = []
    for a in range(4):
        faceSequence.append([])

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
               "P1 Mood Neuron", "P2 Mood Neuron", "P3 Mood Neuron", "P4 Mood Neuron"]

    moodDataFrame = pd.DataFrame(columns=columns)

    readFile = pd.read_pickle(dataset)

    agentsNames = []
    previousBoard = [13,0,0,0,0,0,0,0,0,0,0]
    for lineCounter, row in readFile.iterrows():

        # if lineCounter == 115:
        #     break

        print ("Parsing line:" + str(lineCounter))

        game = row["Game Number"]

        if len(agentsNames) == 0:
            agentsNames = row["Agent Names"]

        player = row["Player"]

        if not player =="":
            time = row["Time"]
            possibleActions = row["Possible Actions"]
            roundNumber = row["Round Number"]
            playerHand = row["Player Hand"]
            board = row["Board"]
            qValues = row["Qvalues"]
            cardsAction = row["Cards Action"]
            actionType = row["Action Type"]
            scores = row["Scores"].tolist()

            probabilities = []
            moods = []
            moodNeurons = []
            faces = []
            category = []
            for a in range(4):
                probabilities.append(-1)
                moods.append([-1,-1])
                moodNeurons.append([-1,-1])
                faces.append(0)
                category.append(-1)

            #Is there an intrinsic model?
            if len(intrinsicModels) > player and not intrinsicModels[player] == "":
                #there is an intrinsic model for this player

                #calculate the self-probability for this player
                #Update the intrinsic model for this player when performing this action
                selfProb, selfMood, selfNeurons = intrinsicModels[player].doSelfAction(qValues)
                probabilities[player] = selfProb
                moods[player] = selfMood
                moodNeurons[player] = selfNeurons
                category[player] = moodReadingToCategorical(selfMood)

                if len(agentFaces) > 0 and not agentFaces[player] == "":
                    # faces[player] = editFace(agentFaces[player], moods[player][0],moods[player][0])

                    faces[player] = editFace(agentFaces[0], moods[player][0],moods[player][1])
                    img = Image.fromarray(faces[player].astype('uint8'))
                    img = img.convert('RGB')
                    img.save(saveDirectory + "/AvatarFace/img" + str(lineCounter) +"_"+ str(moods[player][0])+"_"+str(moods[player][1])+".png")

                    # cv2.imwrite(saveDirectory + "/AvatarFace/img" + str(lineCounter) +"_"+ str(moods[player][0])+"_"+str(moods[player][1])+".png", faces[player])

                if actionType == actionFinish:
                    selfProb, selfMood, selfNeurons = intrinsicModels[player].doEndOfGame(scores,player)
                    probabilities[player] = selfProb
                    moods[player] = selfMood
                    category[player] = moodReadingToCategorical(selfMood)
                    moodNeurons[player] = selfNeurons

            #Update how the others' observe this action

            if actionType == actionPass:
                boardAfter = board
            else:
                boardAfter = numpy.copy(cardsAction).tolist()
                while not len(boardAfter) == 11:
                    boardAfter.append(0)

            if actionType == actionFinish:
                done = True
            else:
                done = False

            #Calculate how many cards the player who did this action has at hand
            cardsInHandN = numpy.nonzero(playerHand[player])[0]
            cardsInHand = len(cardsInHandN)

            #For each of the others, update their observation values

            #Obtain which others will observe this action
            others = numpy.array(range(len(intrinsicModels))).tolist()
            if player in others:
                others.remove(player)

            for otherIndex in others:

                if not intrinsicModels[otherIndex] == "":

                    params = qValues, actionType, numpy.array(previousBoard), numpy.array(
                        boardAfter), possibleActions, cardsInHand, player, otherIndex, done, scores
                    otherProb, otherMood, otherNeuron = intrinsicModels[otherIndex].observeOponentAction(params, qModels[otherIndex])
                    probabilities[otherIndex] = otherProb
                    moods[otherIndex] = otherMood
                    category[player] = moodReadingToCategorical(otherMood)
                    moodNeurons[otherIndex] = otherNeuron

            #Add the probabilities to the dataset

            # print ("Category:" + str(category))
            dataframe = [time, agentsNames, game, roundNumber, player, actionType,
                         probabilities[0], probabilities[1], probabilities[2], probabilities[3],
                         moods[0], category[0],  moods[1], category[1], moods[2], category[2], moods[3], category[3],
                         moodNeurons[0], moodNeurons[1], moodNeurons[2], moodNeurons[3]]
            moodDataFrame.loc[-1] = dataframe
            moodDataFrame.index = moodDataFrame.index + 1

            faceSequence[player].append(faces)
            previousBoard = board


    #
    # for a in range(4):
    #     probabilities = intrinsicModels[a].probabilities
    moodDataFrame.to_pickle(saveDirectory+"/"+"IntrinsicDataset.pkl")
    moodDataFrame.to_csv(saveDirectory+"/"+"IntrinsicDataset.csv", index=False, header=True)

