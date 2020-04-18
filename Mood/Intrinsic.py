#Adapted from: Barros, P., & Wermter, S. (2017, May). A self-organizing model for affective memory. In 2017 International Joint Conference on Neural Networks (IJCNN) (pp. 31-38). IEEE.

from MoodyFramework.Mood.GWR import AssociativeGWR
import numpy
import copy

CONFIDENCE_PHENOMENOLOGICAL = "PHENOMENOLOGICAL"
CONFIDENCE_PMODEL = "PMODEL"


class Intrinsic():
    #Models
    moods =[] # stores the moods model
    pModel =None

    #Lists
    probabilities = []# stores the phenomenological probabilities per action
    moodReadings = [] #stores all the mood readings over all the actions for all the players
    moodNeurons = [] #sotre all the neurons-states over all the actions for all the players

    #Flags
    isUsingSelfMood = False
    selfConfidenceType = CONFIDENCE_PHENOMENOLOGICAL
    isUsingOponentMood = False
    actionNumber = []

    def __init__(self, selfConfidenceType=CONFIDENCE_PHENOMENOLOGICAL, isUsingSelfMood=False, isUsingOponentMood= False, loadPModel = ""):

        self.isUsingSelfMood = isUsingSelfMood
        self.selfConfidenceType = selfConfidenceType
        self.isUsingOponentMood = isUsingOponentMood

        #initialize lists
        self.moods = []
        self.probabilities = []
        self.moodReadings = []
        self.moodNeurons = []

        initialState = [0.5,0.5]#PA

        #Construct my own mood networks
        if self.isUsingSelfMood:
            self.moods.append(AssociativeGWR())
            self.moods[0].initNetwork(numpy.array([initialState, initialState, initialState, initialState, initialState]), 1)
            self.probabilities.append([])
            self.moodReadings.append([])
            self.moodNeurons.append([])
            self.actionNumber.append(0)

        #Construct my oponents mood network
        if self.isUsingOponentMood:
            for i in range (3):
                oponentNetwork = AssociativeGWR()
                oponentNetwork.initNetwork(numpy.array([initialState, initialState, initialState, initialState, initialState]), 1)
                self.moods.append(oponentNetwork)
                self.probabilities.append([])
                self.moodReadings.append([])
                self.moodNeurons.append([])
                self.actionNumber.append(0)


        if not loadPModel=="":
            self.pModel.loadModel(loadPModel)

    def phenomenologicalConfidence(self, qValue, actionIndex=0):

        actionNumber = self.actionNumber[actionIndex]
        # qValue = numpy.absolute(qValue)
        rewardModulator = actionNumber*0.01
        maxreward = 1 - 1*rewardModulator
        maxreward = 1
        theta = 0.0
        # maxreward = 0.1
        probability = (1 - theta) * (1 / 2 * numpy.log10(qValue / maxreward) + 1)

        probability =  numpy.tanh(probability)

        # print ("Q-value:" + str(qValue) + " - Reward:" + str(maxreward)  + " - Probability:" + str(probability))
        if probability <= 0:
            probability = 0
        if probability >= 1:
            probability = 1

        return probability

    def obtainPartialConfidences(self, action, partialHand, board, possibleActions,QModel, moodIndex):

        possibleConfidences = []

        # possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions), 0)

        states = []
        possibleVectors = []
        for handIndex in range(100):
            hand = numpy.array([])
            hand = numpy.append(hand, partialHand)
            for cardIndex in range(17-len(partialHand)):
                cardValue = numpy.random.random_integers(0,12)/13
                hand = numpy.append(hand, cardValue)

            if len(hand) > 17:
                 hand = hand[0:17]

            stateVector = numpy.concatenate((hand,board))
            states.append(stateVector)
            possibleVectors.append(possibleActions)
            # stateVector = numpy.expand_dims(numpy.array(stateVector), 0)
            # states.append((stateVector, possibleActionsVector))

        qValues = QModel.predict([states, possibleVectors])[:, action]

        for value in qValues:
            confidence = self.phenomenologicalConfidence(value, moodIndex)
            possibleConfidences.append(confidence)


        return possibleConfidences


    def observeOponentAction(self, params, QModel):

        if self.isUsingOponentMood:
            action, actionType, board, boardAfter,possibleActions, cardsInHand, thisPlayer, myIndex, done, score = params

            #organize mood networks so the first network is always refereing to P1
            if thisPlayer > myIndex:
                moodIndex = thisPlayer-1
            else:
                moodIndex = thisPlayer

            moodIndex = moodIndex+1 #my oponents

            action = numpy.argmax(action)

            possibleActions = copy.copy(possibleActions)

            board = board
            boardAfter = boardAfter

            partialHand = numpy.array(numpy.nonzero(boardAfter)).flatten()

            partialHand = numpy.copy(boardAfter[partialHand,])[0]
            cardsDiscarded = 17- cardsInHand

            partialHand = numpy.append(partialHand, numpy.zeros(cardsDiscarded))

            partialConfidences = self.obtainPartialConfidences(action, partialHand, board, possibleActions,QModel, moodIndex)


            self.actionNumber[moodIndex] = self.actionNumber[moodIndex] + 1

            self.probabilities[moodIndex].append(numpy.average(partialConfidences))

            avgConfidence = numpy.average(partialConfidences)

            newPartialConfidences = []
            for confidence in partialConfidences:
                # newPartialConfidences.append(confidence*0.1)
                newPartialConfidences.append(self.transformActionConfidence(confidence))

            partialConfidences = newPartialConfidences

            for confidence in partialConfidences:
                a,p = confidence
                self.updateMood(a,p, moodIndex=moodIndex)

            moodReading, neurons = self.readMood(moodIndex=moodIndex)

            if done:
                playerPosition = score.index(thisPlayer)

                if playerPosition == 0:
                    confidence = 1
                else:
                    confidence = 0

                self.actionNumber[moodIndex] = 0

                partialConfidences = []
                for a in range(10):
                    partialConfidences.append(self.transformEndGameConfidence(confidence))

                for confidence in partialConfidences:
                    a,p = confidence
                    self.updateMood(a,p, moodIndex=moodIndex)

                moodReading, neurons = self.readMood(moodIndex=moodIndex)

            return avgConfidence, moodReading, neurons
        else:
            return -1, [-1,-1],[-1,-1]

    def doSelfAction(self, qValue):

        nonZeroActions = numpy.nonzero(qValue)[0]

        if self.selfConfidenceType == CONFIDENCE_PHENOMENOLOGICAL:
            probabilities = []
            for a in nonZeroActions:
                probabilities.append(self.phenomenologicalConfidence(qValue[a], 0))
            probability = numpy.array(probabilities).sum()
            # probability = self.phenomenologicalConfidence(qValue[numpy.argmax(qValue)])
            if probability > 1:
                probability = 1

        # elif self.selfConfidenceType == CONFIDENCE_PMODEL:
        #     probability = self.getPModelProbability(params)

        self.probabilities[0].append(probability)

        self.actionNumber[0] =  self.actionNumber[0]+1

        if self.isUsingSelfMood:
            # confidence = probability*0.1
            a,p = self.transformActionConfidence(probability)
            self.updateMood(a,p, moodIndex=0)
            moodReading, neurons = self.readMood(moodIndex=0)

            return probability, moodReading, neurons



    def transformActionConfidence(self, confidence):


        #Tone down the pleasure of a good action
        p = confidence*0.7

        #Modulate the arousal of a good action
        if confidence > 0.5:
            a = 0.5 + ((confidence-0.5) / (1-0.5))/2
        else:
            a = 0.5

        # print("Probability:" + str(confidence))
        # print ("neuron: [" + str(a)+ str(",") + str(p) + str("]"))
        return a,p

    def transformEndGameConfidence(self, confidence):

        a = confidence*0.7
        p = confidence*1.5

        return a,p


    def updateMood(self, a, p, moodIndex, saveDirectory=""):
        numberOfEpochs = 1  # Number of training epochs
        insertionThreshold = 0.9    # Activation threshold for node insertion
        # insertionThreshold = 0.99  # Activation threshold for node insertion
        # insertionThreshold = 0.0001 # Activation threshold for node insertion
        # learningRateBMU = 0.3  # Learning rate of the best-matching unit (BMU)

        if p > 0:
            learningRateBMU = 0.9
            numberOfEpochs = 5  # Number of training epochs
            amountStimuli = 10
            learningRateNeighbors = 0.0001

        else:
            learningRateBMU = 0.01  # Learning rate of the best-matching unit (BMU)
            numberOfEpochs = 1  # Number of training epochs
            amountStimuli = 1
            learningRateNeighbors = 0.0001

        probs = []
        for i in range(amountStimuli):
            probs.append([a, p])
        # learningRateBMU = 0.35  # Learning rate of the best-matching unit (BMU)
        # learningRateNeighbors = 0.76  # Learning rate of the BMU's topological neighbors
        # learningRateNeighbors = 0.001  # Learning rate of the BMU's topological neighbors



        self.moods[moodIndex].trainAGWR(numpy.array(probs), numberOfEpochs, insertionThreshold, learningRateBMU, learningRateNeighbors)


    def doEndOfGame(self, score, thisPlayer):

        playerPosition = score.index(thisPlayer)

        if playerPosition == 0:
            confidence = 1
            # print ("player :" + str(thisPlayer) + " - WON!")
        else:
            confidence = 0

        # self.probabilities.append(confidence)

        self.actionNumber[0] = 0

        if self.isUsingSelfMood:
            for a in range(10):
                ar,p = self.transformEndGameConfidence(confidence)
                self.updateMood(ar,p, moodIndex=0)
            moodReading, neurons = self.readMood(moodIndex=0)

            return self.probabilities[0][-1], moodReading,neurons
        # return -1, [-1,-1], [-1,-1]


    def readMood(self, moodIndex):

        neuronAge = numpy.copy(self.moods[moodIndex].habn) # age of the neurons
        neuronWeights = numpy.copy(self.moods[moodIndex].weights) # confidence of the neurons
        # habituatedWeights = neuronAge* neuronWeights # weighted neurons

        habituatedWeights = []
        for a in range(len(neuronWeights)):
            habituatedWeights.append(neuronWeights[a] * neuronAge[a])

        habituatedWeights = numpy.array(habituatedWeights)
        # habituatedWeights = neuronWeights # weighted neurons
        # probmood = numpy.average(numpy.array(habituatedWeights).flatten())
        probmoodA = numpy.array(habituatedWeights[:, 0]).flatten()
        probmoodP = numpy.array(habituatedWeights[:, 1]).flatten()
        # probmoodA = numpy.tanh(numpy.array(habituatedWeights[:, 0]).flatten())
        # probmoodP = numpy.tanh(numpy.array(habituatedWeights[:, 1]).flatten())
        probmoodA = numpy.average(probmoodA)
        probmoodP = numpy.average(probmoodP)
        # probmood = numpy.exp(probmood) / numpy.sum(numpy.exp(probmood), axis=0)
        # if probmood > 1:
        #     probmood = 1


        self.moodNeurons[moodIndex].append((neuronWeights.tolist(), neuronAge.tolist()))
        self.moodReadings[moodIndex].append([probmoodA,probmoodP])

        return [probmoodA,probmoodP], (neuronWeights.tolist(), neuronAge.tolist())



    #PModel functions
    def trainPModel(self, params):

        if self.selfConfidenceType == CONFIDENCE_PMODEL:
            self.pModel.train(params)

    def getPModelProbability(self, params):

        return self.pModel.getProbability(params)[0]