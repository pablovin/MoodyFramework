#Adapted from: Barros, P., & Wermter, S. (2017, May). A self-organizing model for affective memory. In 2017 International Joint Conference on Neural Networks (IJCNN) (pp. 31-38). IEEE.

from MoodyFramework.ComplexMood import GWR
import numpy
import copy


class Intrinsic():

    #Models
    mood = None # stores the moods model

    #Expectations
    selfExpectation = 0.7

    opponentsExpectation = []


    def __init__(self, isUsingSelfMood=False):
        #Construct the mood network
        self.mood = GWR.GWR()
        self.mood.initNetwork()


    def phenomenologicalConfidence(self, qValue):

        maxreward = 1
        theta = 0.0
        probability = (1 - theta) * (1 / 2 * numpy.log10(qValue / maxreward) + 1)

        probability =  numpy.tanh(probability)

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


    def event4_LongTermScore(self, myPoints, position, selfAction):
        arousal, valence = -1, -1
        if selfAction:
            arousal = (myPoints * 100 / 15) * 0.01  # How far am I of winning the game
            if position == 0:
                valence = 1.0
            elif position == 1:
                valence = 0.65
            elif position == 2:
                valence = 0.45
            elif position == 3:
                valence = 0

            """Update on the self expectation"""
            # Update the self-expectation based on my finishing position
            self.selfExpectation += self.selfExpectation * arousal * 0.1

            if self.selfExpectation > 1:
                self.selfExpectation = 1
            elif self.selfExpectation < 0:
                self.selfExpectation = 0

            # print ("Long term arousal:" + str(arousal))
            # print("Long term valence:" + str(valence))
            """ Params for training specific for event4_LongTermScore """
            # It happens on every action, so it has a small impact.

            epsilon_bmu = 0.001 * self.selfExpectation
            tau_bmu = 0.001 * self.selfExpectation

            epsilon_n = 0.001 * self.selfExpectation
            tau_n = 0.001 * self.selfExpectation

            epoches = 5

            params = [epsilon_bmu, tau_bmu, epsilon_n, tau_n, epoches]

            self.updateMood(arousal, valence, params)
        return arousal, valence

    def event3_Finish(self,position):

        if position == 0:
            arousal = 1.0
            valence = 1.0
        elif position == 1:
            arousal = 0.5
            valence = 0.5
        elif position == 2:
            arousal = 0.5
            valence = -0
        elif position == 3:
            arousal = 1.0
            valence = -0

        """Update on the self expectation"""
        # Update the self-expectation based on my finishing position
        self.selfExpectation += arousal*0.5

        if self.selfExpectation > 1:
            self.selfExpectation = 1
        elif self.selfExpectation < 0:
            self.selfExpectation = 0



        """ Params for training specific for event3_Finish """
        # This only happens once per game, so it has a high impact
        # we will have a low impact on the update of the BMU
        # and of the neighbours

        epsilon_bmu = 0.5 * self.selfExpectation
        tau_bmu = 0.5 * self.selfExpectation

        epsilon_n = 0.5 * self.selfExpectation
        tau_n = 0.5 * self.selfExpectation

        epoches = 10

        params = [epsilon_bmu, tau_bmu, epsilon_n, tau_n, epoches]

        self.updateMood(arousal, valence, params)

        return arousal, valence

    def event2_Pizza(self, selfAction):

        if selfAction:
            arousal = 1.0
            valence = 1.0
            expectation = self.selfExpectation
        else:
            arousal = 1.0
            valence = 0.0
            expectation = self.selfExpectation

        """Update on the self expectation"""
        # Update the self-expectation based on if I did a pizza or not.
        #
        if selfAction:
            self.selfExpectation += (self.selfExpectation * 0.1)
        else:
            self.selfExpectation += -(self.selfExpectation * 0.1)

        if self.selfExpectation > 1:
            self.selfExpectation = 1
        elif self.selfExpectation < 0:
            self.selfExpectation = 0

        """ Params for training specific for event2_self """
        # As this is a relatively common event
        # we will have a low impact on the update of the BMU
        # and of the neighbours

        epsilon_bmu = 0.03 * self.selfExpectation
        tau_bmu = 0.03 * self.selfExpectation

        epsilon_n = 0.03 * self.selfExpectation
        tau_n = 0.03 * self.selfExpectation

        epoches = 5

        params = [epsilon_bmu, tau_bmu, epsilon_n, tau_n, epoches]

        # print ("Updated on valance and arousal:" + str(arousal) + "," + str(valence))
        self.updateMood(arousal, valence, params)

        return arousal, valence
        # mood, moodNeurons = self.readMood()
        #
        # return mood, moodNeurons


    def event1_Action(self, qvalue, numberCardsAtHand, selfAction):

        arousal = -1
        valence = -1
        if selfAction:
            nonZeroActions = numpy.nonzero(qvalue)[0]
            probabilities = []

            for a in nonZeroActions:
                probabilities.append(self.phenomenologicalConfidence(qvalue[a]))

            probability = numpy.array(probabilities).sum()
            if probability > 1:
                probability = 1

            arousal = probability
            nonZeroCards = len(numpy.nonzero(numberCardsAtHand)[0])
            valence =  1 - (nonZeroCards*100/17)*0.01 # Porcentage of amount of cards at hand.

            """Update on the self expectation"""
            # Update the self-expectation based on my own confidence
            # as confidence happens every game, this has a low impact on the self expectation
            in_min = 0
            out_min = -1
            in_max = 1
            out_max = -1

            scaledConfidence = numpy.tanh((probability - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
            expUpdate = self.selfExpectation * scaledConfidence * 0.05
            self.selfExpectation += expUpdate
            if self.selfExpectation > 1:
                self.selfExpectation = 1
            elif self.selfExpectation < 0:
                self.selfExpectation = 0

            # print("Confidence:" + str(probability))
            # print ("scaledConfidence:" + str(scaledConfidence))
            # print("expUpdate:" + str(expUpdate))
            # input("here")


            """ Params for training specific for event1 """
            # As this is a very common event, with a low impact
            # we will have a low impact on the update of the BMU
            # and of the neighbours

            epsilon_bmu = 0.01 * self.selfExpectation
            tau_bmu = 0.01 * self.selfExpectation

            epsilon_n = 0.01 * self.selfExpectation
            tau_n = 0.01 * self.selfExpectation

            epoches = 5

            params = [epsilon_bmu, tau_bmu, epsilon_n, tau_n, epoches]

            self.updateMood(arousal,valence, params)

        return arousal, valence



    def updateMood(self, a, v, params):

        amountStimuli = 5
        probs = []
        for i in range(amountStimuli):
            probs.append([a, v])
        self.mood.train(numpy.array(probs), params)

    def readMood(self):

        neuronAge = numpy.copy(self.mood.habn)

        neuronWeights = numpy.copy(self.mood.weights)

        habituatedWeights = numpy.array(neuronWeights)

        probmoodA = numpy.array(habituatedWeights[:, 0]).flatten()
        probmoodP = numpy.array(habituatedWeights[:, 1]).flatten()

        probmoodA = numpy.average(probmoodA)
        probmoodP = numpy.average(probmoodP)

        if probmoodA > 1:
            probmoodA = 1
        elif probmoodA < 0:
            probmoodA = 0

        if probmoodP > 1:
            probmoodP = 1
        elif probmoodP < 0:
            probmoodP = 0

        return [probmoodA,probmoodP], (neuronWeights.tolist(), neuronAge.tolist())
