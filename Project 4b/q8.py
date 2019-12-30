from NeuralNetUtil import buildExamplesFromExtraData
from Testing import average, stDeviation
from NeuralNet import buildNeuralNet
from q5 import OUTPUT_FILENAME


def trainExtraDataWithRestartAndVaryingHiddenLayer(n, numPerceptronList):
    for numPerceptrons in numPerceptronList:
        accuracyList = []
        with open(OUTPUT_FILENAME, 'a+') as fh:
            fh.write("-------------------- Training with " + str(
                numPerceptrons) + " Perceptron(s) --------------------\n")
            fh.write("Start training pen data for " + str(n) + " round(s) ...\n")
        for run in range(n):
            _, accuracy = buildNeuralNet(buildExamplesFromExtraData(), maxItr=200, hiddenLayerList=[numPerceptrons])
            accuracyList.append(accuracy)
            with open(OUTPUT_FILENAME, 'a+') as fh:
                fh.write("Round " + str(run + 1) + ": " + str(accuracy) + "\n")
        with open(OUTPUT_FILENAME, 'a+') as fh:
            fh.write("All rounds finished.\n")
            fh.write("---------- Accuracy Statistics ----------\n")
            fh.write("Max               : " + str(max(accuracyList)) + "\n")
            fh.write("Average           : " + str(average(accuracyList)) + "\n")
            fh.write("Standard Deviation: " + str(stDeviation(accuracyList)) + "\n\n")


if __name__ == "__main__":
    trainExtraDataWithRestartAndVaryingHiddenLayer(5, range(0, 41, 5))
