from q5 import trainCarDataWithRestart, trainPenDataWithRestart


def trainPenDataWithRestartAndVaryingHiddenLayer(n, numPerceptronList):
    for numPerceptrons in numPerceptronList:
        trainPenDataWithRestart(n, hiddenLayerPerceptrons=numPerceptrons)


def trainCarDataWithRestartAndVaryingHiddenLayer(n, numPerceptronList):
    for numPerceptrons in numPerceptronList:
        trainCarDataWithRestart(n, hiddenLayerPerceptrons=numPerceptrons)


if __name__ == "__main__":
    trainPenDataWithRestartAndVaryingHiddenLayer(5, range(0, 41, 5))
    trainCarDataWithRestartAndVaryingHiddenLayer(5, range(0, 41, 5))
