from Testing import average, stDeviation, testPenData, testCarData


OUTPUT_FILENAME = "q7_output_raw2.txt"


def trainPenDataWithRestart(n, hiddenLayerPerceptrons=None):
    accuracyList = []
    with open(OUTPUT_FILENAME, 'a+') as fh:
        fh.write("-------------------- Training with " + str(hiddenLayerPerceptrons) + " Perceptron(s) --------------------\n")
        fh.write("Start training pen data for " + str(n) + " round(s) ...\n")
    for run in range(n):
        if hiddenLayerPerceptrons is not None:
            _, accuracy = testPenData(hiddenLayers=[hiddenLayerPerceptrons])
        else:
            _, accuracy = testPenData()
        accuracyList.append(accuracy)
        with open(OUTPUT_FILENAME, 'a+') as fh:
            fh.write("Round " + str(run + 1) + ": " + str(accuracy) + "\n")
    with open(OUTPUT_FILENAME, 'a+') as fh:
        fh.write("All rounds finished.\n")
        fh.write("---------- Accuracy Statistics ----------\n")
        fh.write("Max               : " + str(max(accuracyList)) + "\n")
        fh.write("Average           : " + str(average(accuracyList)) + "\n")
        fh.write("Standard Deviation: " + str(stDeviation(accuracyList)) + "\n\n")


def trainCarDataWithRestart(n, hiddenLayerPerceptrons=None):
    accuracyList = []
    with open(OUTPUT_FILENAME, 'a+') as fh:
        fh.write("-------------------- Training with " + str(hiddenLayerPerceptrons) + " Perceptron(s) --------------------\n")
        fh.write("Start training car data for " + str(n) + " round(s) ...\n")
    for run in range(n):
        if hiddenLayerPerceptrons is not None:
            _, accuracy = testCarData(hiddenLayers=[hiddenLayerPerceptrons])
        else:
            _, accuracy = testCarData()
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
    trainPenDataWithRestart(n=5)
    trainCarDataWithRestart(n=5)
