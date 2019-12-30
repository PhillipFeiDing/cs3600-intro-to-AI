import re


def rawToCSV(inputFileName, outputFileName):
    outputList = []
    outputList.append(["max", "average", "standard deviation"])
    with open(inputFileName, "r") as fh:
        lines = fh.read()
    maxMatch = re.findall("(Max[ ]*:[ ]*)(\d+\.\d*)", lines)
    averageMatch = re.findall("(Average[ ]*:[ ]*)(\d+\.\d*)", lines)
    stdDevMatch = re.findall("(Standard Deviation[ ]*:[ ]*)(\d+\.\d*)", lines)
    if len(maxMatch) == len(averageMatch) == len(stdDevMatch):
        for maxAcc, averageAcc, stdDevAcc in zip(maxMatch, averageMatch, stdDevMatch):
            outputList.append([maxAcc[1], averageAcc[1], stdDevAcc[1]])
    lines = ""
    for row in range(len(outputList)):
        for col in range(len(outputList[row])):
            lines = lines + outputList[row][col]
            if col != len(outputList[row]) - 1:
                lines = lines + ", "
        lines = lines + "\n"
    with open(outputFileName, "w+") as fh:
        fh.write(lines)


if __name__ == "__main__":
    # rawToCSV("q6_output_raw.txt", "q6_output_csv.csv")
    # rawToCSV("q7_output_raw.txt", "q7_output_csv.csv")
    rawToCSV("q7_output_raw2.txt", "q7_output_csv2.csv")
    # rawToCSV("q8_output_raw.txt", "q8_output_csv.csv")
