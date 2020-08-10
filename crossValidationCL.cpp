#include <iostream>
using namespace std;

#include "inc/kNN.h"
#include "inc/evaluation.h"
#include "inc/distance.h"
#include "inc/YoUtil.hpp"

void showInstruction(const char *exe)
{
    cout << "\n"
         << exe << " [CL platform] [instances filename] [n] [k] [# feature attributes] [# decision variables] [similarity] [decision string] [p]" << endl;
    cout << "\n[CL platform]: string such as 'Advanced' or 'Intel' to specify the platform to use.";
    cout << "\n[ninstances filename]: specify a file that contains instances of data with known attributes & decision variables.";
    cout << "\n\t\t Note that the instance file is in CSV (comma separated values) format.";
    cout << "\n[n]: specify the n parameter for n-fold cross validation.";
    cout << "\n[k]: a positive integer that defines the k in kNN algorithm.";
    cout << "\n[# feature attributes]: a positive integer indicating the number of feature attributes.";
    cout << "\n\t\t It should be noted feature attributes must be in the first few columns of the specified CSV files.";
    cout << "\n[# decision variables]: a positive integer indicating the number of decision variables.";
    cout << "\n[similarity]: one of the following integer: ";
    cout << "\n\t\t0: use Euclidean distance measure.";
    cout << "\n\t\t1: use Manhattan distance measure.";
    cout << "\n\t\t2: use Chebyshev distance measure.";
    cout << "\n\t\t3: use Minkowski distance measure.";
    cout << "\n\t\t4: use Cosine Similarity.";
    cout << "\n[decision string]: each character in the string is either 'C' or 'R' and corresponds to a decision variable type.";
    cout << "\n\t\tC: classification, the corresponding decision variable is treated as text labels.";
    cout << "\n\t\tR: regression, the corresponding decision variable is treated as real numbers.";
    cout << "\n[p]: the p parameter used for Minkowski distance measure.";
    cout << endl;
}

int main(int argc, char **argv)
{
    if (argc < 9)
    {
        showInstruction(argv[0]);
        return 255;
    }

    // step 0: get parameters from command line arguments
    const size_t n = atoi(argv[3]);
    const size_t k = atoi(argv[4]);
    const size_t noAttributes = atoi(argv[5]);
    const size_t noDecisionVariables = atoi(argv[6]);
    const size_t distanceMeasureToUse = atoi(argv[7]);
    const string decisionString = argv[8];

    // step 1: do some validations on obtained parameters
    if (decisionString.size() != noDecisionVariables)
    {
        cerr << "\nThe length of decision string(" << decisionString << ") should be equal to " << noDecisionVariables << "(# decision variables)." << endl;
        return 254;
    }
    if (distanceMeasureToUse < 0 || distanceMeasureToUse > 4)
    {
        cerr << "\nThe specified distance measure " << distanceMeasureToUse << " is invalid.";
        return 253;
    }

    myDataFormat data;

    // step 2: read instance data set
    readCSV(data, argv[2], noAttributes, decisionString);
    double t[6] = {0};

    // step 3: normalize the data set on the HOST (e.g. No GPU necessary unless you have lots of time)
    YoUtil::stopWatch watch;
    watch.start();
    normalize(data);
    watch.stop();
    t[0] = watch.elapsedTime();

    // step 4: randomize the order on the HOST
    watch.start();
    randomize(data);
    watch.stop();
    t[1] = watch.elapsedTime();

    // step 5: prepare GPU data by copying all randomly ordered and normalized feature & decision data onto the device
    prepareGPUExecution(data, argv[1]);

    // step 6: repeat n-times (n = # folders)
    for (size_t i = 0; i < n; ++i)
    {
        myDataFormat trainingData, validationData;
        // step 6a: prepare i-th training data set on GPU
        watch.start();
        prepareTrainingData(data, trainingData, n, i);

        // step 6b: prepare i-th validation data set on GPU
        prepareValidationData(data, validationData, n, i);
        watch.stop();
        t[2] += watch.elapsedTime();

        // step 6c: calculate the distance matrix beween the testing and the training data set on GPU
        watch.start();
        calculateDistanceMatrix(trainingData, validationData, distanceMeasureToUse);
        watch.stop();
        t[3] += watch.elapsedTime();

        // step 6d: now make decisions for each testing data on GPU
        watch.start();
        doKNN(k, trainingData, validationData, decisionString);
        watch.stop();
        t[4] += watch.elapsedTime();

        // step 6e: tranfer kNN results from GPU onto the host & evaluate results on the host
        watch.start();
        evaluateError(validationData, decisionString, n, i);
        watch.stop();

        t[5] += watch.elapsedTime();
        freeAllocation(validationData);
        freeAllocation(trainingData);
    }
    cout << "\nT: ";
    t[0] *= n;
    t[1] *= n;
    for (int i = 0; i < 6; ++i)
    {
        cout << t[i] / n << ", ";
    }
    cout << endl;
    // step 6: finally, do some cleanup... Do not forget to clean up GPU resources
    freeAllocation(data);

    return 0;
}
