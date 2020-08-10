#pragma once
#include "kNN.h"
#include <omp.h>
#include <string>
#include <iostream>
#include <random>
#include <time.h>
#include <memory>

using namespace std;

// This function randomly order the input data
void randomize(myDataFormat &data);

// This function prepapres needed GPU objects & buffers for input data
void prepareGPUExecution(myDataFormat &data, const char *platform); 

// Prepare training data set on the device
void prepareTrainingData(const myDataFormat& data, myDataFormat& trainingData, size_t n, size_t i);

// Prepare validation data set on the device
void prepareValidationData(const myDataFormat& data, myDataFormat& validationdata, size_t n, size_t i);

// Tranfer kNN results on the device back to host and evalute kNN reuslts using CPU
void evaluateError(myDataFormat& validationData, const std::string &decisionString, size_t n, size_t i);
