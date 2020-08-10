#include "../inc/evaluation.h"
#include "../inc/YoUtil.hpp"

void randomize(myDataFormat &data)
{
    // std::cout << "\nREMEBER TO IMPLEMENT randomize " << __LINE__ << "@" << __FILE__ << endl;
    srand(1);
    double temp;
    string Stemp;
    size_t index_i = 0;
    // cout << data.row << endl;
    for (size_t i = 0; i < data.row; i++)
    {
        index_i = rand() % data.row;
        for (size_t j = 0; j < (data.column + data.remainColumn); j++)
        {
            // cout<<"["<<i+1<<" , "<<j+1<<"]";
            if (j < data.column)
            {

                temp = data.vector[i * data.column + j];
                data.vector[i * data.column + j] = data.vector[index_i * data.column + j];
                data.vector[index_i * data.column + j] = temp;
            }
            else
            {
                Stemp = data.remain[i * data.remainColumn + (j - data.column)];
                data.remain[i * data.remainColumn + (j - data.column)] = data.remain[index_i * data.remainColumn + (j - data.column)];
                data.remain[index_i * data.remainColumn + (j - data.column)] = Stemp;
            }
        }
    }

    // for (size_t i = 0; i < 1000; i++)
    // {
    //     for (size_t j = 0; j < (data.column + data.remainColumn); j++)
    //     {
    //         if (j < data.column)
    //         {
    //             cout << data.vector[i * data.column + j] << "\t";
    //         }
    //         else
    //         {
    //             cout << data.remain[i * data.remainColumn + (j - data.column)] << "\t";
    //         }
    //     }
    //     cout << endl;
    // }
}

void prepareTrainingData(const myDataFormat &data, myDataFormat &trainingData, size_t n, size_t i)
{
    // std::cout << "\nREMEBER TO IMPLEMENT prepareTrainingData to prepare GPU data for kNN" << __LINE__ << "@" << __FILE__ << endl;
    trainingData.row = data.row - data.row / n;
    trainingData.column = data.column;
    trainingData.remainColumn = data.remainColumn;
    trainingData.vector = new double[trainingData.row * trainingData.column];
    trainingData.remain = new string[trainingData.row * trainingData.remainColumn];

    // //CL part
    // //配置記憶體d_vec 給向量資料
    trainingData.d_vec = new cl::Buffer(data.context, CL_MEM_READ_WRITE, trainingData.column * trainingData.row * sizeof(trainingData.vector));
    trainingData.queue = data.queue;
    trainingData.queue.enqueueWriteBuffer(*trainingData.d_vec, CL_FALSE, 0, trainingData.column * trainingData.row * sizeof(trainingData.vector), trainingData.vector);
    trainingData.device = data.device;
    trainingData.context = data.queue.getInfo<CL_QUEUE_CONTEXT>();
    trainingData.program = data.gpu->getProgram("kernel");
    cl::Kernel krn(*data.program, "getTrainningDataVector");
    //for vector part
    int in_len = data.row * data.column;
    int out_len = trainingData.row * trainingData.column;
    krn.setArg(0, *data.d_vec);
    krn.setArg(1, in_len);
    krn.setArg(2, *trainingData.d_vec);
    krn.setArg(3, out_len);
    krn.setArg(4, (int)i);
    krn.setArg(5, (int)n);
    size_t localSize = 5;
    cl::NDRange global((data.row * data.column + localSize - 1) / localSize * localSize);
    cl::NDRange local(localSize);
    data.queue.enqueueNDRangeKernel(krn, cl::NullRange, global, local);
    data.queue.enqueueReadBuffer(*trainingData.d_vec, CL_TRUE, 0, trainingData.column * trainingData.row * sizeof(trainingData.vector), trainingData.vector);

    if (i != 0)
    {
        for (size_t it = 0; it < i * (data.row / n); it++)
        {
            // cout << "it = " << it << endl;
            for (size_t jt = 0; jt < trainingData.remainColumn; jt++)
            {
                trainingData.remain[it * trainingData.remainColumn + jt] = data.remain[it * data.remainColumn + jt];
                // cout << trainingData.remain[it * trainingData.remainColumn + (jt - trainingData.column)] << "\t";
            }
            // cout << endl;
        }
    }

    for (size_t it = (data.row / n) * i; it < trainingData.row; it++)
    {
        // cout << " it = " << it << endl;
        for (size_t jt = 0; jt < trainingData.remainColumn; jt++)
        {
            trainingData.remain[it * trainingData.remainColumn + jt] = data.remain[(it + (data.row / n)) * data.remainColumn + jt];
            // cout << trainingData.remain[it * trainingData.remainColumn + (jt - trainingData.column)] << "\t";
        }
        // cout << endl;
    }
    // cout << "\nResult: \n";
    // for (size_t i = 0; i < trainingData.row; i++)
    // {
    //     cout << "[" << i << "]";
    //     for (size_t j = 0; j < (trainingData.column + trainingData.remainColumn); j++)
    //     {
    //         if (j < trainingData.column)
    //         {
    //             cout << trainingData.vector[i * trainingData.column + j] << ", ";
    //         }
    //         else
    //         {
    //             cout << trainingData.remain[i * trainingData.remainColumn + (j - trainingData.column)] << ", ";
    //         }
    //     }
    //     cout << endl;
    // }
}

void prepareValidationData(const myDataFormat &data, myDataFormat &vData, size_t n, size_t i)
{
    // std::cout << "\nREMEBER TO IMPLEMENT prepareValidationData to prepare GPU data for kNN " << __LINE__ << "@" << __FILE__;
    vData.row = data.row / n;
    // cout << vData.row << endl;
    vData.column = data.column;
    // cout << vData.column << endl;
    vData.remainColumn = data.remainColumn;
    // cout << vData.remainColumn << endl;
    vData.vector = new double[vData.row * vData.column];
    vData.remain = new string[vData.row * vData.remainColumn];

    //CL part
    vData.d_vec = new cl::Buffer(data.context, CL_MEM_READ_WRITE, vData.column * vData.row * sizeof(vData.vector));
    vData.queue = data.queue;
    vData.queue.enqueueWriteBuffer(*vData.d_vec, CL_FALSE, 0, vData.column * vData.row * sizeof(vData.vector), vData.vector);
    vData.device = data.device;
    vData.context = data.queue.getInfo<CL_QUEUE_CONTEXT>();
    vData.program = data.gpu->getProgram("kernel");
    cl::Kernel krn(*data.program, "getValidationDataVector");
    //for vector part
    int in_len = data.row * data.column;
    int out_len = vData.row * vData.column;
    krn.setArg(0, *data.d_vec);
    krn.setArg(1, in_len);
    krn.setArg(2, *vData.d_vec);
    krn.setArg(3, out_len);
    krn.setArg(4, (int)i);
    krn.setArg(5, (int)n);
    size_t localSize = 256;
    cl::NDRange local(localSize);
    cl::NDRange global((data.row * data.column + localSize - 1) / localSize * localSize);
    data.queue.enqueueNDRangeKernel(krn, cl::NullRange, global, local);
    data.queue.enqueueReadBuffer(*vData.d_vec, CL_TRUE, 0, vData.column * vData.row * sizeof(vData.vector), vData.vector);

    for (size_t iv = 0; iv < vData.row; iv++)
    {
        for (size_t jv = 0; jv < vData.remainColumn; jv++)
        {
            vData.remain[iv * vData.remainColumn + jv] = data.remain[(iv + i * vData.row) * data.remainColumn + jv];
        }
    }
    //     cout << "\nResult: \n";
    // for (size_t i = 0; i < vData.row; i++)
    // {
    //     cout << "[" << i << "]";
    //     for (size_t j = 0; j < (vData.column + vData.remainColumn); j++)
    //     {
    //         if (j < vData.column)
    //         {
    //             cout << vData.vector[i * vData.column + j] << ", ";
    //         }
    //         else
    //         {
    //             cout << vData.remain[i * vData.remainColumn + (j - vData.column)] << ", ";
    //         }
    //     }
    //     cout << endl;
    // }
}
struct record
{
    double wrongClassification = 0;
    double sum = 0;
    double max_Regression_Error = 0;
};
record *result;
int dataNum = 0;

void evaluateError(myDataFormat &data, const std::string &decisionString, size_t n, size_t i)
{
    // // std::cout << "\nREMEBER TO IMPLEMENT evaluateError " << __LINE__ << "@" << __FILE__ << endl;

    double wrong_Regression_Error = 0;
    // for (size_t i = 0; i < decisionString.size() * data.row; i++)
    // {
    //     cout << data.result[i] << "|" << data.remain[i] << "\t";
    //     if (i % decisionString.size() == 1)
    //     {
    //         cout << endl;
    //     }
    // }
    //cout << "i = " << i << endl;
    if (i == 0)
        result = new record[data.resultNum];
    dataNum += data.row;
    for (size_t ie = 0; ie < data.remainColumn; ie++)
    {
        if (decisionString[ie] == 'C')
        {
            for (size_t j = 0; j < data.row; j++)
            {
                if (data.result[j * data.resultNum + ie] != data.remain[j * data.remainColumn + ie])
                {
                    result[ie].wrongClassification++;
                }
                if (j == data.row - 1 && i == (n - 1))
                {
                    cout << "\tClassification error rate:\t" << 100 * result[ie].wrongClassification / dataNum << "%" << endl;
                }
            }
        }
        else
        {
            for (size_t j = 0; j < data.row; j++)
            {
                if (data.result[j * data.resultNum + ie] != data.remain[j * data.remainColumn + ie])
                {
                    wrong_Regression_Error = abs((atof(data.result[j * data.resultNum + ie].c_str()) -
                                                  atof(data.remain[j * data.remainColumn + ie].c_str()))) /
                                             atof(data.remain[j * data.remainColumn + ie].c_str());

                    result[ie].sum += wrong_Regression_Error;

                    if (wrong_Regression_Error > result[ie].max_Regression_Error)
                    {
                        result[ie].max_Regression_Error = wrong_Regression_Error;
                        // cout << data.result[j * data.resultNum + ie] << "|" << data.remain[j * data.remainColumn + ie] << endl;
                        // cout << wrong_Regression_Error << endl;
                    }
                    else if (j == data.row - 1 && i == n - 1)
                    {

                        cout << "\tRegression average error rate:\t" << result[ie].sum * 100 / dataNum << "%" << endl;
                        //cout << "\tRegression maximum error rate:\t" << result[ie].max_Regression_Error * 100 << "%" << endl;
                    }
                }
            }
        }
    }
    // delete[] data.distance;
    // delete[] data.result;
    if (i == n - 1)
    {
        delete[] result;
    }
}
void prepareGPUExecution(myDataFormat &data, const char *platform)
{
    // std::cout << "\nREMEBER TO IMPLEMENT prepareGPUExecution " << __LINE__ << "@" << __FILE__;
    // prepare for GPU execution & copy data set into GPU memory (cl::Buffer)
    // Also prepare program object (load kernels.cl) which compiles the kernels.cl
    //平台層
    if (!strcmp(platform, "Intel"))
        data.gpu = new YoUtil::GPU(CL_DEVICE_TYPE_CPU, false);
    else if (!strcmp(platform, "Advanced"))
        data.gpu = new YoUtil::GPU(CL_DEVICE_TYPE_GPU, false);
    else
        cout << "so sad, cannot find the platform" << endl;
    //把CL放上GPU的程式中
    data.gpu->addProgramByFile("kernel", "kernels.cl");
    //取得佇列
    data.queue = data.gpu->getCommandQueue(0);
    //定義裝置
    data.device = data.queue.getInfo<CL_QUEUE_DEVICE>();
    //輸出裝置資訊
    cout << "\n*** Running on: " << data.device.getInfo<CL_DEVICE_NAME>() << flush;
    //定義程式
    data.program = data.gpu->getProgram("kernel");
    //定義上下文
    data.context = data.queue.getInfo<CL_QUEUE_CONTEXT>();
    //配置記憶體d_vec
    data.d_vec = new cl::Buffer(data.context, CL_MEM_READ_WRITE, data.column * data.row * sizeof(data.vector));
    //寫入data
    data.queue.enqueueWriteBuffer(*data.d_vec, CL_FALSE, 0, data.column * data.row * sizeof(data.vector), data.vector);
    //設置參數
    // int len = data.row * data.column;
    // krn.setArg(0, len);
    // krn.setArg(1, data.d_vec);
    // size_t localSize = 4;
    // cl::NDRange local(localSize);
    // cl::NDRange global(data.row * data.column);
    //啟動程式
    // data.queue.enqueueNDRangeKernel(krn, cl::NullRange, global, local);
    // data.queue.enqueueReadBuffer(data.d_vec, CL_TRUE, 0, data.column * data.row * sizeof(data.vector), data.vector);
    // for (size_t i = 0; i < data.row; i++)
    // {
    //     for (size_t j = 0; j < data.column; j++)
    //     {
    //         cout << data.vector[i * data.column + j] << ", ";
    //     }
    //     cout << endl;
    // }
}
