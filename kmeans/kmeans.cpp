#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <limits>
#include <math.h>       /* fabs */




#define DATA_FILE "data.csv"
#define OUT_FILE "clusters.csv"

using std::vector;
using std::getline;
using std::stringstream;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::string;
using std::endl;

typedef vector<float> Row;
typedef vector<Row> Matrix;
typedef vector<int> Indices;
typedef struct MatDim
{
    int rows;
    int cols;
} MatDim;


int readData(Matrix& dst)
{
    ifstream data(DATA_FILE);
    string line, cell;
    int row_idx = 0;
    int col_idx = 0;
    
    while(getline(data,line))
        {
        stringstream lineStream(line);
        while(getline(lineStream,cell,','))
            {
            dst[row_idx][col_idx] = atof(cell.c_str());
            col_idx++;
            }
        row_idx++;
        col_idx = 0;
        }
    return 0;
    
}
void printMat(Matrix& data)
{
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            cout << std::setw(6) << data[i][j] << " ";
        }
        cout << endl;
    }
}

MatDim getDataDimensions()
{
    MatDim dim;
    string line;
    int rows = 1;
    ifstream data(DATA_FILE);
    getline(data, line);
    cout << line << endl;
    dim.cols = (int)std::count(line.begin(), line.end(), ',' ) + 1;
    while(getline(data, line)) rows++;
    dim.rows = rows;
    return dim;
}

float computeDistance(const Row& v1, const Row& v2) {
    float dist = 0.0;
    for (int i = 0; i < v1.size(); ++i)
        dist += fabs(v1[i] - v2[i]);
    dist /= float(v1.size());
    return dist;
}

int computeCentroid(Matrix& X, Row& dstCentroid) {
    for (int i = 0; i < X.size(); ++i)
        for (int j = 0; j < dstCentroid.size(); ++j)
            dstCentroid[j] += X[i][j];
    for (int j = 0; j < dstCentroid.size(); ++j)
        dstCentroid[j] /= X.size();
    return 0;
}

float computeError(Matrix& X, Row& center) {
    float error = 0.0;
    for (int i = 0; i < X.size(); ++i)
        error += computeDistance(center, X[i]);
    return error;
}


int kmeansCluster(Matrix& X, Indices& cluster_idx, int kSize, float stopThreshold)
{
    float prevError = std::numeric_limits<float>::max();
    float currError = 0.0;
    float errorDelta = std::numeric_limits<float>::max();
    size_t rows = X.size();
    size_t featSize = X[0].size();
    
    do {
        // initialize centers
        currError = 0.0;
        
        Matrix centers(kSize, Row(featSize));
        for (int i = 0; i < kSize; ++i) {
            int idx = rand() % rows;
            centers[i] = X[idx];
        }
  
        // assign each example to a cluster
        for (int i = 0; i < rows; ++i)
            {
            float minDist = std::numeric_limits<float>::max();
            cluster_idx[i] = 0;
            for (int c = 0; c < kSize; ++c)
            {
                float d = computeDistance(centers[c], X[i]);
                if (d < minDist) {
                    minDist = d;
                    cluster_idx[i] = c;
                }
            }
        }
        
        // compute error & update centroids
        for (int c = 0; c < kSize; ++c)
        {
            Matrix members;
            Row centeroid(featSize);
            for (int i = 0; i < rows; ++i)
            {
                if(cluster_idx[i] == c) members.push_back(X[i]);
            }
            computeCentroid(members, centeroid);
            currError += computeError(members, centeroid);
            centers[c] = centeroid;
        }
        errorDelta = fabs(prevError - currError);
        prevError = currError;
        cout << "errorDelta = " << errorDelta << endl;
    } while(errorDelta > stopThreshold);
    
    return 0;
}


int main(int argc, char const *argv[])
{
    MatDim dim = getDataDimensions();
    Matrix X(dim.rows, Row(dim.cols));
    Indices cluster_idx(dim.rows);
    readData(X);
    kmeansCluster(X, cluster_idx, 3, 0.00001);
    ofstream outf(OUT_FILE);
    for (int i = 0; i < cluster_idx.size(); ++i) {
        outf << cluster_idx[i] << ",";
        for (int j = 0; j < X[i].size(); ++j) {
            outf << X[i][j];
            if (j != (X[i].size() -1)) {
                outf << ",";
            }
        }
        outf << endl;
        
    }
    outf.close();
    
    
    return 0;
}