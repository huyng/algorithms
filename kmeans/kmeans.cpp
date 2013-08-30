#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#define DATA_FILE "data.csv"

using std::vector;
using std::getline;
using std::stringstream;

typedef vector<float> Row;
typedef vector<Row> Matrix;
typedef struct dimensions
{
    int rows,cols;
} dimensions;


int read_data(Matrix& dst)
{

    std::ifstream data(DATA_FILE);
    std::string line, cell;
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

dimensions get_dim()
{
    FILE* fp = fopen(DATA_FILE, "r");
    char c;
    dimensions dim;
    int rows = 0;
    int cols = 0;

    if (fp != NULL) {
        rows = fseek(fp, 0, SEEK_END);
    }
    fseek(fp, 0, SEEK_SET);
    std::cout << rows << std::endl;

    // count commas on first line
    do {
      c = fgetc(fp);
      if (c == ','){
        cols++;
      }
      printf("%c\n", c);
    } while (c != '\n');

    fclose(fp);
    // dim.rows = 6000;
    // dim.cols = 2;

    return dim;
}

int main(int argc, char const *argv[])
{   
    dimensions dim = get_dim();
    // // dimensions dim;

    Matrix X(dim.rows, dim.cols);
    read_data(X);
    for (int i = 0; i < X.size(); ++i)
    {
        for (int j = 0; j < X[0].size(); ++j)
        {
            std::cout << X[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}