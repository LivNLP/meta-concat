#include <iostream>
#include <fstream>

using namespace std;

int main(){
    int x = 0;
    int y = 0;
    double r = 0;
    cout << "Hello" << endl;
    ifstream F("./data/cooccurrences.bin", std::ios::binary);
    if (F.fail()){
        cout << "Failed" << endl;
    }
    while (F.good()){
        F.read((char *) &x, sizeof(int));
        F.read((char *) &y, sizeof(int));
        F.read((char *) &r, sizeof(double));
        cout << "x = " << x << " y = " << y << " r = " <<  r << endl;
    }
    F.close();
}