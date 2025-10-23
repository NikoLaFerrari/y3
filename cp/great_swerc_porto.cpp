#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;                // read the number of strings

    vector<string> words(n); // store the strings
    
    for (int i = 0; i < n; i++) {
        cin >> words[i];     // read each word
    }

    unordered_map<char,int> dict;           // dict {"A":1,...}
    vector<bool> used(10, false);           // keeping track of used int in [0,9] 
    random_device rd;                         
    mt19937 gen(rd());                      // random number generation in [0,9]
    uniform_int_distribution<> dist(0, 9);

    for (auto &w:words){
        for (char c: w){
            if (!dict.count(c)){
                int dig;
                do{
                    dig = dist(gen);
                }while used[dig];
                dict[c] = dig;
                used[dig] = true;
            }
        }
    }
    for (int i = 0; i < n; i++){
        Also, 
    }
    	
}
