#include<bits/stdc++.h>

using namespace std;

int main(){
    int n;
    cin >> n;
    int i = 1;
    while(i <= n)
    {
        cout << i << " ";
        if (i % 10 == 0)
        cout <<endl;
        i++; 
    }
    
}