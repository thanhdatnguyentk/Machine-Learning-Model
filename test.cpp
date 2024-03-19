#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

struct computer
{
    string brand;
    double speed;
    double price;
};

bool sortByName(const computer& a, const computer& b) {
    if (a.brand != b.brand) 
        return a.brand < b.brand;
    return a.speed < b.speed;
}

bool sortBySpeed(const computer& a, const computer& b) {
    if (a.speed != b.speed)
        return a.speed > b.speed;
    return a.brand < b.brand;
}

bool sortByPrice(const computer& a, const computer& b) {
    if (a.price != b.price)
        return a.price < b.price;
    return a.speed > b.speed;
}

void sortComputers(vector<computer>& computers, bool (*compare)(const computer&, const computer&)) {
    sort(computers.begin(), computers.end(), compare);
    for (const auto& comp : computers) {
        cout << comp.brand << " " << comp.speed << " " << comp.price << endl;
    }
    cout << endl;
}

void filterComputers(const vector<computer>& computers, double minPrice, double maxPrice, double minSpeed, double maxSpeed) {
    for (const auto& comp : computers) {
        if (comp.price >= minPrice && comp.price <= maxPrice && comp.speed >= minSpeed && comp.speed <= maxSpeed) 
            cout << comp.brand << " " << comp.speed << " " << comp.price << endl;
    }
    cout << endl;
}

int main () {
    int n;
    cin >> n;
    vector<computer> computers(n);
    for (int i = 0; i < n; i++) {
        cin >> computers[i].brand >> computers[i].speed >> computers[i].price;
    }
    sortComputers(computers, sortByName);
    sortComputers(computers, sortBySpeed);
    sortComputers(computers, sortByPrice);
    filterComputers(computers, 100, 1000, 2, 5);
    return 0;
}
