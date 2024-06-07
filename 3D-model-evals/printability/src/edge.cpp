#include <string>
#include <functional>
#include <tuple>
#include <vector>
#include <cmath>
#include <iterator>
#include <ostream>
#include <iostream>

#include "edge.h"

Edge::Edge() {
    this->v1 = std::make_tuple(0.0, 0.0, 0.0);
    this->v2 = std::make_tuple(0.0, 0.0, 0.0);
    this->directed = false;
}

Edge::Edge(std::tuple<double, double, double> v1, std::tuple<double, double, double> v2) {
    this->v1 = v1;
    this->v2 = v2;
    this->directed = false;
}

Edge::Edge(std::tuple<double, double, double> v1, std::tuple<double, double, double> v2, bool directed) {
    this->v1 = v1;
    this->v2 = v2;
    this->directed = directed;
}

std::tuple<double, double, double> Edge::getV1() const{
    return v1;
}

std::tuple<double, double, double> Edge::getV2() const{
    return v2;
}

bool Edge::getDirected() const{
    return directed;
}

double Edge::getDistance() const {
    auto [x1, y1, z1] = v1;
    auto [x2, y2, z2] = v2;

    auto xdist = x2 - x1;
    auto ydist = y2 - y1;
    auto zdist = z2 - z1;

    auto xdistsq = xdist * xdist;
    auto ydistsq = ydist * ydist;
    auto zdistsq = zdist * zdist;

    return sqrt(xdistsq + ydistsq + zdistsq);
}

Edge& Edge::operator=(const Edge& other) {
    if (this != &other) {
        this->v1 = other.getV1();
        this->v2 = other.getV2();
        this->directed = other.getDirected();
    }

    return *this;
}

bool Edge::operator==(const Edge& other) const {
    if (directed) {
        return (v1 == other.getV1() && v2 == other.getV2());
    } else {
        return ((v1 == other.getV1() && v2 == other.getV2()) || 
                (v1 == other.getV2() && v2 == other.getV1()));
    }

    return false;
}

std::vector<double> tupleToVector(const std::tuple<double, double, double>& input_tuple) {
    std::vector<double> result;

    result.push_back(std::get<0>(input_tuple));
    result.push_back(std::get<1>(input_tuple));
    result.push_back(std::get<2>(input_tuple));

    return result;
} 

double calculateDistance(const std::vector<double>& vertex) {
   double sum_of_squares = 0.0;
   for (double coordinate : vertex) {
       sum_of_squares += coordinate * coordinate;
   }
   return sqrt(sum_of_squares);
} 

std::vector<double> concatenateVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result;
    result.reserve(v1.size() + v2.size()); 

    result.insert(result.end(), v1.begin(), v1.end());

    result.insert(result.end(), v2.begin(), v2.end());

    return result;
}

std::size_t hashVector(const std::vector<double>& vec) {
    std::size_t seed = vec.size();
    for (auto& element : vec) {
        std::hash<int> hasher;
        seed ^= hasher(element) + 0x9e3779b9 + (seed << 6) + (seed >> 2); 
    }

    return seed;
}

bool order_equal_vert(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.at(0) == vec2.at(0)) {
        return vec1.at(1) < vec2.at(1);
    } else {
        return vec1.at(0) < vec2.at(0);
    }
}

size_t std::hash<Edge>::operator()(const Edge& obj) const {
    std::vector<double> vertex1 = tupleToVector(obj.getV1());
    std::vector<double> vertex2 = tupleToVector(obj.getV2());

    double v1Dist = calculateDistance(vertex1);
    double v2Dist = calculateDistance(vertex2);

    std::vector<double> combinedArray;

    if (v1Dist < v2Dist) {
        combinedArray = concatenateVectors(vertex1, vertex2);
    } else if (v1Dist > v2Dist) {
        combinedArray = concatenateVectors(vertex2, vertex1);
    } else {
        if (order_equal_vert(vertex1, vertex2)) {
            combinedArray = concatenateVectors(vertex1, vertex2);
        } else {
            combinedArray = concatenateVectors(vertex2, vertex1);
        }
    }

    return hashVector(combinedArray);
}

