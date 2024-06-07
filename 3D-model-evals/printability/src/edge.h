#ifndef EDGE_H
#define EDGE_H

#include <functional>
#include <string>
#include <tuple>

class Edge {
    public:
        Edge();
        Edge(std::tuple<double, double, double> v1, std::tuple<double, double, double> v2);
        Edge(std::tuple<double, double, double> v1, std::tuple<double, double, double> v2, bool directed);
        std::tuple<double, double, double> getV1() const;
        std::tuple<double, double, double> getV2() const;
        bool getDirected() const;
        double getDistance() const;
        Edge& operator=(const Edge& other);
        bool operator==(const Edge& other) const;
    private:
        std::tuple<double, double, double> v1, v2;
        bool directed;
};

namespace std {
    template<>
    struct hash<Edge> {
        std::size_t operator()(const Edge& obj) const;
    };
}

#endif