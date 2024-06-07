#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>

#include "manifold.h"
#include "edge.h"

namespace py = pybind11;

PYBIND11_MODULE(manifold, m) {
    m.def("manifold_edge_check", &manifold_edge_check, "A function that checks for non manifold edges given mesh faces");
    m.def("face_area", &faceArea, "Finds the area of an STL face");

    py::class_<Edge>(m, "Edge")
        .def(py::init<>())
        .def(py::init<std::tuple<double, double, double>, std::tuple<double, double, double>>()) 
        .def(py::init<std::tuple<double, double, double>, std::tuple<double, double, double>, bool>()) 
        .def("getV1", &Edge::getV1)
        .def("getV2", &Edge::getV2)
        .def("getDirected", &Edge::getDirected)
        .def("getDistance", &Edge::getDistance)
        .def("__eq__", &Edge::operator==);
}
