#ifndef MANIFOLD_H
#define MANIFOLD_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;

bool manifold_edge_check(py::array_t<int> faces, py::array_t<double> verticies);

py::array_t<int> clean_faces(py::array_t<int> faces, py::array_t<double> verticies);

double faceArea(py::array_t<double> v1, py::array_t<double> v2, py::array_t<double> v3);

#endif