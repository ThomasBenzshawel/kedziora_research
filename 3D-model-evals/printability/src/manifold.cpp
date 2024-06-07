#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <tuple>
#include <functional>
#include <iostream>
#include <ostream>
#include <vector>

#include "manifold.h"
#include "edge.h"

std::tuple<double, double, double> pyarray_to_tuple(py::array_t<double> input_array) {
    if (input_array.ndim() != 1 || input_array.shape(0) != 3) {
        throw std::runtime_error("Input array must be 1D with size 3");
    }

    auto data = input_array.unchecked(); 

    return std::make_tuple(data(0), data(1), data(2));
}

py::array_t<double> vector_to_double_numpy(const std::vector<double>& faces) {
    size_t size = faces.size();

    py::array_t<double> result = py::array_t<double>(size);

    py::buffer_info buf = result.request(); 
    double* ptr = (double*)buf.ptr; 

    std::copy(faces.begin(), faces.end(), ptr); 

    return result;
}

py::array_t<int> vector_to_int_numpy(const std::vector<py::array_t<int>>& faces) {
    int num_rows = faces.size();
    int num_cols = 3;

    py::array_t<int> result({num_rows, num_cols});

    auto r = result.mutable_unchecked();
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            r(i, j) = faces[i].at(j); 
        }
    }

    return result;
}

py::array_t<double> get_inner_double_array(py::array_t<double> input, const int index) {
    if (input.ndim() != 2 || input.shape(1) != 3) {
        throw std::runtime_error("Input arrays must be 2D with inner arrays of size 3");
    }

    py::array_t<double> result = py::array_t<double>(3);

    py::buffer_info input_buf = input.request();
    py::buffer_info result_buf = result.request();

    double *input_ptr = (double *) input_buf.ptr,
            *result_ptr = (double *) result_buf.ptr;
    
    int X = input_buf.shape[1];

    for (int idx = 0; idx < X; idx++) {
        result_ptr[idx] = input_ptr[index + idx];
    }

    return result;
}

py::array_t<int> get_inner_int_array(py::array_t<int> input, const int index) {
    if (input.ndim() != 2 || input.shape(1) != 3) {
        throw std::runtime_error("Input arrays must be 2D with inner arrays of size 3");
    }

    py::array_t<int> result = py::array_t<int>(3);

    py::buffer_info input_buf = input.request();
    py::buffer_info result_buf = result.request();

    int *input_ptr = (int *) input_buf.ptr,
            *result_ptr = (int *) result_buf.ptr;
    
    int X = input_buf.shape[1];

    for (int idx = 0; idx < X; ++idx) {
        result_ptr[idx] = input_ptr[index + idx];
    }

    return result;
}

std::vector<double> subtractVertices(py::array_t<double> v1, py::array_t<double> v2) {
    std::vector<double> vec;

    vec.push_back(v1.at(0)-v2.at(0));
    vec.push_back(v1.at(1)-v2.at(1));
    vec.push_back(v1.at(2)-v2.at(2));

    return vec;
}

std::vector<double> crossProduct(std::vector<double> vec1, std::vector<double> vec2) {
    std::vector<double> vec;

    vec.push_back(vec1.at(1) * vec2.at(2) - vec1.at(2) * vec2.at(1));
    vec.push_back(vec1.at(2) * vec2.at(0) - vec1.at(0) * vec2.at(2));
    vec.push_back(vec1.at(0) * vec2.at(1) - vec1.at(1) * vec2.at(0));

    return vec;
}

double vecMagnitude(std::vector<double> vec) {
    return sqrt(vec.at(0) * vec.at(0) + vec.at(1) * vec.at(1) + vec.at(2) * vec.at(2));
}

double faceArea(py::array_t<double> v1, py::array_t<double> v2, py::array_t<double> v3) {
    std::vector<double> vec1 = subtractVertices(v2, v1);
    std::vector<double> vec2 = subtractVertices(v3, v1);

    std::vector<double> cross_product = crossProduct(vec1, vec2);

    return 0.5 * vecMagnitude(cross_product);
}

void edge_areas(std::vector<int> const vec, py::array_t<int> faces, py::array_t<double> verticies) {
    int area_count = 0;
    py::array_t<double> v1, v2, v3;
    double area;

    for (int i : vec) {
        v1 = get_inner_double_array(verticies, faces.at(i, 0));
        v2 = get_inner_double_array(verticies, faces.at(i, 1));
        v3 = get_inner_double_array(verticies, faces.at(i, 2));
        area = faceArea(v1, v2, v3);

        area_count++;
        std::cout << "Area" << area_count << ": " << area << std::endl;
    }
}

bool manifold_edge_check(py::array_t<int> faces, py::array_t<double> verticies) {
    if (faces.ndim() != 2 || verticies.ndim() != 2 || faces.shape(1) != 3 || verticies.shape(1) != 3) {
        throw std::runtime_error("Input arrays must be 2D with inner arrays of size 3");
    }

    std::tuple<double, double, double> v1, v2, v3;
    py::array_t<double> vertex1, vertex2, vertex3;
    Edge edge1, edge2, edge3;
    
    std::unordered_map<Edge, int> edge_counts;
    std::unordered_map<Edge, std::vector<int>> face_finder;

    for (int i = 0; i < faces.shape()[0]; ++i) {
        vertex1 = get_inner_double_array(verticies, faces.at(i, 0));
        vertex2 = get_inner_double_array(verticies, faces.at(i, 1));
        vertex3 = get_inner_double_array(verticies, faces.at(i, 2));

        v1 = pyarray_to_tuple(vertex1);
        v2 = pyarray_to_tuple(vertex2);
        v3 = pyarray_to_tuple(vertex3);

        auto [x1, y1, z1] = v1;
        auto [x2, y2, z2] = v2;
        auto [x3, y3, z3] = v3;

        double area = faceArea(vertex1, vertex2, vertex3);

        // std::cout << std::endl;
        // std::cout << "V1: " << x1 << ", " << y1 << ", " << z1 << std::endl;
        // std::cout << "V2: " << x2 << ", " << y2 << ", " << z2 << std::endl;
        // std::cout << "V3: " << x3 << ", " << y3 << ", " << z3 << std::endl;
        // std::cout << "Area: " << area << std::endl << std::endl;

        edge1 = Edge(v1, v2);
        edge2 = Edge(v1, v3);
        edge3 = Edge(v2, v3);
        
        edge_counts[edge1]++;
        edge_counts[edge2]++;
        edge_counts[edge3]++;

        face_finder[edge1].push_back(i);
        face_finder[edge2].push_back(i);
        face_finder[edge3].push_back(i);
        
    }

    bool manifold = true;
    for(auto p : edge_counts) {
        if (p.second != 2) {
            auto [x1, y1, z1] = p.first.getV1();
            auto [x2, y2, z2] = p.first.getV2();

            // std::cout << std::endl;
            // std::cout << "Edge V1: " << x1 << ", " << y1 << ", " << z1 << std::endl;
            // std::cout << "Edge V2: " << x2 << ", " << y2 << ", " << z2 << std::endl;
            // std::cout << "Edge Count: " << p.second << std::endl;
            // edge_areas(face_finder[p.first], faces, verticies);
            // std::cout << std::endl;
            
            manifold = false;
        }
    }
    //std::cout << "end" << std::endl;

    return manifold;
}