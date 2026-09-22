#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <random>
#include <cmath>
#include <omp.h>



bool is_matrix(std::vector<std::vector<double>>& data);
std::vector<std::vector<double>>& read_matrix_from_file(std::string& filename);


class not_a_matrix_error : public std::runtime_error {
public:
    not_a_matrix_error() : std::runtime_error("Not a matrix") {}
    not_a_matrix_error(const std::string& message) : std::runtime_error(message) {}
};


class Matrix {
    std::vector<std::vector<double>> _data;

public:
    Matrix() { auto data = std::vector<std::vector<double>>(); }
    Matrix(std::vector<std::vector<double>>& data) {
        if (!is_matrix(data)) throw not_a_matrix_error();
        _data = data;
    }
    Matrix(std::string& filename) { _data = read_matrix_from_file(filename); }


    std::pair<size_t, size_t> size() const {
        return std::pair<size_t, size_t>(_data[0].size(), _data.size());
    }


    const std::vector<std::vector<double>>& get_data() const {
        return _data;
    }


    Matrix operator*(const Matrix& other) const {
        if (_data[0].size() != other._data.size()) throw std::invalid_argument("Matrices cannot be multiplied");
        if (_data.empty() || other._data.empty()) return Matrix();

        const size_t m = _data.size();
        const size_t n = _data[0].size();
        const size_t p = other._data[0].size();

        std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += _data[i][k] * other._data[k][j];
                }
                result[i][j] = sum;
            }
        }

        return Matrix(result);
    }


    Matrix operator*(const double scalar) const {
        if (_data.empty()) return Matrix();

        const size_t m = _data.size();
        const size_t n = _data[0].size();

        std::vector<std::vector<double>> result(m, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                result[i][j] = _data[i][j] * scalar;
            }
        }

        return Matrix(result);
    }


    bool operator==(const Matrix& other) const {
        if (this->size() != other.size()) return false;
        auto size = this->size();
        for (size_t i = 0; i < size.first; ++i) {
            for (size_t j = 0; j < size.second; ++j) {
                if (std::abs(_data[i][j] - other._data[i][j]) > 1e-9) return false;
            }
        }
        return true;
    }


    void write_to_file(const std::string& filename) const {
        std::ofstream of(filename);
        if (!of) throw std::runtime_error("Error while writing to file: " + filename);

        auto size = this->size();
        for (size_t i = 0; i < size.first; ++i) {
            for (size_t j = 0; j < size.second; ++j) {
                of << _data[i][j] << " ";
            }
            of << "\n";
        }
        of.close();
    }


    Eigen::MatrixXd toMatrixXd() const {
        size_t rows = _data.size();
        size_t cols = _data[0].size();
        Eigen::MatrixXd mat(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat(i, j) = _data[i][j];
            }
        }
        return mat;
    }
};


inline Matrix operator*(const double scalar, const Matrix& matrix) {
    return matrix * scalar;
}


std::vector<std::vector<double>>& read_matrix_from_file(std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("File not found: " + filename);

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<double> numbers;
        double x;
        while (iss >> x) numbers.push_back(x);
        matrix.push_back(numbers);
    }

    in.close();

    if (!is_matrix(matrix)) throw not_a_matrix_error();
    return matrix;
}


bool is_matrix(std::vector<std::vector<double>>& data) {
    if (data.empty()) return true;
    size_t len = data[0].size();
    for (auto row : data) if (row.size() != len) return false;
    return true;
}


bool is_square(std::vector<std::vector<double>>& data) {
    if (data.empty()) return true;
    if (!is_matrix(data)) throw not_a_matrix_error();
    return data.size() == data[0].size();
}


double debug_multiplication(const Matrix& lhs, const Matrix& rhs, std::string output_filename = "none") {
    auto before = std::chrono::high_resolution_clock::now();
    Matrix result = lhs * rhs;
    auto after = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(after - before).count();

    if (lhs.size().first <= 1000) {
        Eigen::MatrixXd lhs_e = lhs.toMatrixXd(), rhs_e = rhs.toMatrixXd();
        Eigen::MatrixXd result_e = lhs_e * rhs_e;
        if (!result.toMatrixXd().isApprox(result_e, 1e-9)) throw std::exception(">:(");
    }

    if (output_filename != "none") {
        try {
            result.write_to_file(output_filename);
        }
        catch (std::runtime_error e) {
            std::cerr << e.what();
        }
    }
    return elapsed;
}


Matrix generate_random_matrix(size_t size, double min = -5.0, double max = 5.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    std::vector<std::vector<double>> data(size, std::vector<double>(size));
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            data[i][j] = dis(gen);
        }
    }
    return Matrix(data);
}


std::vector<double> benchmark(std::vector<int> sizes, int iterations = 1) {
    std::vector<double> results;
    for (auto size : sizes) {
        double total_time = 0;
        for (int i = 0; i < iterations; ++i) {
            Matrix lhs = generate_random_matrix(size);
            Matrix rhs = generate_random_matrix(size);
            total_time += debug_multiplication(lhs, rhs);
        }
        results.push_back(total_time / iterations);
    }
    return results;
}


int main() {
    std::vector<int> sizes = {200, 400, 800, 1200, 2000};
    std::vector<double> results = benchmark(sizes);
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << "Size: " << sizes[i] << ", Operations: " << sizes[i] * sizes[i] * sizes[i] << ", Time: " << results[i] << " seconds\n";
    }

}