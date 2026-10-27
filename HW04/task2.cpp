// Serial N-body simulation, completed from the ECE 759 HW04 skeleton.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

const double G = 1.0;
const double softening = 0.1;
const double dt = 0.01;
const double board_size = 4.0;

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
    for (int i = 0; i < N; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            const double dx = pos[j][0] - pos[i][0];
            const double dy = pos[j][1] - pos[i][1];
            const double dz = pos[j][2] - pos[i][2];
            const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            ax += G * (dx * inv_r3) * mass[j];
            ay += G * (dy * inv_r3) * mass[j];
            az += G * (dz * inv_r3) * mass[j];
        }
        acc[i][0] = ax;
        acc[i][1] = ay;
        acc[i][2] = az;
    }
}

// All positions must be updated before evaluating the new accelerations.
void advanceStep(double pos[][3], double vel[][3], double acc[][3],
                 const double mass[], int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] += acc[i][k] * dt / 2.0;
            pos[i][k] += vel[i][k] * dt;
            pos[i][k] = std::clamp(pos[i][k], -board_size, board_size);
        }
    }
    getAcc(pos, mass, acc, N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] += acc[i][k] * dt / 2.0;
        }
    }
}

// Optional visualization helper; not called during timing runs.
void savePositionsToCSV(const double pos[][3], int N, int step,
                        const std::string &filename) {
    std::ofstream file(filename, std::ios_base::app);
    if (!file) {
        throw std::runtime_error("Unable to open positions file");
    }
    file << std::setprecision(17) << step << ",[";
    for (int i = 0; i < N; ++i) {
        if (i != 0) {
            file << ',';
        }
        file << '[' << pos[i][0] << ',' << pos[i][1] << ',' << pos[i][2] << ']';
    }
    file << "]\n";
}

int main(int argc, char *argv[]) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <number_of_particles> <simulation_end_time>\n";
        return 1;
    }
    try {
        std::size_t used = 0;
        const std::string n_arg(argv[1]), time_arg(argv[2]);
        const int N = std::stoi(n_arg, &used);
        if (used != n_arg.size() || N <= 0) {
            throw std::invalid_argument("Particle count must be a positive integer");
        }
        const double tEnd = std::stod(time_arg, &used);
        const double steps = tEnd / dt;
        if (used != time_arg.size() || !std::isfinite(tEnd) || tEnd < 0.0 ||
            !std::isfinite(steps) || steps > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Invalid simulation end time");
        }
        // Preserve the provided C++ skeleton's truncation rule.
        const int Nt = static_cast<int>(steps);
        {
            std::unique_ptr<double[]> mass(new double[N]);
            std::unique_ptr<double[][3]> pos(new double[N][3]);
            std::unique_ptr<double[][3]> vel(new double[N][3]);
            std::unique_ptr<double[][3]> acc(new double[N][3]);
            // Fixed seed for reproducible tests and later serial/parallel comparisons.
            // C++ distributions need not generate the same samples as NumPy.
            std::mt19937 generator(17);
            std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
            std::normal_distribution<double> normal_dist(0.0, 1.0);
            for (int i = 0; i < N; ++i) {
                mass[i] = uniform_dist(generator);
                for (int k = 0; k < 3; ++k) {
                    pos[i][k] = normal_dist(generator);
                }
                for (int k = 0; k < 3; ++k) {
                    vel[i][k] = normal_dist(generator);
                }
            }
            double velCM[3] = {0.0, 0.0, 0.0};
            double totalMass = 0.0;
            for (int i = 0; i < N; ++i) {
                totalMass += mass[i];
                for (int k = 0; k < 3; ++k) {
                    velCM[k] += mass[i] * vel[i][k];
                }
            }
            if (!(totalMass > 0.0)) {
                throw std::runtime_error("Total mass must be positive");
            }
            for (int k = 0; k < 3; ++k) {
                velCM[k] /= totalMass;
            }
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < 3; ++k) {
                    vel[i][k] -= velCM[k];
                }
            }
            getAcc(pos.get(), mass.get(), acc.get(), N);
            for (int step = 0; step < Nt; ++step) {
                advanceStep(pos.get(), vel.get(), acc.get(), mass.get(), N);
                // For visualization only (disabled during performance measurements):
                // savePositionsToCSV(pos.get(), N, step, "positions.csv");
            }
        } // Include deallocation in timing, as in the provided skeleton.
        const auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << std::setprecision(12) << "time: " << ms << "ms\n";
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
