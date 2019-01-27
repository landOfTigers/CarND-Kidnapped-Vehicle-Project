#ifndef PROJECT_PARTICLE_H
#define PROJECT_PARTICLE_H

#include <random>
#include "helper_functions.h"

using std::normal_distribution;
using std::default_random_engine;

class Particle {
private:
    int id;
    double x;
    double y;
    double theta;
    double weight;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
public:
    Particle();

    Particle(int id, double x, double y, double theta, double weight);

    double get_x();

    double get_y();

    double get_theta();

    double get_weight();

    std::vector<int> get_associations();

    std::vector<double> get_sense_x();

    std::vector<double> get_sense_y();

    void set_associations(std::vector <LandmarkObs> &transformedObservations);

    void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate, default_random_engine &gen);

    void update_weight(const std::vector <LandmarkObs> transformedObservations, const Map &map_landmarks,
                       const double *std_landmark);

    // adds those map landmarks within sensor range of the particle to the predicted vector
    std::vector <LandmarkObs> create_predicted(double sensor_range, const Map &map_landmarks);

    // transforms observations from vehicle to map coordinates
    std::vector <LandmarkObs> transform_observations_coordinates(const std::vector <LandmarkObs> &observations);

};

#endif //PROJECT_PARTICLE_H
