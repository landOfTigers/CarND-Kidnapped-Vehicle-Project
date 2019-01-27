#ifndef PROJECT_PARTICLE_H
#define PROJECT_PARTICLE_H

#include "helper_functions.h"

class Particle {
    // TODO: make some fields private
public:
    int id;
    double x;
    double y;
    double theta;
    double weight;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    void update_weight(const std::vector <LandmarkObs> transformedObservations, const Map &map_landmarks,
                       const double *std_landmark);
};

#endif //PROJECT_PARTICLE_H
