#include "Particle.h"

using std::vector;

void Particle::update_weight(const vector <LandmarkObs> transformedObservations, const Map &map_landmarks,
                             const double *std_landmark) {
    double weight = 1.0;
    for (auto const &obs : transformedObservations) {
        auto lm = map_landmarks.landmark_list[obs.id - 1];
        weight *= multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, lm.x_f, lm.y_f);
    }
    this->weight = weight;
}
