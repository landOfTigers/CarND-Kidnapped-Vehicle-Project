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

vector <LandmarkObs> Particle::create_predicted(double sensor_range, const Map &map_landmarks) {
    vector <LandmarkObs> predicted;
    for (auto const &landmark : map_landmarks.landmark_list) {
        double particle2LandmarkDistance = dist(x, y, landmark.x_f, landmark.y_f);
        if (particle2LandmarkDistance < sensor_range) {
            predicted.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
        }
    }
    return predicted;
}
