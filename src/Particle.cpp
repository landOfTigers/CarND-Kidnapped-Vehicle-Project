#include "Particle.h"

using std::vector;

Particle::Particle() {}

Particle::Particle(int id, double x, double y, double theta, double weight) {
    this->id = id;
    this->x = x;
    this->y = y;
    this->theta = theta;
    this->weight = weight;
}

double Particle::get_x() {
    return x;
}

double Particle::get_y() {
    return x;
}

double Particle::get_theta() {
    return x;
}

double Particle::get_weight() {
    return weight;
}

void
Particle::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate, default_random_engine &gen) {
    // calculate new position from motion data
    x += velocity * (sin(theta + yaw_rate * delta_t) - sin(theta)) / yaw_rate;
    y += velocity * (cos(theta) - cos(theta + yaw_rate * delta_t)) / yaw_rate;
    theta += yaw_rate * delta_t;

    // TODO: maybe extract function
    // create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    // update particle position with added noise
    x = dist_x(gen);
    y = dist_y(gen);
    theta = dist_theta(gen);
}

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

vector <LandmarkObs> Particle::transform_observations_coordinates(const vector <LandmarkObs> &observations) {
    vector <LandmarkObs> transformedObservations;
    for (auto &obs : observations) {
        double x_map = x + (cos(theta) * obs.x) - (sin(theta) * obs.y);
        double y_map = y + (sin(theta) * obs.x) + (cos(theta) * obs.y);
        transformedObservations.push_back({obs.id, x_map, y_map});
    }
    return transformedObservations;
}
