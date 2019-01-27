/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include <iterator>
#include "particle_filter.h"
#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 20;
    for (int i = 0; i < num_particles; i++) {
        Particle p(i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0f);
        weights.push_back(p.get_weight());
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    if (yaw_rate == 0) {
        yaw_rate = 0.0001;
    }

    for (auto &p : particles) {
        p.prediction(delta_t, std_pos, velocity, yaw_rate, gen);
    }
}

void ParticleFilter::dataAssociation(vector <LandmarkObs> predicted, vector <LandmarkObs> &observations) {
    for (auto &obs : observations) {
        // associate map landmark id of closest landmark
        double min_distance = std::numeric_limits<double>::max();
        for (auto const &pred : predicted) {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < min_distance) {
                min_distance = distance;
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector <LandmarkObs> &observations,
                                   const Map &map_landmarks) {

    for (auto &p : particles) {
        vector <LandmarkObs> predicted = p.create_predicted(sensor_range, map_landmarks);
        if (!predicted.empty()) {
            vector <LandmarkObs> transformedObservations = p.transform_observations_coordinates(observations);
            dataAssociation(predicted, transformedObservations);
            p.set_associations(transformedObservations);
            p.update_weight(transformedObservations, map_landmarks, std_landmark);
        }
    }

    // update weights vector
    weights.clear();
    for (auto &p : particles) {
        weights.push_back(p.get_weight());
    }

}

void ParticleFilter::resample() {
    discrete_distribution<> weights_distribution(weights.begin(), weights.end());
    vector <Particle> new_particles;
    for (int i = 0; i < num_particles; i++) {
        int random_index = weights_distribution(gen);
        new_particles.push_back(particles[random_index]);
    }
    particles = new_particles;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}