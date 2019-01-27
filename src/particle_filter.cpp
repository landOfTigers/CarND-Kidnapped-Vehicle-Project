/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 20;  // TODO: Set the number of particles
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0f;
        weights.push_back(p.weight);
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    if (yaw_rate == 0) {
        yaw_rate = 0.0001;
    }

    for (auto &p : particles) {
        // update particle position
        p.x += velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) / yaw_rate;
        p.y += velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) / yaw_rate;
        p.theta += yaw_rate * delta_t;

        // Create normal distributions for x, y and theta
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);

        // add noise
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector <LandmarkObs> predicted,
                                     vector <LandmarkObs> &observations) {
    for (auto &obs : observations) {
        // associate map landmark id of closest landmark
        double min_dist = std::numeric_limits<double>::max();
        for (auto const &pred : predicted) {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < min_dist) {
                min_dist = distance;
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector <LandmarkObs> &observations,
                                   const Map &map_landmarks) {

    for (auto &p : particles) {

        // add those map landmarks within sensor range of the particle to the predicted vector
        vector <LandmarkObs> predicted;
        for (auto const &landmark : map_landmarks.landmark_list) {
            double particle2LandmarkDistance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
            if (particle2LandmarkDistance < sensor_range) {
                predicted.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        // predict landmark measurements (transform from vehicle to map coordinates)
        vector <LandmarkObs> transformedObservations;
        for (auto &obs : observations) {
            double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
            double y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
            transformedObservations.push_back({obs.id, x_map, y_map});
        }

        if (!predicted.empty()) {
            // use dataAssociation fct to associate sensor measurements to map landmarks
            dataAssociation(predicted, transformedObservations);

            // update particle weight
            double weight = 1.0;
            for (auto const &obs : transformedObservations) {
                auto lm = map_landmarks.landmark_list[obs.id - 1];
                weight *= multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, lm.x_f, lm.y_f);
            }
            p.weight = weight;
        }
    }

    // update weights vector
    weights.clear();
    for (auto const &p : particles) {
        weights.push_back(p.weight);
    }

}

void ParticleFilter::resample() {
    std::discrete_distribution<> weights_distribution(weights.begin(), weights.end());
    vector <Particle> new_particles;
    for (int i = 0; i < num_particles; i++) {
        int random_index = weights_distribution(gen);
        new_particles.push_back(particles[random_index]);
    }
    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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