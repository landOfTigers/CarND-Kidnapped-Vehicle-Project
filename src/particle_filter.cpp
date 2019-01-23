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

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

    // Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 100;  // TODO: Set the number of particles
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0f;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

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
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    for (auto &obs : observations) {
        // associate map landmark id of closest landmark
        float min_dist = 1e6;
        for (auto const &pred : predicted) {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < min_dist) {
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a multi-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    for (auto &p : particles) {

        // add those map landmarks within sensor range of the particle to the predicted vector
        vector<LandmarkObs> predicted;
        for (auto const &landmark : map_landmarks.landmark_list) {
            double particle2LandmarkDistance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
            if (particle2LandmarkDistance < sensor_range) {
                LandmarkObs mappedLandmark{
                        landmark.id_i,
                        landmark.x_f,
                        landmark.y_f
                };
                predicted.push_back(mappedLandmark);
            }
        }

        // predict landmark measurements (transform from vehicle to map coordinates)
        vector<LandmarkObs> transformedObservations;
        for (auto &obs : observations) {
            double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
            double y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
            LandmarkObs transformedObservation{
                    obs.id,
                    x_map,
                    y_map
            };
            transformedObservations.push_back(transformedObservation);
        }

        // use dataAssociation fct to associate sensor measurements to map landmarks
        dataAssociation(predicted, transformedObservations);

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        for (auto const &obs : transformedObservations) {
            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        SetAssociations(p, associations, sense_x, sense_y);

        // determine weight
        double weight = 1.0;
        for (auto const &obs : transformedObservations) {
//           // find landmark with obs.id in predicted
//           LandmarkObs associatedLandmark;
//           for(const auto & landmark : predicted) {
//               if(landmark.id == obs.id) {
//                   associatedLandmark = landmark;
//               }
//           }

            auto lm = map_landmarks.landmark_list[obs.id - 1];
            LandmarkObs associatedLandmark{
                    lm.id_i,
                    lm.x_f,
                    lm.y_f
            };

            weight *= multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, associatedLandmark.x,
                                  associatedLandmark.y);
        }

        // TODO: normalize weight?

        p.weight = weight;
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

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