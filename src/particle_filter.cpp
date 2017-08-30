/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

	default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
    Particle p = Particle();

    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    
    particles.push_back(p);
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
	default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  if(abs(yaw_rate) > 1.0e-6) {
    for (int i = 0 ; i < num_particles ; i++){
      double x = particles[i].x;
      double y = particles[i].y;
      double theta = particles[i].theta;
      
      particles[i].x = x + velocity*(sin(theta + yaw_rate*delta_t) - sin(theta))/yaw_rate + dist_x(gen);
      particles[i].y = y + velocity*(cos(theta) - cos(theta+yaw_rate*delta_t))/yaw_rate + dist_y(gen);
      particles[i].theta = theta + yaw_rate*delta_t + dist_theta(gen);
    }
  } else {
    for (int i = 0 ; i < num_particles ; i++){
      double x = particles[i].x;
      double y = particles[i].y;
      double theta = particles[i].theta;
      
      particles[i].x = x + velocity*cos(theta)*delta_t + dist_x(gen);
      particles[i].y = y + velocity*sin(theta)*delta_t + dist_y(gen);
      particles[i].theta = theta + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for(int i = 0 ; i < observations.size() ; i++){
      double min_dist = 0;
      int min_ID = 0;
      
      for(int k = 0 ; k < predicted.size() ; k++) {
        double diff_dist_x = predicted[k].x - observations[i].x;
        double diff_dist_y = predicted[k].y - observations[i].y;
    
        double diff_dist = diff_dist_x*diff_dist_x + diff_dist_y*diff_dist_y;
        if(k == 0) {
          min_dist = diff_dist;
          min_ID = 0;
        } else if(min_dist > diff_dist) {
          min_dist = diff_dist;
          min_ID = k;
        }
      }
      observations[i].id = predicted[min_ID].id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  //Convert the coordinate system
  //Loop through all particles
  for (int i = 0 ; i < num_particles ; i++){
    double x_par = particles[i].x;
    double y_par = particles[i].y;
    double theta_par = particles[i].theta;
    double final_weight = 0.0;
    
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    
    for(int j = 0 ; j < observations.size() ; j++){
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      
      double xm = x_par + (cos(theta_par) * x_obs) - (sin(theta_par)*y_obs);
      double ym = y_par + (sin(theta_par) * x_obs) + (cos(theta_par)*y_obs);
      
//      observations[j].x = xm;
//      observations[j].y = ym;
      
      double dist_x, dist_y, dist, min_dist;
      int minID = 0;

      //Loop through all landmarks and find the best association
      for(int k = 0 ; k < map_landmarks.landmark_list.size() ; k++){
        dist_x = map_landmarks.landmark_list[k].x_f - xm;
        dist_y = map_landmarks.landmark_list[k].y_f - ym;
        dist = dist_x*dist_x + dist_y*dist_y;
        if(k == 0) {
          min_dist = dist;
          minID = 0;
        } else if (min_dist > dist) {
          min_dist = dist;
          minID = k;
        }
      }
      
      //Calculate weight
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double mu_x= map_landmarks.landmark_list[minID].x_f;
      double mu_y= map_landmarks.landmark_list[minID].y_f;
      
      // calculate normalization term
      double gauss_norm = 1/(2 * M_PI * sig_x * sig_y);
      
      // calculate exponent
      double exponent= ((xm - mu_x)*(xm - mu_x))/(2 * sig_x*sig_x) + ((ym - mu_y)*(ym - mu_y))/(2 * sig_y*sig_y);
      
      // calculate weight using normalization terms and exponent
      double weight= gauss_norm * exp(-exponent);
      //cout << "weight = " << weight << endl;
      
      //Loop through number of obeservations
      if(j == 0) {
        final_weight = weight;
      } else {
        final_weight = final_weight * weight;
      }
      associations.push_back(map_landmarks.landmark_list[minID].id_i);
      sense_x.push_back(xm);
      sense_y.push_back(ym);
    }
    particles[i].weight = final_weight;
//    cout << "particles[i].weight =" << particles[i].weight << endl;
    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
  }
  
  double total_weight = 0;
  for(int l = 0 ; l < particles.size() ; l++){
    total_weight += particles[l].weight;
  }
  
  if(total_weight != 0) {
    for(int l = 0 ; l < particles.size() ; l++){
      particles[l].weight = particles[l].weight/total_weight;
      cout << particles[l].weight;
      weights[l] = particles[l].weight;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  
//  std::vector<double> probabilities;
//  for(int i = 0 ; i < particles.size() ; i++){
//    probabilities.push_back(particles[i].weight);
//  }
  
  std::discrete_distribution<> dist {
    weights.begin(),
    weights.end()
  };
  
  std::vector<Particle> particles_new;
  std::vector<double> weights_new;
  for(int j = 0 ; j < particles.size() ; j++){
    int index = dist(seed_gen);
    particles_new.push_back(particles[index]);
    cout << "index = " << index << endl;
    weights_new.push_back(weights[index]);
  }
  particles = particles_new;
  weights = weights_new;
  
  double total_weight = 0.0;
  for(int i = 0 ; i < particles.size() ; i++){
    total_weight += particles[i].weight;
  }
  for(int k = 0 ; k < particles.size() ; k++){
    particles[k].weight = particles[k].weight/total_weight;
    weights[k] = weights[k]/total_weight;
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
