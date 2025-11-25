#ifndef RANDOM_H
#define RANDOM_H

double random_uniform(double min, double max);
double random_gaussian(double mean, double sigma);
double sample_plummer_radius(double a);
Vector3 random_point_on_sphere(double r);

#endif