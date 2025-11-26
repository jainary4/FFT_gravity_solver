#ifndef RANDOM_H
#define RANDOM_H

#include "structs.h"

/* Thread-safe random number utilities (OpenMP). */

/* Uniform in [min, max]. */
double random_uniform(double min, double max);

/* Gaussian with given mean and sigma. */
double random_gaussian(double mean, double sigma);

/* Sample radius from a Plummer sphere with scale length a. */
double sample_plummer_radius(double a);

/* Random point on a sphere of radius r. */
Vector3 random_point_on_sphere(double r);

#endif
