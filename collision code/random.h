#ifndef RANDOM_H
#define RANDOM_H

float random_uniform(float min, float max);
float random_gaussian(float mean, float sigma);
float sample_plummer_radius(float a);
Vector3 random_point_on_sphere(float r);

#endif