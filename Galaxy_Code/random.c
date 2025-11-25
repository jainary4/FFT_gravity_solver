#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "structs.h"
#include "random.h"

double random_uniform(double min, double max) {
    return min + (max - min) * (rand() / (double)RAND_MAX);
}

double random_gaussian(double mean, double sigma) {
    static int has_spare = 0;      
    static double spare;            
    
    if (has_spare) {
        has_spare = 0;
        return mean + sigma * spare;
    }
    
    double u, v, s;
    do {
        u = random_uniform(-1.0f, 1.0f);
        v = random_uniform(-1.0f, 1.0f);
        s = u*u + v*v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = 1;
    
    return mean + sigma * u * s;
}

double sample_plummer_radius(double a) {
    double u = random_uniform(0.0f, 1.0f);
    double term = powf(u, -2.0f/3.0f) - 1.0f;
    if (term <= 0.0f) term = 1e-6f;
    return a / sqrtf(term);
}

Vector3 random_point_on_sphere(double r) {
    Vector3 pos;
    double theta = random_uniform(0.0f, 2.0f * PI);
    double cos_phi = random_uniform(-1.0f, 1.0f);
    double sin_phi = sqrtf(1.0f - cos_phi*cos_phi);
    
    pos.x = r * sin_phi * cosf(theta);
    pos.y = r * sin_phi * sinf(theta);
    pos.z = r * cos_phi;
    
    return pos;
}