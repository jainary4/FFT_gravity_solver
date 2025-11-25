#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "structs.h"
#include "random.h"

double random_uniform(double min, double max) {
    return min + (max - min) * (rand() / (double)RAND_MAX);
}

double random_gaussian(double mean, double sigma) {
    static int    has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + sigma * spare;
    }

    double u, v, s;
    do {
        u = random_uniform(-1.0, 1.0);
        v = random_uniform(-1.0, 1.0);
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);

    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul;
    has_spare = 1;
    return mean + sigma * (u * mul);
}

double sample_plummer_radius(double a) {
    double u = random_uniform(0.0, 1.0);
    double term = pow(u, -2.0/3.0) - 1.0;
    if (term <= 0.0) term = 1e-6;
    return a / sqrt(term);
}

Vector3 random_point_on_sphere(double r) {
    Vector3 pos;
    double theta = random_uniform(0.0, 2.0 * PI);
    double cos_phi = random_uniform(-1.0, 1.0);
    double sin_phi = sqrt(1.0 - cos_phi*cos_phi);
    
    pos.x = r * sin_phi * cos(theta);
    pos.y = r * sin_phi * sin(theta);
    pos.z = r * cos_phi;
    
    return pos;
}