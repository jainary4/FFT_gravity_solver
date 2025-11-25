#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "structs.h"
#include "random.h"

float random_uniform(float min, float max) {
    return min + (max - min) * (rand() / (float)RAND_MAX);
}

float random_gaussian(float mean, float sigma) {
    static int has_spare = 0;      
    static float spare;            
    
    if (has_spare) {
        has_spare = 0;
        return mean + sigma * spare;
    }
    
    float u, v, s;
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

float sample_plummer_radius(float a) {
    float u = random_uniform(0.0f, 1.0f);
    float term = powf(u, -2.0f/3.0f) - 1.0f;
    if (term <= 0.0f) term = 1e-6f;
    return a / sqrtf(term);
}

Vector3 random_point_on_sphere(float r) {
    Vector3 pos;
    float theta = random_uniform(0.0f, 2.0f * PI);
    float cos_phi = random_uniform(-1.0f, 1.0f);
    float sin_phi = sqrtf(1.0f - cos_phi*cos_phi);
    
    pos.x = r * sin_phi * cosf(theta);
    pos.y = r * sin_phi * sinf(theta);
    pos.z = r * cos_phi;
    
    return pos;
}