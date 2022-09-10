#pragma once

#include <math.h>
#include <tuple>

using std::tuple;

tuple<float,float,float> geodetic2ecef(double latitude, double longitude);