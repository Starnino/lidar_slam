#include "projector.hpp"

Projector::Projector(int height, int width, float fov_up, float fov_down, float max_depth, float max_intensity) {
  _height = height;
  _width = width;
  _fov_up = fov_up;
  _fov_down = fov_down;
  _max_depth = max_depth;
  _max_intensity = max_intensity;
}