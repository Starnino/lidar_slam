#pragma once

class Projector {

  private:
    int _height;
    int _width;
    float _fov_up;
    float _fov_down;
    float _max_depth;
    float _max_intensity;

  public:
    explicit Projector(int height, int width, float fov_up, float fov_down, float max_depth, float max_intensity);

    inline const int& height() { return _height; };
    inline const int& width() { return _width; };
    inline const float& fov_up() { return _fov_up; };
    inline const float& fov_down() { return _fov_down; };
    inline const float& max_depth() { return _max_depth; };
    inline const float& max_intensity() { return _max_intensity; };
};