#pragma once

#include <tuple>

using std::tuple;

template <typename D>
struct Data {
  virtual D randomSelect(int n) = 0;
};

template <typename D>
struct Model {
  virtual void fit(D& data) = 0;
  virtual int evaluate(Data<D>& data, float inliers_threshold) = 0;
  virtual Model* clone() const = 0;
};  

template <typename M, typename D>
tuple<bool,Model<D>*> RANSAC(Data<D>& data, int num_iterations, float inliers_threshold, int n) {
  Model<D>* best_model; int best_score = 0;
  int iteration = 0; M model;

  while (iteration++ < num_iterations) {
    D random_data = data.randomSelect(n);
    model.fit(random_data);
    int score = model.evaluate(data, inliers_threshold);
    
    if (score > best_score) {
      best_model = model.clone();
      best_score = score;
    }
  }
  if (best_score == 0) {
    return {false, nullptr};
  }
  else return {true, best_model};
}