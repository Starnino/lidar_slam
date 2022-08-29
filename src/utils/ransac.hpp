#pragma once
#include <iostream>

template <typename D>
struct Data {
  virtual D randomSelect(int n) = 0;
};

template <typename D>
struct Model {
  virtual void fit(D& data) = 0;
  virtual int evaluate(Data<D>& data, float inlier_threshold) = 0;
  virtual Model* clone() const = 0;
};  

template <typename D>
Model<D>* RANSAC(Model<D>& model, Data<D>& data, int num_iterations, float inlier_threshold, int n) {
  int iteration = 0; int best_score = 0;
  Model<D>* best_model;

  while (iteration++ < num_iterations) {
    
    D random_data = data.randomSelect(n);
    model.fit(random_data);
    int score = model.evaluate(data, inlier_threshold);

    if (score > best_score) {
      best_model = model.clone();
      best_score = score;
    }
  }
  return best_model;
}