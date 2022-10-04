#include "icp.hpp"

Affine3f vector2transform(Vector6f v) {
  AngleAxisf Rx(v(3), Vector3f::UnitX());
  AngleAxisf Ry(v(4), Vector3f::UnitY());
  AngleAxisf Rz(v(5), Vector3f::UnitZ());
  Quaternionf q = Rx*Ry*Rz;
  Affine3f transform(q.matrix());
  transform.translation() = Vector3f{v(0),v(1),v(2)};
  return transform;
}

Matrix3f skew(Vector3f v) {
  Matrix3f S = Matrix3f::Zero();
  S.block<1,3>(0,0) = Vector3f{0.f,-v(2),v(1)};
  S.block<1,3>(1,0) = Vector3f{v(2),0.f,-v(0)};
  S.block<1,3>(2,0) = Vector3f{-v(1),v(0),0.f};
  return S;
}

tuple<Vector3f,Matrix3x6f> errorAndJacobian(Affine3f pose, Point3f p, Point3f z) {
  Point3f z_hat = pose*p;
  Vector3f e = z_hat - z;
  Matrix3x6f J = Matrix3x6f::Zero(); 
  J.block<3,3>(0,0).setIdentity();
  J.block<3,3>(0,3) = -skew(z_hat); 
  return {e,J};
}

tuple<Affine3f,vector<float>> ICP(Pointset3f points, int iterations, float kernel_threshold, float damping, Affine3f guess) {
  Affine3f pose = guess;
  vector<float> chi_tot(iterations);
  Matrix6f H;
  Vector6f b;

  for (int i = 0; i < iterations; ++i) {
    H.setZero();
    b.setZero();

    for (pair<Point3f,Point3f>& pointpair : points) {
      auto [e,J] = errorAndJacobian(pose, pointpair.first, pointpair.second);
  
      // kernel
      float chi = e.dot(e);
      float lambda = 1;
      if (chi > kernel_threshold) {
        lambda = sqrt(kernel_threshold/chi);
        chi = kernel_threshold;
      }
      chi_tot[i] += chi;

      // update H and b
      H += J.transpose()*J*lambda;
      b += J.transpose()*e*lambda;
    }

    H += Matrix6f::Identity()*damping;
    Vector6f delta = H.ldlt().solve(-b);
    pose = vector2transform(delta)*pose;    
  }
  return {pose, chi_tot};
}

