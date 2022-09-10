#include "gps_helper.hpp"

double EarthRadius(double latitude_radians) {
  // latitudeRadians is geodetic, i.e. that reported by GPS.
  // http://en.wikipedia.org/wiki/Earth_radius
  double a = 6378137.0;  // equatorial radius in meters
  double b = 6356752.3;  // polar radius in meters
  double c = cos(latitude_radians);
  double s = sin(latitude_radians);
  double t1 = a * a * c;
  double t2 = b * b * s;
  double t3 = a * c;
  double t4 = b * s;
  return sqrt((t1 * t1 + t2 * t2) / (t3 * t3 + t4 * t4));
}

double GeocentricLatitude(double lat) {
  // Convert geodetic latitude 'lat' to a geocentric latitude 'glat'.
  // Geodetic latitude is the latitude as given by GPS.
  // Geocentric latitude is the angle measured from center of Earth between a
  // point and the equator.
  // https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
  double e2 = 0.00669437999014;
  double glat = atan((1.0 - e2) * tan(lat));
  return glat;
}

tuple<float,float,float> geodetic2ecef(double latitude, double longitude) {
  // Convert geodetic coordinates latitue, longitude, 
  // to ECEF (Earth-centered, Earth-fixed) coordinates x, y, z.
  // https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
  double lat = latitude * M_PI / 180.f;
  double lon = longitude * M_PI / 180.f;
  double radius = EarthRadius(lat);
  double glat = GeocentricLatitude(lat);

  double cos_lon = cos(lon);
  double sin_lon = sin(lon);
  double cos_lat = cos(glat);
  double sin_lat = sin(glat);
  double x = radius * cos_lat * cos_lon;
  double y = radius * cos_lat * sin_lon;
  double z = radius * sin_lat;

  return {x, y, z};
}