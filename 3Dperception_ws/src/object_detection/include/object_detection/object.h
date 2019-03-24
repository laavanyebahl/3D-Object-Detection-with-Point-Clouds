#ifndef _OBJECT_DETECTION_OBJECT_H_
#define _OBJECT_DETECTION_OBJECT_H_

#include <string>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Vector3.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace object_detection {
struct Object {
  std::string name;
  double confidence;
  pcl::PCLPointCloud2::Ptr cloud;
  geometry_msgs::Pose pose;
  geometry_msgs::Vector3 dimensions;
};
}  // namespace object_detection

#endif  // _OBJECT_DETECTION_OBJECT_H_
