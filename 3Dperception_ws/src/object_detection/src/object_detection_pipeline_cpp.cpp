#include <ros/ros.h>
// PCL specific includes
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>   // VERY IMPORTANT TO CONVERT  pcl::PCLPointCloud2

#include <boost/foreach.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/kdtree/kdtree.h>

#include "visualization_msgs/Marker.h"

#include "object_detection/object.h"
#include "object_detection/3D_box_fitter.h"
#include "object_detection/typedefs.h"

/** IMPORTANT:    // https://stackoverflow.com/questions/36380217/pclpclpointcloud2-usage
pcl::PCLPointCloud2 is a ROS message type replacing the old sensors_msgs::PointCloud2. Hence, it should only be used when interacting with ROS. (see an example here)
If needed, PCL provides two functions to convert from one type to the other:
void fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, cl::PointCloud<PointT>& cloud);
void toPCLPointCloud2 (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg);
**/

typedef pcl::PointCloud<pcl::PointXYZRGB> CloudType;

// Publishers
ros::Publisher pb_filtered_cloud;
ros::Publisher pb_segmented_cloud;
ros::Publisher pb_color_clustered;
ros::Publisher pb_object_marker;

//----------------------------------------------------------------------------------
//Voxel Grid Downsampling filter
//----------------------------------------------------------------------------------
pcl::PCLPointCloud2::Ptr voxel_filter_downsampling(const pcl::PCLPointCloud2ConstPtr& cloud){
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2());
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.005f, 0.005f, 0.005f);
  sor.filter (*cloud_filtered);
  return cloud_filtered;
}

//----------------------------------------------------------------------------------
// Statistical Outlier Removal
//----------------------------------------------------------------------------------
// StatisticalOutlierRemoval IS VERY SLOW (Maybe fast on GPU) //TODO
pcl::PCLPointCloud2::Ptr statistical_outlier_removal(const pcl::PCLPointCloud2ConstPtr& cloud){
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2());
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (20);
  sor.setStddevMulThresh (2);
  // sor.setNegative (true);   // To view inliers
  sor.filter (*cloud_filtered);
  return cloud_filtered;
}


//----------------------------------------------------------------------------------
//PassThrough filter (Horizontal Axis)
//----------------------------------------------------------------------------------
pcl::PCLPointCloud2::Ptr passthrough_filter_horizontal(const pcl::PCLPointCloud2ConstPtr& cloud){
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2());
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-2.5, 2.5);  // 0.6095, 1.1  //left and right
  // pass.setFilterLimitsNegative (true);   // To view outliers
  pass.filter (*cloud_filtered);
  return cloud_filtered;
}


//----------------------------------------------------------------------------------
// PassThrough filter (Vertical Axis)
//----------------------------------------------------------------------------------
pcl::PCLPointCloud2::Ptr passthrough_filter_vertical(const pcl::PCLPointCloud2ConstPtr& cloud){
  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2());
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("y");
  pass.setFilterLimits (-0.5, 5);  // -0.456, 0.456    //up (the less, the higher it can see ) and down (the more, the lower it can see)
  // pass.setFilterLimitsNegative (true);   // To view outliers
  pass.filter (*cloud_filtered);
  return cloud_filtered;
}


//----------------------------------------------------------------------------------
// RANSAC plane segmentation
//----------------------------------------------------------------------------------
pcl::PCLPointCloud2::Ptr ransac_plane_segmentation(const pcl::PCLPointCloud2ConstPtr& cloud){

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*cloud, *white_cloud);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  pcl::PCLPointCloud2::Ptr extracted_plane (new pcl::PCLPointCloud2());
  pcl::PCLPointCloud2::Ptr extracted_objects (new pcl::PCLPointCloud2());

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.008); //0.006 // max distance

  seg.setInputCloud (white_cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);

  // Get the points associated with the planar surface
  extract.filter (*extracted_plane);
  // std::cout << "PointCloud representing the planar component: " << extracted_plane->data.size () << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*extracted_objects);

  return extracted_objects;
}


//----------------------------------------------------------------------------------
// Euclidean Clustering
//----------------------------------------------------------------------------------
std::vector<pcl::PointIndices> euclidean_clustering(const pcl::PCLPointCloud2ConstPtr& extracted_objects){

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*extracted_objects, *white_cloud);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (white_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.03); // 3cm
  ec.setMinClusterSize (100);  // Min number of popints to make a cluster
  ec.setMaxClusterSize (5000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (white_cloud);
  ec.extract (cluster_indices);

  return cluster_indices;
}


struct Color
{
   int r, g, b;
};

uint32_t rgb_to_float(Color color){
  uint8_t r = color.r, g = color.g, b = color.b;    // Example: Red color
  uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
  return rgb;
}

//----------------------------------------------------------------------------------
// Create Cluster-Mask Point Cloud to visualize each cluster separately
//----------------------------------------------------------------------------------
pcl::PCLPointCloud2::Ptr color_euclidean_clusters_and_get_3D_bounding_boxes(const pcl::PCLPointCloud2ConstPtr& segmented_objects_cloud, const std::vector<pcl::PointIndices> &cluster_indices, std::vector<object_detection::Object> &objects){

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*segmented_objects_cloud, *white_cloud);

  // Generate a random color list
  std::vector<Color>  color_list;
  for(int i=0; i<cluster_indices.size(); ++i){
    color_list.push_back({ rand() % 255, rand() % 255, rand() % 255 });
  }
  std::cout << "Detected " << color_list.size () << " objects" << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;
  extract.setInputCloud(segmented_objects_cloud);

   // // Fill in the cloud data
  for(int i=0; i<cluster_indices.size(); ++i){


    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = cluster_indices[i];

    /**
    extract.setIndices(indexes);
    pcl::PCLPointCloud2::Ptr object_cloud(new pcl::PCLPointCloud2);
    extract.filter(*object_cloud);

    pcl::PCLPointCloud2::Ptr extract_out(new pcl::PCLPointCloud2);
    shape_msgs::SolidPrimitive shape;
    geometry_msgs::Pose obj_pose;
    object_detection::FitBox(*object_cloud, coeff, *extract_out, shape, obj_pose);

    object_detection::Object obj;
    obj.cloud = object_cloud;
    obj.pose = obj_pose;
    obj.dimensions.x = shape.dimensions[0];
    obj.dimensions.y = shape.dimensions[1];
    obj.dimensions.z = shape.dimensions[2];
    objects->push_back(obj);
    **/

    for (int j =0; j<indexes->indices.size(); ++j){
      pcl::PointXYZRGB point ;
      point.x= white_cloud->points[indexes->indices[j]].x;
      point.y= white_cloud->points[indexes->indices[j]].y;
      point.z= white_cloud->points[indexes->indices[j]].z;
      uint32_t rgb = rgb_to_float(color_list[i]) ;
      point.rgb= *reinterpret_cast<float*>( &rgb );

      colored_masked_cluster->points.push_back( point );
    }
  }

  colored_masked_cluster->width = colored_masked_cluster->points.size();
  colored_masked_cluster->height = 1;
  colored_masked_cluster->is_dense = true;

  pcl::PCLPointCloud2::Ptr colored_masked_cluster_ros (new pcl::PCLPointCloud2);
  pcl::toPCLPointCloud2(*colored_masked_cluster, *colored_masked_cluster_ros);

  colored_masked_cluster_ros->header.frame_id = "camera_depth_optical_frame";

  return colored_masked_cluster_ros;
}


// void GetAxisAlignedBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
//                                geometry_msgs::Pose* pose,
//                                geometry_msgs::Vector3* dimensions) {
//   Eigen::Vector4f min_pt, max_pt;
//   pcl::getMinMax3D(*cloud, min_pt, max_pt);
//
//   pose->position.x = (max_pt.x() + min_pt.x()) / 2;
//   pose->position.y = (max_pt.y() + min_pt.y()) / 2;
//   pose->position.z = (max_pt.z() + min_pt.z()) / 2;
//   pose->orientation.w = 1;
//
//   dimensions->x = max_pt.x() - min_pt.x();
//   dimensions->y = max_pt.y() - min_pt.y();
//   dimensions->z = max_pt.z() - min_pt.z();
// }


void visualize_bounding_boxes(const std::vector<object_detection::Object> &objects){
  for (size_t i = 0; i < objects.size(); ++i) {
    const object_detection::Object& object = objects[i];
    // Publish a bounding box around it.
    visualization_msgs::Marker object_marker;
    object_marker.ns = "objects";
    object_marker.id = i;
    object_marker.header.frame_id = "base_link";
    object_marker.type = visualization_msgs::Marker::CUBE;
    object_marker.pose = object.pose;
    object_marker.scale = object.dimensions;
    object_marker.color.g = 1;
    object_marker.color.a = 0.3;
    pb_object_marker.publish(object_marker);
  }
}


pcl::PCLPointCloud2::Ptr filtered_cloud;
pcl::PCLPointCloud2::Ptr segmented_objects_cloud;
std::vector<pcl::PointIndices> cluster_indices;
pcl::PCLPointCloud2::Ptr colored_masked_cluster;

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud)
{
  if (!(*cloud).is_dense){ // if true, then there are no NaNs, No need to convert to pcl::PointCloud<pcl::PointXYZRGB>
    ROS_INFO("Cloud not dense, process for NaNs");
  }

  // std::cout << "point cloud frame id: " << cloud->header.frame_id << std::endl;

  filtered_cloud =  passthrough_filter_vertical( passthrough_filter_horizontal( statistical_outlier_removal( voxel_filter_downsampling( cloud ) ) ) );
  segmented_objects_cloud = ransac_plane_segmentation( filtered_cloud );
  cluster_indices =  euclidean_clustering(segmented_objects_cloud);

  // colored_masked_cluster = cluster_masking(segmented_objects_cloud, cluster_indices);
  std::vector<object_detection::Object> objects;
  colored_masked_cluster = color_euclidean_clusters_and_get_3D_bounding_boxes(segmented_objects_cloud, cluster_indices, objects);

  pb_filtered_cloud.publish ( filtered_cloud );
  pb_segmented_cloud.publish ( segmented_objects_cloud );
  pb_color_clustered.publish ( colored_masked_cluster );

  // visualize_bounding_boxes(objects);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detection_node_cpp");
  ros::NodeHandle nh;

  pb_filtered_cloud = nh.advertise<pcl::PCLPointCloud2> ("/filtered_cloud", 1);
  pb_segmented_cloud = nh.advertise<pcl::PCLPointCloud2> ("/segmented_cloud", 1);
  pb_color_clustered = nh.advertise<pcl::PCLPointCloud2> ("/color_clustered_cloud", 1);

  ros::Subscriber sub = nh.subscribe<pcl::PCLPointCloud2>("/camera/depth/color/points", 1, cloud_cb);
  ros::spin();
}
