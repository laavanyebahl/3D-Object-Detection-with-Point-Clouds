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
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "object_detection/3D_box_fitter.h"
#include "object_detection/typedefs.h"

#include <object_detection/DetectedObjectsArray.h>
#include <object_detection/DetectedObject.h>

#include <sensor_msgs/PointCloud2.h>

#include <dynamic_reconfigure/server.h>
#include <object_detection/pcl_parametersConfig.h>


#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/features/boundary.h>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <cmath>
/** IMPORTANT:    // https://stackoverflow.com/questions/36380217/pclpclpointcloud2-usage
pcl::PCLPointCloud2 is a ROS message type replacing the old sensors_msgs::PointCloud2. Hence, it should only be used when interacting with ROS. (see an example here)
If needed, PCL provides two functions to convert from one type to the other:
void fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud);
void toPCLPointCloud2 (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg);
**/

typedef pcl::PointCloud<pcl::PointXYZRGB> CloudType;

#include <object_tracking/object_tracker.hpp>
using namespace object_tracking;

// Publishers
ros::Publisher pb_filtered_cloud;
ros::Publisher pb_beforeSeg_statistical_outlier;
ros::Publisher pb_segmented_cloud;
ros::Publisher pb_shrinked_segmented_cloud;
ros::Publisher pb_afterSeg_statistical_outlier;
ros::Publisher pb_color_clustered;
ros::Publisher pb_objects_pose;
ros::Publisher pb_objects_3D_bbox;

ros::Publisher pb_shrunk_hull_cloud;

// Parameters
float passthrough_filter_x_low = 0;
float passthrough_filter_x_high  = 1.6;
float passthrough_filter_y_low = -1;
float passthrough_filter_y_high =  1;
float passthrough_filter_z_low = -1;
float passthrough_filter_z_high = 0.25;
float voxel_filter_downsampling_leaf = 0.004f;
float beforeSeg_statistical_outlier_removal_K = 20;
float beforeSeg_statistical_outlier_removal_deviation = 0.01;
float afterSeg_statistical_outlier_removal_K = 20;
float afterSeg_statistical_outlier_removal_deviation = 0.25;
float ransac_plane_segmentation_deviation = 0.008;
float ransac_plane_segmentation_iterations = 100;

float euclidean_clustering_tolerance = 0.015;
float euclidean_clustering_max   = 125;
float euclidean_clustering_min   = 800;

bool enable_passthrough_filter_x = true;
bool enable_passthrough_filter_y = true;
bool enable_passthrough_filter_z = true;
bool enable_voxel_filter_downsampling = true;
bool enable_beforeSeg_statistical_outlier_removal = false;
bool enable_ransac_plane_segmentation = true;
bool enable_shrinking = true;
bool enable_afterSeg_statistical_outlier_removal = true;
bool enable_euclidean_clustering = true;
bool enable_3D_bounding_boxes = true;

std::unique_ptr<objectTracker> tracker;


struct Color
{
   int r, g, b;
};

std::vector<Color>  color_list;

void generate_color_list(){
  for(int i=0; i<500; ++i){
    color_list.push_back({ rand() % 255, rand() % 255, rand() % 255 });
  }
}

uint32_t rgb_to_float(Color color){
  uint8_t r = color.r, g = color.g, b = color.b;    // Example: Red color
  uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
  return rgb;
}


//----------------------------------------------------------------------------------
//PassThrough filter (X Axis)
//----------------------------------------------------------------------------------
void passthrough_filter_x(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr passthrough_filter_x_cloud){
  if(!enable_passthrough_filter_x){
    *passthrough_filter_x_cloud = *cloud;
    return;
  }
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (passthrough_filter_x_low, passthrough_filter_x_high);  // 0.6095, 1.1  //left and right
  // pass.setFilterLimitsNegative (true);   // To view outliers
  pass.filter (*passthrough_filter_x_cloud);
}


//----------------------------------------------------------------------------------
// PassThrough filter (Y Axis)
//----------------------------------------------------------------------------------
void passthrough_filter_y(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr passthrough_filter_y_cloud){
  if(!enable_passthrough_filter_y){
    *passthrough_filter_y_cloud = *cloud;
    return;
  }
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("y");
  pass.setFilterLimits (passthrough_filter_y_low, passthrough_filter_y_high);
  // pass.setFilterLimitsNegative (true);
  pass.filter (*passthrough_filter_y_cloud);
}


//----------------------------------------------------------------------------------
// PassThrough filter (Z Axis)
//----------------------------------------------------------------------------------
void passthrough_filter_z(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr passthrough_filter_z_cloud){
  if(!enable_passthrough_filter_z){
    *passthrough_filter_z_cloud = *cloud;
    return;
  }
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (passthrough_filter_z_low, passthrough_filter_z_high);
  // pass.setFilterLimitsNegative (true);
  pass.filter (*passthrough_filter_z_cloud);
}


//----------------------------------------------------------------------------------
//Voxel Grid Downsampling filter
//----------------------------------------------------------------------------------
void voxel_filter_downsampling(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr voxel_filter_downsampled_cloud){
  if(!enable_voxel_filter_downsampling){
    *voxel_filter_downsampled_cloud = *cloud;
    return;
  }
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (voxel_filter_downsampling_leaf, voxel_filter_downsampling_leaf, voxel_filter_downsampling_leaf);
  sor.filter (*voxel_filter_downsampled_cloud);
}

//----------------------------------------------------------------------------------
// Statistical Outlier Removal
//----------------------------------------------------------------------------------
void beforeSeg_statistical_outlier_removal(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr statistical_outlier_removed_cloud){
  if(!enable_beforeSeg_statistical_outlier_removal){
    *statistical_outlier_removed_cloud = *cloud;
    return;
  }
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (beforeSeg_statistical_outlier_removal_K); //20
  sor.setStddevMulThresh (beforeSeg_statistical_outlier_removal_deviation); //2
  // sor.setNegative (true);
  sor.filter (*statistical_outlier_removed_cloud);
}

void afterSeg_statistical_outlier_removal(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr statistical_outlier_removed_cloud){
  if(!enable_afterSeg_statistical_outlier_removal){
    *statistical_outlier_removed_cloud = *cloud;
    return;
  }
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (afterSeg_statistical_outlier_removal_K); //20
  sor.setStddevMulThresh (afterSeg_statistical_outlier_removal_deviation); //2
  // sor.setNegative (true);   // To view inliers
  sor.filter (*statistical_outlier_removed_cloud);
}

//----------------------------------------------------------------------------------
// RANSAC plane segmentation
//----------------------------------------------------------------------------------
void ransac_plane_segmentation(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr segmented_objects_cloud, pcl::PCLPointCloud2::Ptr extracted_plane, pcl::ModelCoefficients::Ptr plane_coefficients ){
  if(!enable_ransac_plane_segmentation){
    *segmented_objects_cloud = *cloud;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*cloud, *white_cloud);

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (ransac_plane_segmentation_deviation); //0.006 // max distance
  seg.setMaxIterations(ransac_plane_segmentation_iterations);
  seg.setInputCloud (white_cloud);
  seg.segment (*inliers, *plane_coefficients);

  if (inliers->indices.size () == 0)
  {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);
  extract.filter (*extracted_plane);
  extract.setNegative (true);
  extract.filter (*segmented_objects_cloud);
}

void shrink_cloud(const pcl::PCLPointCloud2ConstPtr& segmented_cloud, const pcl::PCLPointCloud2ConstPtr& extracted_plane, pcl::PCLPointCloud2::Ptr shrunk_segmented_cloud, pcl::PCLPointCloud2::Ptr shrunk_hull_cloud_pcl2){
  if(!enable_shrinking){
    *shrunk_segmented_cloud = *segmented_cloud;
    return;
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
  pcl::fromPCLPointCloud2(*segmented_cloud, *white_cloud);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_extracted_plane(new pcl::PointCloud<pcl::PointXYZRGB> );
  pcl::fromPCLPointCloud2(*extracted_plane, *white_extracted_plane);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_shrinked_segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );

  std::vector<pcl::Vertices> polygons;

  // Find the convex hull
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
  pcl::ConvexHull<pcl::PointXYZRGB> convex_hull;
  convex_hull.setInputCloud(white_extracted_plane);
  convex_hull.reconstruct(*hull_cloud, polygons);

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid (*hull_cloud, centroid);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr shrunk_hull_cloud (new pcl::PointCloud<pcl::PointXYZRGB> );

  for (size_t i = 0; i < hull_cloud->points.size (); ++i){
      pcl::PointXYZRGB point ;
      point.x= 0.88*(hull_cloud->points[i].x - centroid[0]) + centroid[0];
      point.y= 0.88*(hull_cloud->points[i].y - centroid[1]) + centroid[1];
      point.z= 1*(hull_cloud->points[i].z - centroid[2]) + centroid[2];
      point.rgb= 255;
      shrunk_hull_cloud->points.push_back( point );
  }

  shrunk_hull_cloud->width = shrunk_hull_cloud->points.size();
  shrunk_hull_cloud->height = 1;
  shrunk_hull_cloud->is_dense = true;

  pcl::toPCLPointCloud2(*shrunk_hull_cloud, *shrunk_hull_cloud_pcl2);
  shrunk_hull_cloud_pcl2->header.frame_id = "front_base_footprint";

  pcl::CropHull<pcl::PointXYZRGB> crop_hull;
  crop_hull.setInputCloud(white_cloud);
  crop_hull.setHullCloud(shrunk_hull_cloud);
  crop_hull.setHullIndices (polygons);
  crop_hull.setDim (2);
  crop_hull.filter (*white_shrinked_segmented_cloud);

  pcl::toPCLPointCloud2(*white_shrinked_segmented_cloud, *shrunk_segmented_cloud);
}

//----------------------------------------------------------------------------------
// Euclidean Clustering
//----------------------------------------------------------------------------------
void euclidean_clustering(const pcl::PCLPointCloud2ConstPtr& segmented_objects_cloud, std::vector<pcl::PointIndices> &cluster_indices){
  cluster_indices.clear();

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*segmented_objects_cloud, *white_cloud);
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (white_cloud);
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (euclidean_clustering_tolerance); // 3cm
  ec.setMinClusterSize (euclidean_clustering_max);  // Min number of popints to make a cluster
  ec.setMaxClusterSize (euclidean_clustering_min);
  ec.setSearchMethod (tree);
  ec.setInputCloud (white_cloud);
  ec.extract (cluster_indices);
}


void publish_and_visualize_bounding_boxes(const std::vector<object_detection::DetectedObject> &detected_objects){
    object_detection::DetectedObjectsArray detected_objects_msg;
    visualization_msgs::MarkerArray marker_array_msg;

    for (size_t i = 0; i < detected_objects.size(); ++i) {
      const object_detection::DetectedObject& detected_object = detected_objects[i];
      detected_objects_msg.objects.push_back(detected_object);

      // BROADCAST TRANSFORM
      static tf2_ros::TransformBroadcaster br;
      geometry_msgs::TransformStamped transformStamped;
      transformStamped.header.stamp = ros::Time::now();
      transformStamped.header.frame_id = "front_base_footprint";
      transformStamped.child_frame_id = "object_"+std::to_string(detected_object.id);
      transformStamped.transform.translation.x = detected_object.pose.pose.position.x;
      transformStamped.transform.translation.y = detected_object.pose.pose.position.y;
      transformStamped.transform.translation.z = detected_object.pose.pose.position.z;
      transformStamped.transform.rotation = detected_object.pose.pose.orientation;
      br.sendTransform(transformStamped);

      // PUBLISH 3D BOX
      visualization_msgs::Marker object_marker;
      object_marker.header.stamp = ros::Time::now();
      object_marker.lifetime = ros::Duration(0.15);
      object_marker.ns = "objects";
      object_marker.id = i;
      object_marker.header.frame_id = "front_base_footprint";
      object_marker.type = visualization_msgs::Marker::CUBE;
      object_marker.pose = detected_object.pose.pose;
      object_marker.scale = detected_object.dimensions;
      object_marker.color.b = 1;
      object_marker.color.a = 0.5;
      marker_array_msg.markers.push_back(object_marker);

    }
    pb_objects_3D_bbox.publish(marker_array_msg);
    pb_objects_pose.publish(detected_objects_msg);
}


//----------------------------------------------------------------------------------
// Create Cluster-Mask Point Cloud to visualize each cluster separately
//----------------------------------------------------------------------------------
void color_euclidean_clusters_and_get_3D_bounding_boxes(const pcl::PCLPointCloud2ConstPtr& segmented_objects_cloud, const pcl::ModelCoefficients::Ptr plane_coefficients,
   pcl::PCLPointCloud2::Ptr colored_masked_cluster_cloud_ros){

  if(!enable_euclidean_clustering){
    *colored_masked_cluster_cloud_ros = *segmented_objects_cloud;
    return;
  }

  std::vector<pcl::PointIndices> cluster_indices;
  euclidean_clustering(segmented_objects_cloud, cluster_indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*segmented_objects_cloud, *white_cloud);
  // std::cout << "Detected " << cluster_indices.size() << " objects" << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;
  extract.setInputCloud(segmented_objects_cloud);

  std::vector<object_detection::DetectedObject> objects;


  std::vector<Cluster> clusters;


   // // Fill in the cloud data
  for(int i=0; i<cluster_indices.size(); ++i){
    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = cluster_indices[i];
    std::cout << "\rObject " << (i+1)<< " : " <<indexes->indices.size() << " points" << std::endl;

    extract.setIndices(indexes);
    pcl::PCLPointCloud2::Ptr object_cloud (new pcl::PCLPointCloud2);
    extract.filter(*object_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_object_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
    pcl::fromPCLPointCloud2(*object_cloud, *white_object_cloud);

    Cluster::Ptr cluster(new Cluster(white_object_cloud));
    float dist = cluster->centroid.matrix().head<2>().norm();
    // if ( dist > 1.3) {
    //   continue;
    // }
    clusters.push_back(*cluster);
  }

  struct less_than_key
  {
      inline bool operator() (const Cluster& struct1, const Cluster& struct2)
      {
          float dist1 = struct1.centroid.matrix().head<2>().norm();
          float dist2 = struct2.centroid.matrix().head<2>().norm();
          return (dist1 < dist2);
      }
  };

  // Sort according to distance
  std::sort(clusters.begin(), clusters.end(), less_than_key());

  // update object tracker
  ros::Time time= ros::Time::now();
  tracker->predict(time);
  tracker->correct(time, clusters);

  for(int i=0; i<tracker->object.size(); i++) {
    const auto& track = tracker->object[i];
    const Cluster* associated = boost::any_cast<Cluster>(&track->lastAssociated());
    if(!associated) {
      continue;
    }

    // BOUNDING BOX DETECTION
    if (enable_3D_bounding_boxes){

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_object_cloud = associated->cloud;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_out (new pcl::PointCloud<pcl::PointXYZRGB>);
      shape_msgs::SolidPrimitive shape;
      geometry_msgs::Pose obj_pose;
      object_detection::FitBox(*white_object_cloud, plane_coefficients, *extract_out, shape, obj_pose);

      object_detection::DetectedObject detected_object;
      detected_object.id = track->id();
      // detected_object.confidence = 0.0;
      // detected_object.object_cloud = object_cloud;
      detected_object.pose.pose.position = obj_pose.position;
      detected_object.pose.pose.orientation = obj_pose.orientation;
      detected_object.dimensions.x = shape.dimensions[0];
      detected_object.dimensions.y = shape.dimensions[1];
      detected_object.dimensions.z = shape.dimensions[2];

      // REMOVE CLUSTERS
      if (detected_object.dimensions.z >0.15) break;
      if (detected_object.dimensions.z <0.012) break;
      if (detected_object.dimensions.z *detected_object.dimensions.y* detected_object.dimensions.z <0.00001) break;
      if (detected_object.pose.pose.position.z>0.09) break;
      float dist = sqrt( pow(detected_object.pose.pose.position.x,2) + pow(detected_object.pose.pose.position.y,2) + pow(detected_object.pose.pose.position.z,2) );
      if (dist>1.3) break;

      objects.push_back(detected_object);
      publish_and_visualize_bounding_boxes(objects);
    }
    // // ASSIGN COLOR TO EACH CLUSTER
    // for (int j =0; j<indexes->indices.size(); ++j){
    //   pcl::PointXYZRGB point ;
    //   point.x= white_cloud->points[indexes->indices[j]].x;
    //   point.y= white_cloud->points[indexes->indices[j]].y;
    //   point.z= white_cloud->points[indexes->indices[j]].z;
    //   uint32_t rgb = rgb_to_float(color_list[i]) ;
    //   point.rgb= *reinterpret_cast<float*>( &rgb );
    //
    //   colored_masked_cluster->points.push_back( point );
    // }

  }

  for(int i=0; i<cluster_indices.size(); ++i){
    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = cluster_indices[i];
    // ASSIGN COLOR TO EACH CLUSTER
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

  pcl::toPCLPointCloud2(*colored_masked_cluster, *colored_masked_cluster_cloud_ros);
  colored_masked_cluster_cloud_ros->header.frame_id = "front_base_footprint";
}


pcl::PCLPointCloud2::Ptr passthrough_filter_x_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_y_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_z_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr voxel_filter_downsampled_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr beforeSeg_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr afterSeg_objects_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr segmented_objects_cloud  (new pcl::PCLPointCloud2) ;
pcl::PCLPointCloud2::Ptr extracted_plane (new pcl::PCLPointCloud2) ;
pcl::PCLPointCloud2::Ptr shrunk_segmented_cloud (new pcl::PCLPointCloud2) ;
pcl::PCLPointCloud2::Ptr shrunk_hull_cloud (new pcl::PCLPointCloud2) ;
pcl::PCLPointCloud2::Ptr colored_masked_cluster_cloud_ros (new pcl::PCLPointCloud2);

pcl::ModelCoefficients::Ptr plane_coefficients (new pcl::ModelCoefficients);

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& input_cloud)
{
  if (!(*input_cloud).is_dense){ // if true, then there are no NaNs, No need to convert to pcl::PointCloud<pcl::PointXYZRGB>
    ROS_INFO("Cloud not dense, process for NaNs");
  }

  passthrough_filter_x( input_cloud , passthrough_filter_x_cloud);
  passthrough_filter_y( passthrough_filter_x_cloud , passthrough_filter_y_cloud);
  passthrough_filter_z( passthrough_filter_y_cloud , passthrough_filter_z_cloud);
  voxel_filter_downsampling( passthrough_filter_z_cloud , voxel_filter_downsampled_cloud);
  filtered_cloud = voxel_filter_downsampled_cloud;
  beforeSeg_statistical_outlier_removal( voxel_filter_downsampled_cloud , beforeSeg_outlier_removed_cloud);
  ransac_plane_segmentation( beforeSeg_outlier_removed_cloud , segmented_objects_cloud, extracted_plane, plane_coefficients);
  shrink_cloud(segmented_objects_cloud, extracted_plane, shrunk_segmented_cloud, shrunk_hull_cloud);
  afterSeg_statistical_outlier_removal( shrunk_segmented_cloud, afterSeg_objects_outlier_removed_cloud);
  color_euclidean_clusters_and_get_3D_bounding_boxes(afterSeg_objects_outlier_removed_cloud, plane_coefficients, colored_masked_cluster_cloud_ros);

  pb_filtered_cloud.publish ( filtered_cloud );
  pb_segmented_cloud.publish ( segmented_objects_cloud );
  pb_color_clustered.publish ( colored_masked_cluster_cloud_ros );

  pb_beforeSeg_statistical_outlier.publish ( beforeSeg_outlier_removed_cloud );
  pb_shrinked_segmented_cloud.publish(shrunk_segmented_cloud);
  pb_shrunk_hull_cloud.publish ( shrunk_hull_cloud );
  pb_afterSeg_statistical_outlier.publish ( afterSeg_objects_outlier_removed_cloud );
}


void dyanamic_reconfigure_callback(object_detection::pcl_parametersConfig &config, uint32_t level) {
  enable_passthrough_filter_x = config.enable_passthrough_filter_x;
  enable_passthrough_filter_y = config.enable_passthrough_filter_y;
  enable_passthrough_filter_z = config.enable_passthrough_filter_z;
  enable_voxel_filter_downsampling = config.enable_voxel_filter_downsampling;
  enable_beforeSeg_statistical_outlier_removal = config.enable_beforeSeg_statistical_outlier_removal;
  enable_ransac_plane_segmentation = config.enable_ransac_plane_segmentation;
  enable_shrinking = config.enable_shrinking;
  enable_afterSeg_statistical_outlier_removal = config.enable_afterSeg_statistical_outlier_removal;
  enable_euclidean_clustering = config.enable_euclidean_clustering;
  enable_3D_bounding_boxes = config.enable_3D_bounding_boxes;

  passthrough_filter_x_low = config.passthrough_filter_x_low ;
  passthrough_filter_x_high  = config.passthrough_filter_x_high;
  passthrough_filter_y_low = config.passthrough_filter_y_low ;
  passthrough_filter_y_high =  config.passthrough_filter_y_high;
  passthrough_filter_z_low = config.passthrough_filter_z_low ;
  passthrough_filter_z_high = config.passthrough_filter_z_high;
  voxel_filter_downsampling_leaf = config.voxel_filter_downsampling_leaf;
  beforeSeg_statistical_outlier_removal_K = config.beforeSeg_statistical_outlier_removal_K;
  beforeSeg_statistical_outlier_removal_deviation = config.beforeSeg_statistical_outlier_removal_deviation ;
  afterSeg_statistical_outlier_removal_K = config.afterSeg_statistical_outlier_removal_K;
  afterSeg_statistical_outlier_removal_deviation = config.afterSeg_statistical_outlier_removal_deviation ;
  ransac_plane_segmentation_deviation = config.ransac_plane_segmentation_deviation;
  ransac_plane_segmentation_iterations = config.ransac_plane_segmentation_iterations;
  euclidean_clustering_tolerance = config.euclidean_clustering_tolerance;
  euclidean_clustering_max   = config.euclidean_clustering_max ;
  euclidean_clustering_min   = config.euclidean_clustering_min;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detection_node_cpp");
  ros::NodeHandle nh;

  generate_color_list();

  dynamic_reconfigure::Server<object_detection::pcl_parametersConfig> server;
  dynamic_reconfigure::Server<object_detection::pcl_parametersConfig>::CallbackType f;
  f = boost::bind(&dyanamic_reconfigure_callback, _1, _2);
  server.setCallback(f);
  pb_beforeSeg_statistical_outlier = nh.advertise<pcl::PCLPointCloud2> ("/before_seg_outlier_cloud", 1);
  pb_shrinked_segmented_cloud = nh.advertise<pcl::PCLPointCloud2> ("/shrinked_segmented_cloud", 1);
  pb_shrunk_hull_cloud = nh.advertise<pcl::PCLPointCloud2> ("/region_cloud", 1);
  pb_afterSeg_statistical_outlier = nh.advertise<pcl::PCLPointCloud2> ("/after_seg_outlier_cloud", 1);

  pb_filtered_cloud = nh.advertise<pcl::PCLPointCloud2> ("/filtered_cloud", 1);
  pb_segmented_cloud = nh.advertise<pcl::PCLPointCloud2> ("/segmented_cloud", 1);
  pb_color_clustered = nh.advertise<pcl::PCLPointCloud2> ("/color_clustered_cloud", 1);
  pb_objects_pose =  nh.advertise<object_detection::DetectedObjectsArray>("/objects_pose", 1);
  pb_objects_3D_bbox =  nh.advertise<visualization_msgs::MarkerArray>("/objects_3D_bbox", 1);

  ros::Subscriber sub = nh.subscribe<pcl::PCLPointCloud2>("/front_base_footprint/points", 1, cloud_cb);

  tracker.reset(new objectTracker(nh));

  ros::spin();
}
