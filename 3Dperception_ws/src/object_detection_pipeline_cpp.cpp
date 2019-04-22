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
#include "visualization_msgs/MarkerArray.h"

#include "object_detection/object.h"
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

/** IMPORTANT:    // https://stackoverflow.com/questions/36380217/pclpclpointcloud2-usage
pcl::PCLPointCloud2 is a ROS message type replacing the old sensors_msgs::PointCloud2. Hence, it should only be used when interacting with ROS. (see an example here)
If needed, PCL provides two functions to convert from one type to the other:
void fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud);
void toPCLPointCloud2 (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg);
**/

typedef pcl::PointCloud<pcl::PointXYZRGB> CloudType;

// Publishers
ros::Publisher pb_filtered_cloud;
ros::Publisher pb_segmented_cloud;
ros::Publisher pb_beforeSeg_statistical_outlier;
ros::Publisher pb_afterSeg_statistical_outlier;
ros::Publisher pb_color_clustered;
ros::Publisher pb_objects_desc;
ros::Publisher pb_objects_3D_bbox;

ros::Publisher pb_region_cloud;

// Parameters
float passthrough_filter_x_low = 0;
float passthrough_filter_x_high  = 1.5;
float passthrough_filter_y_low = -0.85;
float passthrough_filter_y_high =  0.85;
float passthrough_filter_z_low = 0.003;
float passthrough_filter_z_high = 10;
float voxel_filter_downsampling_leaf = 0.01f;
float beforeSeg_statistical_outlier_removal_K = 20;
float beforeSeg_statistical_outlier_removal_deviation = 0.05;
float afterSeg_statistical_outlier_removal_K = 20;
float afterSeg_statistical_outlier_removal_deviation = 0.05;
float ransac_plane_segmentation_deviation = 0.01;
float euclidean_clustering_tolerance = 0.03;
float euclidean_clustering_max   = 150;
float euclidean_clustering_min   = 1000;

float region_k_search   = 50;
float region_min_cluster  = 20;
float region_max_cluster   = 10000;
float region_neighbours  = 30;
float region_smoothness   = 3.0;
float region_curvature = 1.0;

bool enable_passthrough_filter_x = true;
bool enable_passthrough_filter_y = true;
bool enable_passthrough_filter_z = true;
bool enable_voxel_filter_downsampling = true;
bool enable_beforeSeg_statistical_outlier_removal = true;
bool enable_ransac_plane_segmentation = true;
bool enable_afterSeg_statistical_outlier_removal = true;
bool enable_euclidean_clustering = true;
bool enable_3D_bounding_boxes = true;

bool enable_region_clustering = true;

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
  pass.setFilterLimits (passthrough_filter_y_low, passthrough_filter_y_high);  // -0.456, 0.456    //up (the less, the higher it can see ) and down (the more, the lower it can see)
  // pass.setFilterLimitsNegative (true);   // To view outliers
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
  pass.setFilterLimits (passthrough_filter_z_low, passthrough_filter_z_high);  // -0.456, 0.456    //up (the less, the higher it can see ) and down (the more, the lower it can see)
  // pass.setFilterLimitsNegative (true);   // To view outliers
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
// StatisticalOutlierRemoval IS VERY SLOW (Maybe fast on GPU) //TODO
void beforeSeg_statistical_outlier_removal(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr statistical_outlier_removed_cloud){
  if(!enable_beforeSeg_statistical_outlier_removal){
    *statistical_outlier_removed_cloud = *cloud;
    return;
  }
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (beforeSeg_statistical_outlier_removal_K); //20
  sor.setStddevMulThresh (beforeSeg_statistical_outlier_removal_deviation); //2
  // sor.setNegative (true);   // To view inliers
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
void ransac_plane_segmentation(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr segmented_objects_cloud, pcl::ModelCoefficients::Ptr plane_coefficients ){
  if(!enable_ransac_plane_segmentation){
    *segmented_objects_cloud = *cloud;
    return;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*cloud, *white_cloud);

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  pcl::PCLPointCloud2::Ptr extracted_plane (new pcl::PCLPointCloud2());

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (ransac_plane_segmentation_deviation); //0.006 // max distance
  seg.setMaxIterations(100);

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

  // Get the points associated with the planar surface
  extract.filter (*extracted_plane);
  // std::cout << "PointCloud representing the planar component: " << extracted_plane->data.size () << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  // extract.filter (*extracted_objects);
  extract.filter (*segmented_objects_cloud);
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



void publish_and_visualize_bounding_boxes(const std::vector<object_detection::Object> &objects){
    object_detection::DetectedObjectsArray detected_objects_array;
    visualization_msgs::MarkerArray marker_array_msg;

    for (size_t i = 0; i < objects.size(); ++i) {
      const object_detection::Object& object = objects[i];

      sensor_msgs::PointCloud2::Ptr object_cloud_ros (new sensor_msgs::PointCloud2);
      pcl_conversions::fromPCL(*object.object_cloud, *object_cloud_ros);

      object_detection::DetectedObject detected_object;
      detected_object.label = object.label;
      detected_object.confidence = object.confidence;
      // detected_object.object_cloud = *object_cloud_ros;
      detected_object.pose = object.pose;
      detected_object.dimensions = object.dimensions;
      detected_objects_array.objects.push_back(detected_object);

      // Publish a bounding box around it.
      visualization_msgs::Marker object_marker;
      object_marker.ns = "objects";
      object_marker.id = i;
      object_marker.header.frame_id = "front_base_footprint";
      object_marker.type = visualization_msgs::Marker::CUBE;
      object_marker.pose = object.pose;
      object_marker.scale = object.dimensions;
      object_marker.color.g = 1;
      object_marker.color.a = 0.3;
      // object_marker.duration = rospy.Duration(0.1);
      marker_array_msg.markers.push_back(object_marker);
    }
    pb_objects_3D_bbox.publish(marker_array_msg);
    pb_objects_desc.publish(detected_objects_array);

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

  std::cout << "Detected " << cluster_indices.size() << " objects" << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;

  extract.setInputCloud(segmented_objects_cloud);

  std::vector<object_detection::Object> objects;


   // // Fill in the cloud data
  for(int i=0; i<cluster_indices.size(); ++i){
    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = cluster_indices[i];
    std::cout << "Object " << (i+1)<< " has " <<indexes->indices.size() << " points" << std::endl;

    //----------------------------------------------------------------------------------
    // BOUNDING BOX DETECTION
    if (enable_3D_bounding_boxes){


      extract.setIndices(indexes);
      pcl::PCLPointCloud2::Ptr object_cloud (new pcl::PCLPointCloud2);
      extract.filter(*object_cloud);

      std::cout<<"3"<<std::endl;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_object_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
      pcl::fromPCLPointCloud2(*object_cloud, *white_object_cloud);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_out (new pcl::PointCloud<pcl::PointXYZRGB>);
      shape_msgs::SolidPrimitive shape;
      geometry_msgs::Pose obj_pose;
      object_detection::FitBox(*white_object_cloud, plane_coefficients, *extract_out, shape, obj_pose);
      //
      std::cout<<"4"<<std::endl;

      object_detection::Object obj;
      obj.label = std::to_string(i);
      obj.confidence = 0.0;
      obj.object_cloud = object_cloud;
      obj.pose = obj_pose;
      obj.dimensions.x = shape.dimensions[0];
      obj.dimensions.y = shape.dimensions[1];
      obj.dimensions.z = shape.dimensions[2];

      // REMOVE CLUSTERS with bounding box height greater than 0.1
      if (shape.dimensions[2]>0.1) break;
      if (obj_pose.position.z>0.05) break;

      std::cout<<"5"<<std::endl;

      objects.push_back(obj);

      publish_and_visualize_bounding_boxes(objects);

      std::cout<<"6"<<std::endl;
    }

    //----------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------
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
  //----------------------------------------------------------------------------------
  std::cout<<"7"<<std::endl;

  colored_masked_cluster->width = colored_masked_cluster->points.size();
  colored_masked_cluster->height = 1;
  colored_masked_cluster->is_dense = true;

  pcl::toPCLPointCloud2(*colored_masked_cluster, *colored_masked_cluster_cloud_ros);
  colored_masked_cluster_cloud_ros->header.frame_id = "front_base_footprint";
  std::cout<<"8"<<std::endl;

}


void region(const pcl::PCLPointCloud2ConstPtr& in_cloud, pcl::PCLPointCloud2::Ptr region_cloud){
  if(!enable_region_clustering){
    *region_cloud = *in_cloud;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2 (*in_cloud, *cloud);

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (region_k_search);
  normal_estimator.compute (*normals);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (region_min_cluster);
  reg.setMaxClusterSize (region_max_cluster);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (region_neighbours);
  reg.setInputCloud (cloud);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (region_smoothness / 180.0 * M_PI);
  reg.setCurvatureThreshold (region_curvature);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  // // Fill in the cloud data
  for(int i=0; i<clusters.size(); ++i){
    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = clusters[i];

    //----------------------------------------------------------------------------------
    // ASSIGN COLOR TO EACH CLUSTER
    for (int j =0; j<indexes->indices.size(); ++j){
      pcl::PointXYZRGB point ;
      point.x= cloud->points[indexes->indices[j]].x;
      point.y= cloud->points[indexes->indices[j]].y;
      point.z= cloud->points[indexes->indices[j]].z;
      uint32_t rgb = rgb_to_float(color_list[i]) ;
      point.rgb= *reinterpret_cast<float*>( &rgb );

      colored_cloud->points.push_back( point );
    }
  }
  //----------------------------------------------------------------------------------

  colored_cloud->width = colored_cloud->points.size();
  colored_cloud->height = 1;
  colored_cloud->is_dense = true;

  pcl::toPCLPointCloud2(*colored_cloud, *region_cloud);
  region_cloud->header.frame_id = "front_base_footprint";
}


pcl::PCLPointCloud2::Ptr passthrough_filter_x_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_y_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_z_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr voxel_filter_downsampled_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr beforeSeg_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr afterSeg_objects_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr segmented_objects_cloud  (new pcl::PCLPointCloud2) ;
pcl::PCLPointCloud2::Ptr colored_masked_cluster_cloud_ros (new pcl::PCLPointCloud2);

pcl::PCLPointCloud2::Ptr region_cloud (new pcl::PCLPointCloud2);

pcl::ModelCoefficients::Ptr plane_coefficients (new pcl::ModelCoefficients);

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& input_cloud)
{
  if (!(*input_cloud).is_dense){ // if true, then there are no NaNs, No need to convert to pcl::PointCloud<pcl::PointXYZRGB>
    ROS_INFO("Cloud not dense, process for NaNs");
  }

  // std::cout << "point cloud frame id: " << cloud->header.frame_id << std::endl;

  passthrough_filter_x( input_cloud , passthrough_filter_x_cloud);
  passthrough_filter_y( passthrough_filter_x_cloud , passthrough_filter_y_cloud);
  passthrough_filter_z( passthrough_filter_y_cloud , passthrough_filter_z_cloud);

  voxel_filter_downsampling( passthrough_filter_z_cloud , voxel_filter_downsampled_cloud);

  filtered_cloud = voxel_filter_downsampled_cloud;

  beforeSeg_statistical_outlier_removal( voxel_filter_downsampled_cloud , beforeSeg_outlier_removed_cloud);

  ransac_plane_segmentation( beforeSeg_outlier_removed_cloud , segmented_objects_cloud, plane_coefficients);
  afterSeg_statistical_outlier_removal( segmented_objects_cloud , afterSeg_objects_outlier_removed_cloud);

  color_euclidean_clusters_and_get_3D_bounding_boxes(afterSeg_objects_outlier_removed_cloud, plane_coefficients, colored_masked_cluster_cloud_ros);

  region(colored_masked_cluster_cloud_ros, region_cloud);

  pb_filtered_cloud.publish ( filtered_cloud );
  pb_beforeSeg_statistical_outlier.publish ( beforeSeg_outlier_removed_cloud );
  pb_segmented_cloud.publish ( segmented_objects_cloud );
  pb_afterSeg_statistical_outlier.publish ( afterSeg_objects_outlier_removed_cloud );
  pb_color_clustered.publish ( colored_masked_cluster_cloud_ros );

  pb_region_cloud.publish ( region_cloud );
}


void dyanamic_reconfigure_callback(object_detection::pcl_parametersConfig &config, uint32_t level) {
  enable_passthrough_filter_x = config.enable_passthrough_filter_x;
  enable_passthrough_filter_y = config.enable_passthrough_filter_y;
  enable_passthrough_filter_z = config.enable_passthrough_filter_z;
  enable_voxel_filter_downsampling = config.enable_voxel_filter_downsampling;
  enable_beforeSeg_statistical_outlier_removal = config.enable_beforeSeg_statistical_outlier_removal;
  enable_ransac_plane_segmentation = config.enable_ransac_plane_segmentation;
  enable_afterSeg_statistical_outlier_removal = config.enable_afterSeg_statistical_outlier_removal;
  enable_euclidean_clustering = config.enable_euclidean_clustering;
  enable_3D_bounding_boxes = config.enable_3D_bounding_boxes;

  enable_region_clustering = config.enable_region_clustering;

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
  euclidean_clustering_tolerance = config.euclidean_clustering_tolerance;
  euclidean_clustering_max   = config.euclidean_clustering_max ;
  euclidean_clustering_min   = config.euclidean_clustering_min;

  region_k_search   = config.region_k_search;
  region_min_cluster  = config.region_min_cluster;
  region_max_cluster   = config.region_max_cluster;
  region_neighbours  = config.region_neighbours;
  region_smoothness   = config.region_smoothness;
  region_curvature = config.region_curvature;

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

  pb_filtered_cloud = nh.advertise<pcl::PCLPointCloud2> ("/filtered_cloud", 1);
  pb_segmented_cloud = nh.advertise<pcl::PCLPointCloud2> ("/segmented_cloud", 1);
  pb_beforeSeg_statistical_outlier = nh.advertise<pcl::PCLPointCloud2> ("/before_seg_outlier_cloud", 1);
  pb_afterSeg_statistical_outlier = nh.advertise<pcl::PCLPointCloud2> ("/after_seg_outlier_cloud", 1);
  pb_color_clustered = nh.advertise<pcl::PCLPointCloud2> ("/color_clustered_cloud", 1);
  pb_objects_desc =  nh.advertise<object_detection::DetectedObjectsArray>("/objects_desc", 1);
  pb_objects_3D_bbox =  nh.advertise<visualization_msgs::MarkerArray>("/objects_3D_bbox", 1);

  pb_region_cloud = nh.advertise<pcl::PCLPointCloud2> ("/region_cloud", 1);

  ros::Subscriber sub = nh.subscribe<pcl::PCLPointCloud2>("/front_base_footprint/points", 1, cloud_cb);
  ros::spin();
}
