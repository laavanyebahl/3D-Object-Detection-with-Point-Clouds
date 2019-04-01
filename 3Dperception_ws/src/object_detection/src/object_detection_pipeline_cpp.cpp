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
  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (beforeSeg_statistical_outlier_removal_K); //20
  sor.setStddevMulThresh (beforeSeg_statistical_outlier_removal_deviation); //2
  // sor.setNegative (true);   // To view inliers
  sor.filter (*statistical_outlier_removed_cloud);
}

void afterSeg_statistical_outlier_removal(const pcl::PCLPointCloud2ConstPtr& cloud, pcl::PCLPointCloud2::Ptr statistical_outlier_removed_cloud){
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

//----------------------------------------------------------------------------------
// Create Cluster-Mask Point Cloud to visualize each cluster separately
//----------------------------------------------------------------------------------
void color_euclidean_clusters_and_get_3D_bounding_boxes(const pcl::PCLPointCloud2ConstPtr& segmented_objects_cloud,
  const std::vector<pcl::PointIndices> &cluster_indices, const pcl::ModelCoefficients::Ptr plane_coefficients,
   pcl::PCLPointCloud2::Ptr colored_masked_cluster_cloud_ros, std::vector<object_detection::Object> &objects){

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

  objects.clear();

  pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2(*segmented_objects_cloud, *white_cloud);

  std::cout << "Detected " << cluster_indices.size() << " objects" << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_masked_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ExtractIndices<pcl::PCLPointCloud2> extract;

  extract.setInputCloud(segmented_objects_cloud);

   // // Fill in the cloud data
  for(int i=0; i<cluster_indices.size(); ++i){
    pcl::PointIndices::Ptr indexes(new pcl::PointIndices);
    *indexes = cluster_indices[i];

    //----------------------------------------------------------------------------------
    // BOUNDING BOX DETECTION
    extract.setIndices(indexes);
    pcl::PCLPointCloud2::Ptr object_cloud (new pcl::PCLPointCloud2);
    extract.filter(*object_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_object_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
    pcl::fromPCLPointCloud2(*object_cloud, *white_object_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_out (new pcl::PointCloud<pcl::PointXYZRGB>);
    shape_msgs::SolidPrimitive shape;
    geometry_msgs::Pose obj_pose;
    object_detection::FitBox(*white_object_cloud, plane_coefficients, *extract_out, shape, obj_pose);
//
    object_detection::Object obj;
    obj.label = std::to_string(i);
    obj.confidence = 0.0;
    obj.object_cloud = object_cloud;
    obj.pose = obj_pose;
    obj.dimensions.x = shape.dimensions[0];
    obj.dimensions.y = shape.dimensions[1];
    obj.dimensions.z = shape.dimensions[2];
    objects.push_back(obj);
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

  colored_masked_cluster->width = colored_masked_cluster->points.size();
  colored_masked_cluster->height = 1;
  colored_masked_cluster->is_dense = true;

  pcl::toPCLPointCloud2(*colored_masked_cluster, *colored_masked_cluster_cloud_ros);
  colored_masked_cluster_cloud_ros->header.frame_id = "front_base_footprint";
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

pcl::PCLPointCloud2::Ptr passthrough_filter_x_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_y_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr passthrough_filter_z_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr voxel_filter_downsampled_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr beforeSeg_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr afterSeg_objects_outlier_removed_cloud (new pcl::PCLPointCloud2);
pcl::PCLPointCloud2::Ptr filtered_cloud (new pcl::PCLPointCloud2);

pcl::PCLPointCloud2::Ptr segmented_objects_cloud  (new pcl::PCLPointCloud2) ;
pcl::ModelCoefficients::Ptr plane_coefficients (new pcl::ModelCoefficients);

std::vector<pcl::PointIndices> cluster_indices;
pcl::PCLPointCloud2::Ptr colored_masked_cluster_cloud_ros (new pcl::PCLPointCloud2);

std::vector<object_detection::Object> objects;

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& input_cloud)
{
  if (!(*input_cloud).is_dense){ // if true, then there are no NaNs, No need to convert to pcl::PointCloud<pcl::PointXYZRGB>
    ROS_INFO("Cloud not dense, process for NaNs");
  }

  // std::cout << "point cloud frame id: " << cloud->header.frame_id << std::endl;
  cluster_indices.clear();
  objects.clear();


  passthrough_filter_x( input_cloud , passthrough_filter_x_cloud);
  passthrough_filter_y( passthrough_filter_x_cloud , passthrough_filter_y_cloud);
  passthrough_filter_z( passthrough_filter_y_cloud , passthrough_filter_z_cloud);

  voxel_filter_downsampling( passthrough_filter_z_cloud , voxel_filter_downsampled_cloud);

  filtered_cloud = voxel_filter_downsampled_cloud;

  beforeSeg_statistical_outlier_removal( voxel_filter_downsampled_cloud , beforeSeg_outlier_removed_cloud);
  ransac_plane_segmentation( beforeSeg_outlier_removed_cloud , segmented_objects_cloud, plane_coefficients);
  afterSeg_statistical_outlier_removal( segmented_objects_cloud , afterSeg_objects_outlier_removed_cloud);

  euclidean_clustering(afterSeg_objects_outlier_removed_cloud, cluster_indices);
  color_euclidean_clusters_and_get_3D_bounding_boxes(afterSeg_objects_outlier_removed_cloud, cluster_indices, plane_coefficients, colored_masked_cluster_cloud_ros, objects);

  pb_filtered_cloud.publish ( filtered_cloud );
  pb_beforeSeg_statistical_outlier.publish ( beforeSeg_outlier_removed_cloud );
  pb_segmented_cloud.publish ( segmented_objects_cloud );
  pb_afterSeg_statistical_outlier.publish ( afterSeg_objects_outlier_removed_cloud );
  pb_color_clustered.publish ( colored_masked_cluster_cloud_ros );

  publish_and_visualize_bounding_boxes(objects);
}


void dyanamic_reconfigure_callback(object_detection::pcl_parametersConfig &config, uint32_t level) {
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

  ros::Subscriber sub = nh.subscribe<pcl::PCLPointCloud2>("/front_base_footprint/points", 1, cloud_cb);
  ros::spin();
}
