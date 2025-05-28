/*
 * @Author: Diantao Tu
 * @Date: 2021-11-18 17:24:35
 */

#include "CameraLidarOptimizer.h"

using namespace std;

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4d _T_cl):T_cl_init(_T_cl)
{

}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4f _T_cl)       
{
    T_cl_init = _T_cl.cast<double>();
}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4d _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config):
                        T_cl_init(_T_cl), lidars(lidars), frames(frames), config(_config)
{}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4f _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config):
                        lidars(lidars), frames(frames), config(_config)
{
    T_cl_init = _T_cl.cast<double>();
}

// 这个函数用于优化相机和激光雷达之间的外参标定
// 输入参数:
// - line_pairs: 图像和激光雷达之间匹配的直线对,每个直线对包含一条图像直线和一条激光雷达直线
// - T_cl: 初始的相机到激光雷达的变换矩阵
int CameraLidarOptimizer::Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>&line_pairs, 
                        const Eigen::Matrix4d& T_cl)
{
    // 创建ceres优化问题
    ceres::Problem problem;
    // 使用Huber损失函数,阈值为2度
    ceres::LossFunction *  loss_function = new ceres::HuberLoss(2.0 * M_PI / 180.0);
    
    // 从变换矩阵中提取旋转矩阵和平移向量
    Eigen::Matrix3d R = T_cl.block<3,3>(0,0);
    Eigen::Vector3d t = T_cl.block<3,1>(0,3);
    // 将旋转矩阵转换为轴角表示
    Eigen::Vector3d angle_axis;
    ceres::RotationMatrixToAngleAxis(R.data(), angle_axis.data());
    
    // 创建全景相机模型
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    
    // 遍历所有匹配的直线对
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs.begin();
                it != line_pairs.end(); it++)
    {
        const vector<CameraLidarLinePair>& line_pair = it->second;
        for(const CameraLidarLinePair& pair : line_pair)
        {
            const cv::Vec4f& l = pair.image_line;
            // 将图像直线的端点投影到单位球面上
            cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), float(5.0));
            cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), float(5.0));
            cv::Point3f p3(0,0,0);
            
            // 计算过这三个点的平面法向量
            double a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
            double b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
            double c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );

            // 添加平面到平面的约束
            ceres::CostFunction *cost_function = Plane2Plane_Relative::Create(Eigen::Vector3d(a,b,c), pair.lidar_line_end, pair.lidar_line_start);
            problem.AddResidualBlock(cost_function, loss_function, angle_axis.data(), t.data());

            // 添加平面相对IOU的约束
            ceres::CostFunction *cost_function2 = PlaneRelativeIOUResidual::Create(
                        Eigen::Vector4d(a,b,c,0), (pair.lidar_line_start + pair.lidar_line_end) / 2.0, p1, p2, 2);
            problem.AddResidualBlock(cost_function2, nullptr, angle_axis.data(), t.data());
        }
    }

    LOG(INFO) << "total residual: " <<problem.NumResidualBlocks() << endl;
    
    // 配置求解器参数
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.max_linear_solver_iterations = 100;
    options.preconditioner_type = ceres::JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 10;

    // 求解优化问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.BriefReport() << '\n';
    
    // 将优化后的结果转换回变换矩阵
    ceres::AngleAxisToRotationMatrix(angle_axis.data(), R.data());
    T_cl_optimized = Eigen::Matrix4d::Identity();
    T_cl_optimized.block<3,3>(0,0) = R;
    T_cl_optimized.block<3,1>(0,3) = t;

    return 1;
}

// 这个函数用于从图像中提取直线特征
// 输入参数:
// - image_line_folder: 存储图像直线特征的文件夹路径
// - visualization: 是否需要可视化结果
bool CameraLidarOptimizer::ExtractImageLines(string image_line_folder, bool visualization)
{
    const size_t length = frames.size();
    // 先尝试从本地文件夹读取之前提取好的图像直线特征
    // 如果读取成功就直接返回,避免重复计算
    if(ReadPanoramaLines(image_line_folder, config.image_path, image_lines_all))
        return true;
    
    image_lines_all.reserve(length);
    LOG(INFO) << "Extract image lines for " << length << " images";
    if(length == 0)
        return false;
        
    // 读取深度图像文件名列表(目前未使用)
    vector<string> depth_image_names = IterateFiles(config.depth_path, ".bin");
    
    // 读取掩码图像(如果有的话)
    cv::Mat img_mask;
    if(!config.mask_path.empty())
        img_mask = cv::imread(config.mask_path, CV_LOAD_IMAGE_GRAYSCALE);
        
    // 显示处理进度条
    ProcessBar bar(length, 0.2);
    
    // 使用OpenMP多线程并行处理
    omp_set_num_threads(5);
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < length; i++)
    {
        // 获取灰度图像
        cv::Mat img_gray = frames[i].GetImageGray();
        // 创建全景图像直线检测器
        PanoramaLine detect(img_gray, i);
        detect.SetName(frames[i].name);
        
        // 根据是否有掩码图像选择不同的直线检测方式
        if(!img_mask.empty())
            detect.Detect(img_mask);
        else 
            detect.Detect(45, -45);  // 检测45度到-45度范围内的直线
            
        // 融合相近的直线
        detect.Fuse(config.ncc_threshold);
        
        // 如果需要可视化,则绘制检测到的直线
        if(visualization)
        {
            // 定义不同颜色用于绘制直线
            vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                        cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                        cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                        cv::Scalar(255,0,255)};                         // 紫
            // 在图像上绘制直线                            
            cv::Mat img_line = DrawLinesOnImage(img_gray, detect.GetLines(), colors, 6, true);
            // 根据优化模式选择保存路径
            if(optimization_mode == CALIBRATION)
                cv::imwrite(config.calib_result_path + "/img_line_filtered" + num2str(frames[i].id) + ".jpg", img_line);
            else 
                cv::imwrite(config.joint_result_path + "/img_line_filtered" + num2str(frames[i].id) + ".jpg", img_line);
        }
        
        // 临界区:将检测结果添加到全局变量中
        #pragma omp critical
        {
            image_lines_all.push_back(detect);
        }
        bar.Add();
    }
    
    // 对所有图像直线按ID排序
    sort(image_lines_all.begin(), image_lines_all.end(), [this](PanoramaLine& a, PanoramaLine& b){return a.id < b.id;});
    
    // 将提取的直线特征保存到本地,方便下次直接读取使用
    ExportPanoramaLines(image_line_folder, image_lines_all);
    return true;
}

// 这段代码实现了从激光雷达数据中提取直线特征的功能。主要步骤如下:

// ExtractLidarLines函数接收一个visualization参数用于控制是否可视化结果
bool CameraLidarOptimizer::ExtractLidarLines(bool visualization)
{
    // 获取激光雷达数据的数量
    const size_t length = lidars.size();

    LOG(INFO) << "Extract lidar lines for " << length << " lidar data";
    
    // 设置OpenMP并行线程数
    omp_set_num_threads(config.num_threads);
    
    // 使用OpenMP并行处理每一帧激光雷达数据
    #pragma omp parallel for
    for(size_t i = 0; i < length; i++)
    {
        // 如果当前帧没有特征点(cornerLessSharp为空),说明需要重新进行特征提取
        // 可能的原因:1.没有进行过特征提取 2.没有读取点云数据
        if(lidars[i].cornerLessSharp.empty())
        {
            // 重新加载激光雷达数据
            lidars[i].LoadLidar(lidars[i].name);
            // 重新排序VLP点云
            lidars[i].ReOrderVLP();
            // 提取特征点,包括:
            // - 曲率阈值(max_curvature)
            // - 交叉角度阈值(intersection_angle_threshold)
            // - 特征提取方法(extraction_method)
            // - 是否进行分割(lidar_segmentation)
            lidars[i].ExtractFeatures(config.max_curvature, config.intersection_angle_threshold, 
                                    config.extraction_method, config.lidar_segmentation);
        }

        // 如果需要可视化,保存特征点
        if(visualization)
            lidars[i].SaveFeatures(config.joint_result_path);
            
        // 如果点云在世界坐标系下,转换到局部坐标系
        if(lidars[i].IsInWorldCoordinate())
            lidars[i].Transform2Local();    
    }
    return true;
}

// 这个函数实现了相机和激光雷达的联合优化功能。主要包含两种模式:

// 1. 标定模式(CALIBRATION):
// - 用于标定相机和激光雷达之间的外参
// - 要求相机和激光雷达数据是同步的,数量相等
// - 迭代优化过程:
//   a. 使用当前外参进行直线特征关联
//   b. 优化外参
//   c. 计算外参变化量
//   d. 如果变化量小于阈值则结束,否则继续迭代
// - 可选择是否可视化结果

// 2. 建图模式(MAPPING):
// - 用于多相机多激光雷达的整体位姿优化
// - 主要步骤:
//   a. 保存初始的相机和激光雷达位姿用于可视化
//   b. 读取或重新计算图像特征点的三维结构
//   c. 迭代优化:
//     - 进行直线特征关联
//     - 优化位姿和结构
//     - 保存中间结果
//     - 检查收敛条件(cost变化<1%或连续两次迭代步数<5)
//   d. 可选择是否可视化最终结果

// 函数开始时会输出当前使用的配置信息,包括:
// - 是否使用角度残差
// - 是否使用点到线残差
// - 是否使用线到线残差
// - 是否使用点到面残差
// - 激光雷达平面容差
// - 是否归一化距离
// - 相机-相机权重
// - 激光雷达-激光雷达权重
// - 相机-激光雷达权重

bool CameraLidarOptimizer::JointOptimize(bool visualization)
{
    LOG(INFO) << "============== Camera-LiDAR joint optimization begin ======================";
    LOG(INFO) << "Configuration: use angle - " << (config.angle_residual ? "true\r\n" : "false\r\n") << 
            "\t\t use point to line - " << (config.point_to_line_residual ? "true\r\n" : "false\r\n") << 
            "\t\t use line to line - " << (config.line_to_line_residual ? "true\r\n" : "false\r\n") <<
            "\t\t use point to plane - " << (config.point_to_plane_residual ? "true\r\n" : "false\r\n") <<
            "\t\t lidar plane tolerance - " << config.lidar_plane_tolerance << "\r\n" << 
            "\t\t use normalized distance - " << (config.normalize_distance ? "true\r\n" : "false\r\n") << 
            "\t\t camera-camera weight - " << config.camera_weight << "\r\n" << 
            "\t\t LiDAR-LiDAR weight - " << config.lidar_weight << "\r\n" << 
            "\t\t camera-LiDAR weight - " << config.camera_lidar_weight;
            

    // 在进行直线关联的时候需要考虑一下之后要做什么，如果要进行标定任务（calibration），那么只需要
    // 对单帧进行直线关联，因为相当于雷达和相机是同步的，两者只有一个外参
    // 如果是要进行多相机多雷达的整体位姿优化，就要考虑一张图像和多个雷达进行关联，因为这时候相机和雷达
    // 不再同步，可以认为有n个外参（n=相机数 或 n=雷达数）
    if(optimization_mode == CALIBRATION)
    {
        LOG(INFO) << "start camera - LiDAR calibration";
        if(lidars.size() != frames.size())
        {
            const size_t size = min(lidars.size(), frames.size());
            LOG(WARNING) << "In calibration mode, num lidar != num frame, resize them to " << size;
            lidars = vector<Velodyne>(lidars.begin(), lidars.begin() + size);
            frames = vector<Frame>(frames.begin(), frames.begin() + size);
        }
        Eigen::Matrix4d T_cl_last = T_cl_init;
        eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all = AssociateLineSingle(T_cl_last);
        // 显示投影结果，用于debug
        if(visualization)
        {
            Visualize(line_pairs_all, config.calib_result_path + "/init/");
        }
        for(int iter = 0; iter < 35; iter++)
        {
            LOG(INFO) << "iteration: " << iter ;
            
            // 优化外参
            Optimize(line_pairs_all, T_cl_last);
            // 计算外参的变化量
            Eigen::Matrix3d rotation_last = T_cl_last.block<3,3>(0,0);
            Eigen::Vector3d trans_last = T_cl_last.block<3,1>(0,3);
            Eigen::Matrix3d rotation_curr = T_cl_optimized.block<3,3>(0,0);
            float rotation_change = acos(((rotation_last.transpose() * rotation_curr).trace() - 1) / 2.0);
            rotation_change *= 180.0 / M_PI;
            float trans_change = (trans_last - T_cl_optimized.block<3,1>(0,3)).norm();
            // 更新变化之后的外参
            T_cl_last = T_cl_optimized;
            // 特征关联
            line_pairs_all = AssociateLineSingle(T_cl_last);
            LOG(INFO) << "rotation change: " << rotation_change << "  translate change: " << trans_change << endl;
            if(rotation_change < 0.1 && trans_change < 0.01)
                break; 
        }
        // 显示投影结果，用于debug
        if(visualization)
        {
            Visualize(line_pairs_all, config.calib_result_path + "/final/");
        }
    }
    else if(optimization_mode == MAPPING)
    {
        LOG(INFO) << "start camera-LiDAR mapping";
        CameraCenterPCD(config.joint_result_path + "/camera_center_init.pcd", GetCameraTranslation());
        CameraCenterPCD(config.joint_result_path + "/lidar_center_init.pcd", GetLidarTranslation());
        CameraPoseVisualize(config.joint_result_path + "/camera_pose_init.ply", GetCameraRotation(), GetCameraTranslation());
        CameraPoseVisualize(config.joint_result_path + "/lidar_pose_init.ply", GetLidarRotation(), GetLidarTranslation());

        if(false)
        {
            // 对图像特征点重新进行三角化
            vector<MatchPair> image_pairs;
            ReadMatchPair(config.match_pair_joint_path, image_pairs, min(config.num_threads, 4));
            EstimateStructure(image_pairs);
        }
        else 
            ReadPointTracks(config.sfm_result_path + "points.bin", structure);
        

        double last_cost = 0, curr_cost = 0;
        int last_step = INT32_MAX, curr_step = INT32_MAX; 
        eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all = 
                    AssociateLineMulti(config.neighbor_size_joint, true, false);
        for(int iter = 0; iter < config.num_iteration_joint; iter++)
        {
            LOG(INFO) << "iteration: " << iter;
            Optimize(line_pairs_all, structure, true, true, true, true, true, curr_cost, curr_step);
            CameraCenterPCD(config.joint_result_path + "/camera_center_" + num2str(iter) + ".pcd", GetCameraTranslation());
            CameraCenterPCD(config.joint_result_path + "/lidar_center_" + num2str(iter) + ".pcd", GetLidarTranslation());
            CameraPoseVisualize(config.joint_result_path + "/lidar_pose_" + num2str(iter) + ".ply", GetLidarRotation(), GetLidarTranslation());
            CameraPoseVisualize(config.joint_result_path + "/camera_pose_" + num2str(iter) + ".ply", GetCameraRotation(), GetCameraTranslation());
            line_pairs_all.clear();
            line_pairs_all =  AssociateLineMulti(config.neighbor_size_joint, true, false);   
            if(abs(curr_cost - last_cost) / last_cost < 0.01)
            {
                LOG(INFO) << "early termination condition fulfilled. cost change is less than 1%";
                break;
            }
            if(curr_step < 5 && last_step < 5)
            {
                LOG(INFO) << "early termination condition fulfilled. iteration step is less than 5 steps in last two iterations";
                break;
            }
            last_cost = curr_cost;
            last_step = curr_step;
    
        }
        // 输出所有匹配的直线，用于debug
        if(visualization)
            Visualize(line_pairs_all, config.joint_result_path + "/visualization/");

    }
    else 
    {
        LOG(ERROR) << "mode not supported";
        return false;
    }
    LOG(INFO) << "============== Camera-LiDAR joint optimization end ======================";
    return true;
}

// {image id, lidar id} => {line pair}
// 这个函数用于将相机图像中的直线和激光雷达点云中的直线进行关联匹配
// 输入参数T_cl是相机到激光雷达的变换矩阵
// 返回值是一个map,键是pair<图像id,激光雷达id>,值是对应的直线匹配对
eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> CameraLidarOptimizer::AssociateLineSingle(Eigen::Matrix4d T_cl)
{
    // 设置OpenMP的线程数
    omp_set_num_threads(config.num_threads);
    // 用于存储所有的直线匹配对
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all;
    // 并行处理每一帧图像
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < frames.size(); i++)
    {
        // 创建直线匹配器对象,传入图像的尺寸和灰度图
        CameraLidarLineAssociate associate(frames[i].GetImageRows(), frames[i].GetImageCols(), frames[i].GetImageGray());
        // 如果激光雷达已经进行了边缘分割
        if(!lidars[i].edge_segmented.empty())
            // 使用角度信息进行直线匹配
            associate.AssociateByAngle(image_lines_all[i].GetLines(), lidars[i].edge_segmented, lidars[i].segment_coeffs, 
                                lidars[i].cornerLessSharp, lidars[i].point_to_segment, lidars[i].end_points, T_cl);
        else 
            // 否则直接进行直线匹配
            associate.Associate(image_lines_all[i].GetLines(), lidars[i].cornerLessSharp, T_cl);
        // 临界区,将匹配结果存入map
        #pragma omp critical 
        {
            line_pairs_all[pair<size_t, size_t>(i,i)] = associate.GetAssociatedPairs();;
        }
    }
    
    // 统计匹配的直线对数量
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs";

    return line_pairs_all;
}

// {image id, lidar id} => {line pair}
// 这个函数用于在多帧图像和激光雷达数据之间进行直线特征的关联匹配
// 输入参数:
// - neighbor_size: 每帧图像需要匹配的相邻激光雷达帧数
// - temporal: 是否考虑时序关系进行匹配
// - use_lidar_track: 是否使用激光雷达的跟踪信息
// - use_image_track: 是否使用图像的跟踪信息
// 返回值是一个map,键是pair<图像id,激光雷达id>,值是对应的直线匹配对
eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> CameraLidarOptimizer::AssociateLineMulti(
        const int neighbor_size, const bool temporal, const bool use_lidar_track, const bool use_image_track)
{
    // 设置OpenMP的线程数
    omp_set_num_threads(config.num_threads);
    LOG(INFO) << "Associate lines, neighbor size = " << neighbor_size;
    
    // 为每个图像帧找到用于匹配的相邻激光雷达帧
    vector<vector<int>> each_frame_neighbor = NeighborEachFrame(neighbor_size, temporal);
    
    // 如果使用跟踪信息,创建掩码数组
    vector<vector<bool>> image_mask_all(frames.size()), lidar_mask_all(lidars.size());
    if(use_lidar_track)
        lidar_mask_all = LidarMaskByTrack();
    if(use_image_track)
        image_mask_all = ImageMaskByTrack();
        
    // 用于存储所有的直线匹配对
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all;
    
    // 并行处理每一帧图像
    #pragma omp parallel for schedule(dynamic)
    for(size_t frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        // 遍历当前图像帧的所有相邻激光雷达帧
        for(const int& lidar_id : each_frame_neighbor[frame_id])
        {
            const Velodyne& lidar = lidars[lidar_id];
            // 计算相机到激光雷达的变换矩阵
            Eigen::Matrix4d T_cl = T_cl_init;
            if(frames[frame_id].IsPoseValid() && lidar.IsPoseValid())
            {
                Eigen::Matrix4d T_wc = frames[frame_id].GetPose();
                Eigen::Matrix4d T_wl = lidar.GetPose();
                T_cl = T_wc.inverse() * T_wl;
            }
            
            // 创建直线匹配器对象
            CameraLidarLineAssociate associate(frames[frame_id].GetImageRows(), frames[frame_id].GetImageCols());
            
            // 根据激光雷达是否已经进行直线拟合选择不同的匹配方式
            if(!lidar.edge_segmented.empty())
                // 使用角度信息进行直线匹配
                associate.AssociateByAngle(image_lines_all[frame_id].GetLines(), lidar.edge_segmented, lidar.segment_coeffs,
                             lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points, T_cl, true, 
                             image_mask_all[frame_id], lidar_mask_all[lidar_id]);
            else 
                // 直接进行直线匹配
                associate.Associate(image_lines_all[frame_id].GetLines(), lidar.cornerLessSharp, T_cl);
                
            // 获取匹配结果
            vector<CameraLidarLinePair> pairs = associate.GetAssociatedPairs();
            
            // 临界区,将匹配结果存入map
            #pragma omp critical
            {
                line_pairs_all[pair<size_t,size_t>(frame_id, lidar_id)] = pairs;
            }
        }
    }
    
    // 统计匹配的直线对数量
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs";
    
    return line_pairs_all;
}


// 这个函数实现了相机和激光雷达的联合优化。主要功能包括:

// 输入参数:
// - line_pairs: 相机和激光雷达之间匹配的直线对
// - structure: 三角化得到的3D点云结构
// - refine_camera_rotation/trans: 是否优化相机的旋转/平移
// - refine_lidar_rotation/trans: 是否优化激光雷达的旋转/平移  
// - refine_structure: 是否优化3D点云结构
// - cost: 输出优化后的代价值
// - steps: 输出优化迭代步数

int CameraLidarOptimizer::Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, 
                std::vector<PointTrack>& structure, const bool refine_camera_rotation,
                 const bool refine_camera_trans, const bool refine_lidar_rotation, 
                const bool refine_lidar_trans, const bool refine_structure,
                double& cost, int& steps)
{
    // 1. 取出所有相机和激光雷达的位姿,转换为轴角表示进行优化
    eigen_vector<Eigen::Vector3d> angleAxis_cw_list(frames.size()), angleAxis_lw_list(lidars.size());
    eigen_vector<Eigen::Vector3d> t_cw_list(frames.size()), t_lw_list(lidars.size());
    pcl::PointCloud<pcl::PointXYZI> lidar_center;
    
    // 提取相机位姿
    for(size_t i = 0; i < frames.size(); i++) {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_cw = frames[i].GetPose().inverse();
        Eigen::Matrix3d R_cw = T_cw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_cw.data(), angleAxis_cw_list[i].data());
        t_cw_list[i] = T_cw.block<3,1>(0,3);
    }
    
    // 提取激光雷达位姿
    for(size_t i = 0; i < lidars.size(); i++) {
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        Eigen::Matrix4d T_wl = lidars[i].GetPose();
        Eigen::Matrix4d T_lw = T_wl.inverse();
        Eigen::Matrix3d R_lw = T_lw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_lw.data(), angleAxis_lw_list[i].data());
        t_lw_list[i] = T_lw.block<3,1>(0,3);
        lidars[i].Transform2LidarWorld();
    }
    
    // 2. 构建优化问题
    ceres::Problem problem;
    ceres::LossFunction* loss_function1 = new ceres::HuberLoss(3 * M_PI / 180.0);  // 夹角阈值为3度
    
    // 3. 添加各类约束
    // 添加相机-激光雷达之间的约束
    size_t residual_camera_lidar = AddCameraLidarResidual(frames, lidars,
                    angleAxis_cw_list, t_cw_list, angleAxis_lw_list, t_lw_list, 
                    line_pairs, loss_function1, problem, config.camera_lidar_weight);
    LOG(INFO) << "num residual blocks for camera-lidar : " << residual_camera_lidar;

    // 添加相机-相机之间的重投影约束
    size_t residual_camera = AddCameraResidual(frames, angleAxis_cw_list, t_cw_list, structure, 
                        problem, RESIDUAL_TYPE::ANGLE_RESIDUAL_1, config.camera_weight);
    LOG(INFO) << "num residual blocks for camera-camera: " << residual_camera;

    // 添加激光雷达-激光雷达之间的约束
    vector<vector<int>> neighbors = FindNeighbors(lidars, 6);
    size_t residual_lidar = 0;
    if(config.point_to_line_residual)
        residual_lidar += AddLidarPointToLineResidual(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance, config.lidar_weight);
    if(config.line_to_line_residual) {
        LidarLineMatch matcher(lidars);
        matcher.SetNeighborSize(4);
        matcher.SetMinTrackLength(3);
        matcher.GenerateTracks();
        residual_lidar += AddLidarLineToLineResidual2(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, matcher.GetTracks(),
                                    config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance);
    }
    if(config.point_to_plane_residual)
        residual_lidar += AddLidarPointToPlaneResidual(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                config.point_to_plane_dis_threshold, config.lidar_plane_tolerance, config.angle_residual, 
                                config.normalize_distance, config.lidar_weight);
    LOG(INFO) << "num residual blocks for lidar-lidar: " << residual_lidar;

    // 4. 根据参数设置固定部分变量
    // 固定3D点云结构
    if(refine_structure == false) {
        for(const PointTrack& track : structure)
            problem.SetParameterBlockConstant(track.point_3d.data());
    }
    
    // 固定相机位姿
    for(size_t i = 0 ; i < frames.size(); i++) {
        if(frames[i].IsPoseValid()) {
            if(refine_camera_rotation == false)
                problem.SetParameterBlockConstant(angleAxis_cw_list[i].data());
            if(refine_camera_trans == false)
                problem.SetParameterBlockConstant(t_cw_list[i].data());
        }
    }
    
    // 固定激光雷达位姿
    for(size_t i = 0 ; i < lidars.size(); i++) {
        if(lidars[i].IsPoseValid() && lidars[i].valid) {
            if(refine_lidar_rotation == false)
                problem.SetParameterBlockConstant(angleAxis_lw_list[i].data());
            if(refine_lidar_trans == false)
                problem.SetParameterBlockConstant(t_lw_list[i].data());
        }
    }    

    // 固定第一帧相机位姿作为参考坐标系
    problem.SetParameterBlockConstant(angleAxis_cw_list[0].data());
    problem.SetParameterBlockConstant(t_cw_list[0].data());

    // 5. 求解优化问题
    LOG(INFO) << "total residual blocks : " << problem.NumResidualBlocks();
    ceres::Solver::Options options = SetOptionsSfM(config.num_threads);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 6. 检查优化结果
    if(!summary.IsSolutionUsable()) {
        LOG(INFO) << summary.FullReport();
        // 保存失败时的位姿用于调试
        ofstream f(config.joint_result_path + "/camera_pose_fail.txt");
        for(size_t i = 0; i < angleAxis_cw_list.size(); i++)
            f << angleAxis_cw_list[i].x() << " " << angleAxis_cw_list[i].y() << " " << angleAxis_cw_list[i].z() << " " 
              << t_cw_list[i].x() << " " << t_cw_list[i].y() << " " << t_cw_list[i].z() << endl;
        f.close();
        f.open(config.joint_result_path + + "/lidar_pose_fail.txt");
        for(size_t i = 0; i < angleAxis_lw_list.size(); i++)
            f << angleAxis_lw_list[i].x() << " " << angleAxis_lw_list[i].y() << " " << angleAxis_lw_list[i].z() << " " 
              << t_lw_list[i].x() << " " << t_lw_list[i].y() << " " << t_lw_list[i].z() << endl;
        f.close();
        LOG(ERROR) << "Camera LiDAR optimization failed";
        return false;
    }
    
    // 7. 更新优化后的位姿
    LOG(INFO) << summary.BriefReport();
    // 更新相机位姿
    for(size_t i = 0; i < frames.size(); i++) {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix3d R_cw;
        ceres::AngleAxisToRotationMatrix(angleAxis_cw_list[i].data(), R_cw.data());
        Eigen::Vector3d t_cw = t_cw_list[i];
        Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
        T_cw.block<3,3>(0,0) = R_cw;
        T_cw.block<3,1>(0,3) = t_cw;
        frames[i].SetPose(T_cw.inverse());
    }

    // 更新激光雷达位姿
    #pragma omp parallel for 
    for(size_t i = 0; i < lidars.size(); i++) {
        if(!lidars[i].IsPoseValid())
            continue;
        // 如果雷达在世界坐标系下，需要先转换到局部坐标系再更新位姿
        if(lidars[i].IsInWorldCoordinate())
            lidars[i].Transform2Local();
        Eigen::Matrix3d R_lw;
        ceres::AngleAxisToRotationMatrix(angleAxis_lw_list[i].data(), R_lw.data());
        Eigen::Vector3d t_lw = t_lw_list[i];
        Eigen::Matrix4d T_lw = Eigen::Matrix4d::Identity();
        T_lw.block<3,3>(0,0) = R_lw;
        T_lw.block<3,1>(0,3) = t_lw;
        lidars[i].SetPose(T_lw.inverse());
    }
    
    // 8. 返回优化结果
    cost = summary.final_cost;
    steps = summary.num_successful_steps;
    return 1;
}


std::vector<std::vector<int>> CameraLidarOptimizer::NeighborEachFrame(const int neighbor_size, const bool temporal)
{
    vector<vector<int>> each_frame_neighbor(frames.size());
    if(temporal)
    {
        for(int frame_id = 0; frame_id < frames.size(); frame_id++)
        {
            // 确定LiDAR的起始和结尾，注意结尾是不包含在近邻范围内的。这里的方法就是先确定LiDAR的起始位置，
            // 根据起始位置和neighbor size确定结尾。然后把结尾限制在lidrs.size()内，接着根据结尾反过来确定起始
            // 这样就能保证图像都有相同数量的近邻了，只有一种是例外，也就是neighbor size > lidars.size
            int lidar_id_start = max(0, frame_id - (neighbor_size / 2));
            int lidar_id_end = min(static_cast<int>(lidars.size()), lidar_id_start + neighbor_size);
            lidar_id_start = max(0, lidar_id_end - neighbor_size);
            for(int lidar_id = lidar_id_start; lidar_id < lidar_id_end; lidar_id++)
                each_frame_neighbor[frame_id].push_back(lidar_id);
        }
    }
    else 
    {
        pcl::PointCloud<PointType> lidar_center;
        for(size_t i = 0; i < lidars.size(); i++)
        {
            if(!lidars[i].IsPoseValid() || !lidars[i].valid)
                continue;
            PointType center;
            Eigen::Vector3d t_wl = lidars[i].GetPose().block<3,1>(0,3);
            center.x = t_wl.x();
            center.y = t_wl.y();
            center.z = t_wl.z();
            // 设置intensity只是为了知道当前的点对应于哪一帧雷达，因为可能有的雷达没有位姿就没被记录下来
            center.intensity = i;   
            lidar_center.push_back(center);
        }
        pcl::KdTreeFLANN<PointType>::Ptr kd_center(new pcl::KdTreeFLANN<PointType>());
        kd_center->setInputCloud(lidar_center.makeShared());
        vector<int> neighbors;
        for(int i = 0; i < frames.size(); i++)
        {
            if(!frames[i].IsPoseValid())
                continue;
            PointType center;
            Eigen::Vector3d t_wl = frames[i].GetPose().block<3,1>(0,3);
            center.x = t_wl.x();
            center.y = t_wl.y();
            center.z = t_wl.z();
            kd_center->nearestKSearch(center, neighbor_size, neighbors, *(new vector<float>()));
            for(int& n_idx : neighbors)
            {
                n_idx = lidar_center[n_idx].intensity;
            }
            set<int> neighbors_set(neighbors.begin(), neighbors.end());
            if(neighbors_set.count(i - 1) == 0 && (i - 1 >= 0))
                neighbors.push_back(i - 1);
            if(neighbors_set.count(i + 1) == 0 && (i + 1 < lidars.size()))
                neighbors.push_back(i + 1);
            each_frame_neighbor[i] = neighbors;
        }
    }
    return each_frame_neighbor;
}

std::vector<std::vector<bool>> CameraLidarOptimizer::LidarMaskByTrack(const int min_track_length, const int neighbor_size)
{
    #pragma omp parallel for
    for(Velodyne& lidar : lidars)
    {
        if(!lidar.IsInWorldCoordinate())
            lidar.Transform2LidarWorld();
    }
    LidarLineMatch lidar_line_matcher(lidars);
    lidar_line_matcher.SetMinTrackLength(min_track_length);
    lidar_line_matcher.SetNeighborSize(neighbor_size);
    lidar_line_matcher.GenerateTracks();

    // 默认所有的LiDAR直线都是被掩模的状态，也就是对应位置为false
    vector<vector<bool>> lidar_mask_all;
    for(size_t i = 0; i < lidars.size(); i++)
        lidar_mask_all.push_back(vector<bool>(lidars[i].edge_segmented.size(), false));
    for(const LineTrack& track : lidar_line_matcher.GetTracks())
    {
        // feature是 {lidar id, line id}
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            lidar_mask_all[feature.first][feature.second] = true;
    }
    #pragma omp parallel for
    for(Velodyne& lidar : lidars)
    {
        if(lidar.IsInWorldCoordinate())
            lidar.Transform2Local();
    }
    return lidar_mask_all;
}

std::vector<std::vector<bool>> CameraLidarOptimizer::ImageMaskByTrack(const int min_track_length, const int neighbor_size)
{
    eigen_vector<Eigen::Matrix3d> R_wc_list;
    eigen_vector<Eigen::Vector3d> t_wc_list;
    for(const Frame& frame : frames)
    {
        R_wc_list.push_back(frame.GetPose().block<3,3>(0,0));
        t_wc_list.push_back(frame.GetPose().block<3,1>(0,3));
    }
    PanoramaLineMatcher image_line_matcher(image_lines_all, R_wc_list, t_wc_list);
    image_line_matcher.SetMinTrackLength(min_track_length);
    image_line_matcher.SetNeighborSize(neighbor_size);
    image_line_matcher.GenerateTracks(BASIC);
    // 除去包含平行直线的track
    image_line_matcher.RemoveParallelLines();
    vector<LineTrack> image_tracks = image_line_matcher.GetTracks();
    for(size_t i = 0; i < image_tracks.size(); i++)
        image_tracks[i].id = i;

    vector<vector<bool>> image_mask_all;
    for(size_t i = 0; i < image_lines_all.size(); i++)
        image_mask_all.push_back(vector<bool>(image_lines_all[i].GetLines().size(), false));
    for(const LineTrack& track : image_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            image_mask_all[feature.first][feature.second] = true;
    }
    return image_mask_all;
}

bool CameraLidarOptimizer::SetLineWeight(vector<CameraLidarLinePair>& line_pair, const size_t frame_idx, const size_t lidar_idx)
{
    
    if(!frames[frame_idx].IsPoseValid())
        return false;
    // 使用当期帧和之后的帧的相对运动作为当前的前进方向
    size_t next_valid_id = frame_idx + 1;
    while(next_valid_id < frames.size() && !frames[next_valid_id].IsPoseValid())
        next_valid_id++;
    if(next_valid_id == frames.size())
        return false;
    Eigen::Matrix4d T_12 = frames[frame_idx].GetPose().inverse() * frames[next_valid_id].GetPose();
    Eigen::Vector3d moving_direction = T_12.block<3,1>(0,3);
    const vector<cv::Vec4f>& lines = image_lines_all[frame_idx].GetLines();
    vector<double> weight;
    Equirectangular eq(frames[frame_idx].GetImageRows(), frames[frame_idx].GetImageCols());
    // 把直线变成单位球面上的点
    for(size_t i = 0; i < line_pair.size(); i++)
    {
        const cv::Vec4f& l = line_pair[i].image_line;
        cv::Point3f start = eq.ImageToCam(cv::Point2f(l[0], l[1]), 1.f);
        cv::Point3f end = eq.ImageToCam(cv::Point2f(l[2],l[3]), 1.f);
        Eigen::Vector3d line_3d(end.x - start.x, end.y - start.y, end.z - start.z);
        // 计算直线和前进方向的夹角，这里是借用了平面夹角的计算，因为两者是一样的，而且都要返回锐角
        double angle = PlaneAngle(moving_direction.data(), line_3d.data());
        // 权重用e的负指数计算，也就是 e^(-angle), angle=0时权重最大为1， angle越大，权重越小
        weight.push_back(exp(-angle));
        line_pair[i].weight = exp(-angle);
    }
    // 可视化各个直线的权重
    // {
    //     cv::Mat img_gray = frames[frame_idx].GetImageGray();
    //     cv::cvtColor(img_gray, img_gray, CV_GRAY2BGR);
    //     double min_weight = exp(- M_PI_2);
    //     double weight_length = 1.0 - min_weight;
    //     for(size_t i = 0; i < lines.size(); i++)
    //     {
    //         uchar relative_weight = static_cast<uchar>((weight[i] - min_weight) / weight_length * 255.0);
    //         cv::Vec3b color = Gray2Color(relative_weight);
    //         DrawLine(img_gray, lines[i], color, 5, true);
    //     }
    //     cv::imwrite("./weighted_lines_" + num2str(int(frames[frame_idx].id)) + ".jpg", img_gray);
    // }
    return true;
}

bool CameraLidarOptimizer::EstimateStructure(const std::vector<MatchPair>& image_pairs)
{
    LOG(INFO) << "==================== Estimate Initial Structure start =================";
    structure = TriangulateTracks(frames, image_pairs);
    if(!FilterTracksToFar(frames, structure, 8))
        return false;
    LOG(INFO) << "Successfully triangulate " << structure.size() << " tracks";
    LOG(INFO) << "==================== Estimate Initial Structure end =================";
    return true;
}


bool CameraLidarOptimizer::GlobalBundleAdjustment(std::vector<PointTrack>& structure ,bool refine_structure, bool refine_rotation, bool refine_translation)
{
    if(!SfMGlobalBA(frames, structure, RESIDUAL_TYPE::ANGLE_RESIDUAL_1, 
                    config.num_threads, refine_structure, refine_rotation, refine_translation))
    {
        LOG(ERROR) << "Global BA failed";
        return false;
    }
    return true;
}

void CameraLidarOptimizer::Visualize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs_all,
                    const std::string path, int line_width, int point_size)
{
    if(!boost::filesystem::exists(path))
        boost::filesystem::create_directories(path);
    LOG(INFO) << "save joint visualization result in " << path;
    #pragma omp parallel
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
    {
        #pragma omp single nowait
        {
            const size_t image_idx = it->first.first;
            const size_t lidar_idx = it->first.second;
            
            Eigen::Matrix4d T_cl = T_cl_init;
            if(frames[image_idx].IsPoseValid() && lidars[image_idx].IsPoseValid())
                T_cl = frames[image_idx].GetPose().inverse() * lidars[lidar_idx].GetPose();

            cv::Mat img_gray = frames[image_idx].GetImageGray();
            cv::Mat img_line = DrawLinePairsOnImage(img_gray, it->second, T_cl, line_width);                
            cv::imwrite(path + "/line_pair_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_line);
            
            cv::Mat img_cloud = ProjectLidar2PanoramaRGB(lidars[lidar_idx].cloud, img_gray,
                        T_cl, config.min_depth, config.max_depth_visual, point_size);
            cv::imwrite(path + "/cloud_project_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_cloud);
        
            cv::Mat img_corner = ProjectLidar2PanoramaRGB(lidars[lidar_idx].cornerLessSharp, img_gray,
                        T_cl, config.min_depth, config.max_depth_visual, point_size);
            cv::imwrite(path + "/corner_project_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_corner);
        }
    }
}

pcl::PointCloud<PointType> CameraLidarOptimizer::FuseLidar(int skip, double min_range, double max_range)
{
    pcl::PointCloud<PointType> cloud_fused;
    double sq_min_range = min_range * min_range, sq_max_range = max_range * max_range;
    for(size_t i = 0; i < lidars.size(); i += (skip + 1))
    {
        if(!lidars[i].valid || !lidars[i].IsPoseValid())
            continue;
        if(lidars[i].cloud.empty())
            lidars[i].LoadLidar(lidars[i].name);
        pcl::PointCloud<pcl::PointXYZI> cloud_filtered;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_raw = lidars[i].cloud_scan.makeShared();
        if(cloud_raw->empty())
            cloud_raw = lidars[i].cloud.makeShared();
        for(const pcl::PointXYZI& pt : cloud_raw->points)
        {
            double range = pt.x * pt.x + pt.y * pt.z + pt.z * pt.z;
            if(range > sq_max_range || range < sq_min_range)
                continue;
            cloud_filtered.push_back(pt);
        }
        pcl::transformPointCloud(cloud_filtered, cloud_filtered, lidars[i].GetPose());
        cloud_fused += cloud_filtered;
    }
    return cloud_fused;
}

Eigen::Matrix4d CameraLidarOptimizer::GetResult()
{
    return T_cl_optimized;
}

void CameraLidarOptimizer::SetOptimizationMode(int mode)
{
    optimization_mode = mode;
}

eigen_vector<Eigen::Matrix3d> CameraLidarOptimizer::GetCameraRotation(bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> global_rotation;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = f.GetPose();
        global_rotation.push_back(T_wc.block<3,3>(0,0));
    }
    return global_rotation;
}

eigen_vector<Eigen::Vector3d> CameraLidarOptimizer::GetCameraTranslation(bool with_invalid)
{
    eigen_vector<Eigen::Vector3d> global_translation;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = f.GetPose();
        global_translation.push_back(T_wc.block<3,1>(0,3));
    }
    return global_translation;
}

std::vector<std::string> CameraLidarOptimizer::GetImageNames(bool with_invalid)
{
    vector<string> names;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        names.push_back(f.name);
    }
    return names;
}

eigen_vector<Eigen::Matrix3d> CameraLidarOptimizer::GetLidarRotation(bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> global_rotation;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = l.GetPose();
        global_rotation.push_back(T_wc.block<3,3>(0,0));
    }
    return global_rotation;
}

eigen_vector<Eigen::Vector3d> CameraLidarOptimizer::GetLidarTranslation(bool with_invalid)
{
    eigen_vector<Eigen::Vector3d> global_translation;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = l.GetPose();
        global_translation.push_back(T_wc.block<3,1>(0,3));
    }
    return global_translation;
}

std::vector<std::string> CameraLidarOptimizer::GetLidarNames(bool with_invalid)
{
    vector<string> names;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        names.push_back(l.name);
    }
    return names;
}

const std::vector<Frame>& CameraLidarOptimizer::GetFrames()
{
    return frames;
}

const std::vector<Velodyne>& CameraLidarOptimizer::GetLidars()
{
    return lidars;
}

void CameraLidarOptimizer::SetFrames(const std::vector<Frame>& _frames)
{
    frames = _frames;
}

void CameraLidarOptimizer::SetLidars(const std::vector<Velodyne>& _lidars)
{
    lidars = _lidars;
}

bool CameraLidarOptimizer::ExportStructureBinary(const std::string file_name)
{
    if(structure.empty())
        return false;
    return ExportPointTracks(file_name, structure);
}