/*
 * @Author: Diantao Tu
 * @Date: 2022-07-18 15:50:39
 */

#include "Structure.h"

std::vector<PointTrack> TriangulateTracks(const std::vector<Frame>& frames, const std::vector<MatchPair>& image_pairs)
{
    // 存储最终的3D点轨迹结果
    vector<PointTrack> structure;
    
    // === 步骤1：提取图像对和匹配关系 ===
    // 从MatchPair中分离出纯粹的图像对索引和特征匹配关系
    // 这样做是为了适配TrackBuilder的接口要求
    vector<pair<size_t, size_t>> pairs;           // 存储图像对索引 (image1_id, image2_id)
    vector<vector<cv::DMatch>> pair_matches;      // 存储对应的特征匹配关系
    
    for(const MatchPair& p : image_pairs)
    {
        pairs.push_back(p.image_pair);          // 提取图像对
        pair_matches.push_back(p.matches);      // 提取特征匹配
    }
    
    // === 步骤2：构建特征点轨迹(Tracks) ===
    // Track的概念：同一个3D点在不同图像中的对应特征点形成一条轨迹
    // 例如：一个3D点P在图像A、B、C中分别对应特征点a、b、c，则(P, {a,b,c})构成一条track
    TrackBuilder tracks_builder;
    
    // 从图像对匹配关系中构建轨迹
    // 算法会自动将传递性匹配连接起来：如果A-B匹配，B-C匹配，则A-B-C构成一条轨迹
    tracks_builder.Build(pairs, pair_matches);
    
    // 过滤掉观测数量少于3的轨迹
    // 原因：至少需要3个观测才能进行稳定的三角化
    // 观测数量太少的轨迹通常质量不高，可能是误匹配
    tracks_builder.Filter(3);
    
    // === 步骤3：导出轨迹数据结构 ===
    // 数据结构说明：{track_id, {(image_id, feature_id), (image_id, feature_id), ...}}
    // track_id: 轨迹的唯一标识符
    // image_id: 图像索引
    // feature_id: 该图像中对应特征点的索引
    map<uint32_t, set<pair<uint32_t, uint32_t>>> tracks;
    tracks_builder.ExportTracks(tracks);
    
    LOG(INFO) << "Prepare to triangulate " << tracks.size() << " tracks";
    
    // 获取最大的轨迹ID，用于并行处理时的循环范围
    size_t max_track_id = tracks_builder.GetMaxID();
    
    // 检查是否成功构建了轨迹
    if(tracks.empty())
    {
        LOG(ERROR) << "Fail to estimate initial structure";
        return structure;
    }
    
    // === 内存优化：释放不再需要的数据 ===
    pairs.clear();
    pair_matches.clear();

    // === 步骤4：准备三角化所需的投影模型 ===
    // 创建等距柱状投影对象，用于全景图像的坐标转换
    // 将2D图像坐标转换为3D单位球面坐标
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    
    // === 步骤5：并行三角化所有轨迹 ===
    #pragma omp parallel for
    for(size_t track_idx = 0; track_idx < max_track_id; track_idx++)
    {
        // 检查当前索引是否对应有效的轨迹
        // 由于轨迹ID可能不连续，需要检查是否存在
        if(!tracks.count(track_idx))
            continue;
        
        // 获取当前轨迹的迭代器
        map<uint32_t, set<pair<uint32_t, uint32_t>>>::iterator it_track = tracks.find(track_idx);
        
        // === 准备三角化的数据 ===
        // 存储所有观测该轨迹的相机位姿和观测方向
        eigen_vector<Eigen::Matrix3d> R_cw_list;  // 相机到世界的旋转矩阵列表
        eigen_vector<Eigen::Vector3d> t_cw_list;  // 相机到世界的平移向量列表
        vector<cv::Point3f> points;               // 观测射线方向（单位球面坐标）
        
        // 遍历当前轨迹中的所有特征点观测
        for(const pair<uint32_t, uint32_t>& feature_pair : (*it_track).second)
        {
            const uint32_t image_idx = feature_pair.first;   // 图像索引
            const uint32_t feature_idx = feature_pair.second; // 特征点索引
            
            // 获取相机位姿矩阵并转换为相机到世界的变换
            // frames中存储的是世界到相机的变换T_wc，需要求逆得到T_cw
            const Eigen::Matrix4d T_cw = frames[image_idx].GetPose().inverse();
            
            // 提取旋转和平移部分
            R_cw_list.push_back(T_cw.block<3,3>(0,0));  // 3x3旋转矩阵
            t_cw_list.push_back(T_cw.block<3,1>(0,3));  // 3x1平移向量
            
            // 将图像像素坐标转换为单位球面上的3D坐标
            // 这给出了从相机中心指向特征点的单位方向向量
            points.push_back(eq.ImageToCam(frames[image_idx].GetKeyPoints()[feature_idx].pt));
        }
        
        // === 执行多视图三角化 ===
        // 使用多个相机位姿和对应的观测射线进行三角化
        // 算法会求解最小化重投影误差的3D点位置
        Eigen::Vector3d point_world = TriangulateNView(R_cw_list, t_cw_list, points);
        
        // === 检查三角化结果的有效性 ===
        // 过滤掉无限远点或无效的三角化结果
        // 这种情况通常发生在相机位姿不准确或观测射线几乎平行时
        if(isinf(point_world.x()) || isinf(point_world.y()) || isinf(point_world.z()))
            continue;
        
        // === 线程安全地添加结果 ===
        #pragma omp critical
        {
            // 创建PointTrack对象，包含轨迹ID、特征点观测和3D坐标
            structure.push_back(PointTrack(it_track->first, it_track->second, point_world));
        }
    }
    
    // === 步骤6：质量过滤 ===
    // 过滤掉重投影角度误差超过25度的3D点
    // 角度误差过大通常表明三角化不准确或存在误匹配
    // 25度是一个经验阈值，在精度和数量之间取得平衡
    FilterTracksAngleResidual(frames, structure, 25);
    
    // === 可选的距离过滤（当前被注释） ===
    // 可以进一步过滤掉距离相机过远的点，减少异常值
    // if(!RemoveFarPoints(5))
    //     return false;
    
    LOG(INFO) << "Successfully triangulate " << structure.size() << " tracks";
    
    return structure;

    // === 调试代码（当前被注释） ===
    // 用于可视化调试，显示每个track在各个图像上匹配的特征点
    // 以及三角化后的点重投影到图像上的位置
    // 这对于分析三角化质量和调试算法很有用
    // set<size_t> frame_ids = {162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172};
    // #pragma omp parallel for
    // for(size_t i = 0; i < structure.size(); i++)
    // {
    //     for(auto pair : structure[i].feature_pairs)
    //     {
    //         if(frame_ids.count(pair.first) > 0)
    //         {
    //             VisualizeTrack(structure[i], config.sfm_result_path);
    //             break;
    //         }
    //     }
    // }
}

size_t FilterTracksToFar(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    vector<PointTrack> valid_tracks;
    #pragma omp parallel for
    for(size_t i = 0; i < tracks.size(); i++)
    {
        // 找到当前三维点所对应的全部图像，然后找出其中相距最远的两个图像，把它们
        // 之间的距离当做 baseline （基线）
        set<size_t> frame_ids;
        for(const auto& pair : tracks[i].feature_pairs)
            frame_ids.insert(pair.first);
        eigen_vector<Eigen::Vector3d> frame_center;
        for(const size_t& id : frame_ids)
            if(frames[id].IsPoseValid())
                frame_center.push_back(frames[id].GetPose().block<3,1>(0,3));
        double baseline_distance = 0;
        FurthestPoints(frame_center, *(new int), *(new int), baseline_distance);
        // 计算当前三维点到各个图像之间的距离的平均值，如果这个距离超过了基线的 threshold 倍，就认为
        // 当前三维点不可靠，过滤掉
        double average_distance = 0;
        for(const Eigen::Vector3d& center : frame_center)
            average_distance += (center - tracks[i].point_3d).norm();
        average_distance /= frame_center.size();
        if(threshold * baseline_distance < average_distance)
            continue;
        #pragma omp critical
        {
            valid_tracks.push_back(tracks[i]);
        }
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}

size_t FilterTracksPixelResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    if(threshold < 0)
        return 0;
    double sq_threshold = Square(threshold);
    vector<PointTrack> valid_tracks;
    eigen_vector<Eigen::Matrix4d> T_cw_list;
    for(const Frame& frame : frames)
    {
        if(frame.IsPoseValid())
            T_cw_list.push_back(frame.GetPose().inverse());
        else 
            T_cw_list.push_back(Eigen::Matrix4d::Zero());
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(size_t i = 0; i < tracks.size(); i++)
    {
        bool valid = true;
        const Eigen::Vector3d pt_world = tracks[i].point_3d;
        for(const auto& feature : tracks[i].feature_pairs)
        {
            Eigen::Vector3d pt_cam = (T_cw_list[feature.first] * pt_world.homogeneous()).head(3);
            Eigen::Vector2d pt_proj = eq.CamToImage(pt_cam);
            const cv::Point2f& pt = frames[feature.first].GetKeyPoints()[feature.second].pt;
            double sq_dist = Square(pt.x - pt_proj.x()) + Square(pt.y - pt_proj.y());
            if(sq_dist > sq_threshold)
            {
                valid = false;
                break;
            }
        }
        if(valid)
            valid_tracks.push_back(tracks[i]);
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}

size_t FilterTracksAngleResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    // 计算夹角的阈值对应的余弦值，减少后面的计算量
    double cos_threshold = cos(threshold * M_PI / 180.0);
    vector<PointTrack> valid_tracks;
    eigen_vector<Eigen::Matrix4d> T_cw_list;
    for(const Frame& frame : frames)
    {
        if(frame.IsPoseValid())
            T_cw_list.push_back(frame.GetPose().inverse());
        else 
            T_cw_list.push_back(Eigen::Matrix4d::Zero());
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(size_t i = 0; i < tracks.size(); i++)
    {
        bool valid = true;
        const Eigen::Vector3d pt_world = tracks[i].point_3d;
        for(const auto& feature : tracks[i].feature_pairs)
        {
            Eigen::Vector3d pt_cam = (T_cw_list[feature.first] * pt_world.homogeneous()).head(3);
            cv::Point3f view_ray = eq.ImageToCam(frames[feature.first].GetKeyPoints()[feature.second].pt);
            double cos_angle = (pt_cam.x() * view_ray.x + pt_cam.y() * view_ray.y +  pt_cam.z() * view_ray.z)
                            / pt_cam.norm() / cv::norm(view_ray);
            if(cos_angle < cos_threshold)
            {
                valid = false;
                break;
            }
        }
        if(valid)
            valid_tracks.push_back(tracks[i]);
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}