/*
 * @Author: Diantao Tu
 * @Date: 2021-11-22 16:44:10
 */

#include "SfM.h"
#include "../base/Geometry.hpp"

using namespace std;

SfM::SfM(const Config& _config):config(_config)
{
    track_triangulated = false;
}

bool SfM::ReadImages(const vector<string>& image_names, const cv::Mat& mask)
{
    // 输出日志信息，标记图像读取和特征提取流程的开始
    LOG(INFO) << "=============== Read image and extract features begin ==============" << endl;
    
    // 设置OpenMP并行线程数为配置文件中指定的线程数
    // 这样可以充分利用多核CPU进行并行计算，加速特征提取过程
    omp_set_num_threads(config.num_threads);
    
    // 清空frames容器，确保从干净的状态开始处理
    frames.clear();
    
    // 读取第一张图像来获取图像尺寸信息
    // 这里假设所有图像的分辨率都是相同的，这在SLAM/SfM应用中是常见的假设
    cv::Mat img = cv::imread(image_names[0]);
    
    // 创建进度条对象，用于显示处理进度
    // 参数：总任务数为图像数量，进度更新间隔为0.1（即每完成10%显示一次）
    ProcessBar bar(image_names.size(), 0.1);
    
    // 使用OpenMP并行处理所有图像
    // schedule(dynamic)：动态调度，适合处理时间不均匀的任务
    // 因为不同图像的特征点数量可能差异很大，动态调度能更好地平衡负载
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < image_names.size(); i++)
    {
        // 为每张图像创建Frame对象
        // 参数：图像高度、宽度、图像ID、图像文件路径
        Frame frame(img.rows, img.cols, i, image_names[i]);
        
        // 加载图像的灰度版本到内存
        // 大多数特征检测算法（如SIFT）都在灰度图像上工作
        frame.LoadImageGray(frame.name);
        
        // 提取图像的关键点（特征点）
        // config.num_sift：要提取的SIFT特征点数量上限
        // mask：掩码图像，指定在图像的哪些区域提取特征点
        //       mask为空的区域不会提取特征点，可用于排除天空、车辆等动态区域
        frame.ExtractKeyPoints(config.num_sift, mask);
        
        // 为每个关键点计算描述子（特征描述符）
        // config.root_sift：是否使用RootSIFT，这是SIFT的改进版本
        // 描述子用于后续的特征匹配，是一个高维向量，描述了特征点周围的图像信息
        frame.ComputeDescriptor(config.root_sift);
        
        // 释放灰度图像的内存，节省内存空间
        // 因为后续处理不再需要原始图像数据，只需要提取出的特征点和描述子
        frame.ReleaseImageGray();
        
        // OpenMP临界区：确保多线程安全地访问共享资源
        // 因为frames是所有线程共享的容器，需要互斥访问避免数据竞争
        #pragma omp critical 
        {
            // 将处理完的Frame对象加入到frames容器中
            frames.push_back(frame);
            
            // 更新进度条，显示当前处理进度
            bar.Add();
        }
    }
    
    // 对frames按照id进行排序
    // 由于OpenMP的并行执行，frames中的元素顺序可能被打乱
    // 排序确保frames中的元素按照原始图像顺序排列，这对后续处理很重要
    sort(frames.begin(), frames.end(), [this](Frame& a, Frame& b){
        return a.id < b.id;  // 按照Frame的id从小到大排序
    });
    
    // 输出处理结果的统计信息
    LOG(INFO) << "Read " << frames.size() << " images";
    
    // 输出日志信息，标记图像读取和特征提取流程的结束
    LOG(INFO) << "=============== Read image and extract features end ==============" << endl;
    
    // 返回是否成功处理了图像
    // 如果frames.size() > 0说明至少处理了一张图像，返回true；否则返回false
    return frames.size() > 0;
}

bool SfM::LoadFrameBinary(const std::string& image_path, const std::string& frame_path, const bool skip_descriptor)
{
    return ReadFrames(frame_path, image_path, frames, config.num_threads, skip_descriptor);
}

bool SfM::InitImagePairs(const int frame_match_type)
{
    // 清空现有的图像匹配对，从干净状态开始
    image_pairs.clear();
    
    // === 策略1：穷举匹配（EXHAUSTIVE） ===
    // 如果匹配类型包含穷举匹配标志位
    if(frame_match_type & FrameMatchMethod::EXHAUSTIVE)
    {
        LOG(INFO) << "init match pairs with exhausive match";
        // 穷举所有可能的图像对组合
        // 对于N张图像，会生成N*(N-1)/2个匹配对
        // 这种方法最完整但计算量巨大，通常只在图像数量较少时使用
        for(int i = 0; i < frames.size(); i++)
            for(int j = i + 1; j < frames.size(); j++)
                image_pairs.push_back(MatchPair(i,j));  // 创建图像i和图像j的匹配对
        return image_pairs.size() > 0;
    }
    
    // === 组合策略处理 ===
    // 图像对的初始生成方式是可以组合的，比如可以同时使用VLAD和连续匹配
    // 为了避免出现重复的匹配对，使用set来记录已有的匹配对
    // set会自动去重，确保每个图像对只出现一次
    set<pair<size_t,size_t>> pairs;
    
    // === 策略2：连续匹配（CONTIGUOUS） ===
    // 如果匹配类型包含连续匹配标志位
    if(frame_match_type & FrameMatchMethod::CONTIGUOUS)
    {
        // 设置邻域大小，每张图像只与其后续的20张图像进行匹配
        int neighbor_size = 20;
        LOG(INFO) << "init match pairs with contiguous match, neighbor size = " << neighbor_size;
        
        // 基于图像序列的连续性进行匹配
        // 假设相邻的图像更可能有重叠区域和共同特征
        // 这种策略适用于按时间顺序或空间顺序排列的图像序列
        for(int i = 0; i < frames.size(); i++)
            for(int j = i + 1; j < i + neighbor_size && j < frames.size(); j++)
            {
                image_pairs.push_back(MatchPair(i,j));
                pairs.insert({i, j});  // 记录这个匹配对，避免后续重复添加
            }
    }
    
    // === 策略3：VLAD匹配 ===
    // 如果匹配类型包含VLAD匹配标志位
    if(frame_match_type & FrameMatchMethod::VLAD)
    {
        // 动态计算邻域大小：总图像数的1/40，但不少于15
        // 这样可以根据数据集大小自适应调整匹配策略
        int neighbor_size = max(int(frames.size() / 40), 15);
        LOG(INFO) << "init match pairs with VLAD, neighbor size = " << neighbor_size;
        
        // 创建VLAD匹配器
        // VLAD (Vector of Locally Aggregated Descriptors) 是一种图像检索方法
        // 它可以将图像表示为紧凑的特征向量，用于快速的图像相似性比较
        VLADMatcher vlad(frames, config, RESIDUAL_NORMALIZATION_PWR_LAW);
        
        // 生成词典（CodeBook）
        // 参数0.5可能是词典大小的比例或其他相关参数
        vlad.GenerateCodeBook(0.5);
        
        // 为每张图像计算VLAD嵌入向量
        // 将每张图像的特征转换为固定维度的VLAD描述符
        vlad.ComputeVLADEmbedding();
        
        // 为每张图像找到最相似的邻居图像
        // 返回每张图像的neighbor_size个最相似图像的索引
        std::vector<std::vector<size_t>> neighbors_all = vlad.FindNeighbors(neighbor_size);
        
        // 遍历每张图像及其邻居，生成匹配对
        for(size_t i = 0; i < frames.size(); i++)
        {
            for(const size_t& neighbor : neighbors_all[i])
            {
                // 跳过自己与自己的匹配
                if(neighbor == i)
                    continue;
                
                // 确保匹配对的id按从小到大排序，避免重复
                // 例如：(3,5)和(5,3)实际上是同一个匹配对
                size_t min_id = min(i, neighbor);
                size_t max_id = max(i, neighbor);
                
                // 检查这个匹配对是否已经存在
                if(pairs.count({min_id, max_id}) == 0)
                {
                    image_pairs.push_back(MatchPair(min_id, max_id));
                    pairs.insert({min_id, max_id});
                }
            }
        }
    }
    
    // === 策略4：GPS匹配 ===
    // 如果匹配类型包含GPS匹配标志位
    if(frame_match_type & FrameMatchMethod::GPS)
    {
        int neighbor_size = 15;           // 每个位置最多匹配15个邻居
        float distance_threshold = 7;     // GPS距离阈值7米
        LOG(INFO) << "init match pairs with GPS, neighbor size = " << neighbor_size << ", distance threshold = " << distance_threshold;
        
        // 尝试加载GPS数据
        if(LoadGPS(config.gps_path))
        {
            // 将每张图像的GPS位置记录为点云格式，便于进行空间搜索
            pcl::PointCloud<pcl::PointXYZI> frame_center;
            for(size_t i = 0; i < frames.size(); i++)
            {
                pcl::PointXYZI pt(i);  // intensity字段存储图像索引
                // 将GPS坐标转换为PCL点格式
                EigenVec2PclPoint(frames[i].GetGPS(), pt);
                frame_center.push_back(pt);
            }
            
            // 构建KD树用于快速近邻搜索
            // KD树是一种高效的空间数据结构，可以快速找到最近邻点
            pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_center(new pcl::KdTreeFLANN<pcl::PointXYZI>());
            kd_center->setInputCloud(frame_center.makeShared());
            
            // 为每个GPS位置寻找空间上的邻居
            for(const pcl::PointXYZI& pt : frame_center)
            {
                vector<float> sq_neighbor_dist;  // 存储邻居的平方距离
                vector<int> neighbor_ids;        // 存储邻居的索引
                
                // 搜索最近的neighbor_size+1个点（包括自己）
                kd_center->nearestKSearch(pt, neighbor_size + 1, neighbor_ids, sq_neighbor_dist);
                
                // 遍历找到的邻居（跳过第一个，因为是自己）
                for(size_t i = 1; i < neighbor_ids.size(); i++)
                {
                    // 如果距离超过阈值，停止处理（因为结果是按距离排序的）
                    if(sq_neighbor_dist[i] > Square(distance_threshold))
                        break;
                    
                    // 创建匹配对，确保id按从小到大排序
                    pair<size_t,size_t> curr_pair(min(pt.intensity, frame_center[neighbor_ids[i]].intensity), 
                                                  max(pt.intensity, frame_center[neighbor_ids[i]].intensity));
                    
                    // 检查是否已存在这个匹配对
                    if(pairs.count(curr_pair) > 0)
                        continue;
                    
                    pairs.insert(curr_pair);
                    image_pairs.push_back(MatchPair((size_t)pt.intensity, (size_t)frame_center[neighbor_ids[i]].intensity));
                }
            }
        }
        else 
            LOG(ERROR) << "Unable to init match pairs with GPS";
    }
    
    // === 策略5：GPS+VLAD组合匹配 ===
    // 如果匹配类型包含GPS_VLAD匹配标志位
    if(frame_match_type & FrameMatchMethod::GPS_VLAD)
    {
        int neighbor_size = frames.size() / 40;  // 邻居数量与总图像数成比例
        float distance_threshold = 20;           // GPS距离阈值20米（比纯GPS匹配更宽松）
        LOG(INFO) << "init match pairs with VLAD and filter with GPS, neighbor size = " << neighbor_size << ", distance threshold = " << distance_threshold;;
        
        // 确保GPS数据可用
        if(LoadGPS(config.gps_path))
        {
            // 首先使用VLAD找到视觉上相似的图像
            VLADMatcher vlad(frames, config, RESIDUAL_NORMALIZATION_PWR_LAW);
            vlad.GenerateCodeBook(0.5);
            vlad.ComputeVLADEmbedding();
            std::vector<std::vector<size_t>> neighbors_all = vlad.FindNeighbors(neighbor_size);
            
            // 遍历VLAD找到的邻居，但用GPS距离进行过滤
            for(size_t i = 0; i < frames.size(); i++)
            {
                for(const size_t& neighbor : neighbors_all[i])
                {
                    if(neighbor == i)
                        continue;
                    
                    size_t min_id = min(i, neighbor);
                    size_t max_id = max(i, neighbor);
                    
                    // 计算两张图像GPS位置之间的欧几里得距离
                    double gps_distance = (frames[min_id].GetGPS() - frames[max_id].GetGPS()).norm();
                    
                    // 如果GPS距离大于阈值，过滤掉这个匹配对
                    // 这样可以排除视觉相似但空间位置相距很远的图像（如重复场景）
                    if(gps_distance > distance_threshold)
                        continue;
                    
                    // 添加通过GPS距离验证的匹配对
                    if(pairs.count({min_id, max_id}) == 0)
                    {
                        image_pairs.push_back(MatchPair(min_id, max_id));
                        pairs.insert({min_id, max_id});
                    }
                }
            }
        }
    }
    
    // 返回是否成功生成了图像匹配对
    return image_pairs.size() > 0;
}

bool SfM::ComputeDepthImage(const Eigen::Matrix4d T_cl)
{
    // 输出日志信息，标记深度图计算流程的开始
    LOG(INFO) << "=================== Compute Depth Image begin =====================";
    
    // 检查激光雷达数据和图像帧的数量是否匹配
    // 在相机-激光雷达融合系统中，通常要求每个图像帧都有对应的激光雷达数据
    // 这样才能为每张图像生成对应的深度图
    if(lidars.size() != frames.size())
    {
        LOG(ERROR) << "warning: lidar size != frame size" << endl;
        return false;  // 数量不匹配则返回失败
    }
    
    // 判断是否需要保存深度图到本地文件
    // 如果config.depth_path不为空，说明需要保存深度图
    bool save_depth = !config.depth_path.empty();
    
    // 如果需要保存深度图且目标目录不存在，则创建目录
    if(save_depth && !boost::filesystem::exists(config.depth_path))
    {
        boost::filesystem::create_directory(config.depth_path);
        LOG(INFO) << "save depth image in " << config.depth_path;
    }

    // 创建深度图可视化结果的保存路径
    // 可视化图像通常是将深度信息用伪彩色表示，便于人眼观察
    string visualize_path = config.sfm_result_path + "/depth_visualize";
    if(!boost::filesystem::exists(visualize_path))
        boost::filesystem::create_directory(visualize_path);

    // 设置是否使用半分辨率来生成深度图
    // 使用半分辨率可以大大节约内存和计算时间，通常对最终效果影响不大
    bool half_size = true;
    
    // 设置OpenMP并行线程数
    omp_set_num_threads(config.num_threads);
    
    // 创建进度条，用于显示深度图计算进度
    ProcessBar bar(lidars.size(), 0.1);
    
    // 使用OpenMP并行处理每个激光雷达数据
    // schedule(dynamic)：动态调度，适合处理时间不均匀的任务
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 从原始雷达数据中复制一份用于当前线程处理
        // 这样做的好处是：读取完点云数据后可以立即释放内存，避免同时加载所有雷达数据
        Velodyne lidar(lidars[i]);
        
        // 从文件中加载当前雷达的点云数据到内存
        // 雷达点云通常包含XYZ坐标和强度信息
        lidar.LoadLidar(lidar.name);
        
        // 声明深度图变量
        cv::Mat depth;
        
        // 根据是否使用半分辨率来生成深度图
        if(half_size)
        {
            // 使用半分辨率：将图像的行列数各减半（+1是为了处理奇数尺寸）
            // ProjectLidar2PanoramaDepth函数将3D雷达点云投影到2D全景深度图
            // 参数说明：
            // - lidar.cloud: 输入的雷达点云
            // - (frames[i].GetImageRows() + 1) / 2: 目标深度图的行数（半分辨率）
            // - (frames[i].GetImageCols() + 1) / 2: 目标深度图的列数（半分辨率）
            // - T_cl: 相机到激光雷达的外参变换矩阵
            // - 4: 可能是距离阈值或其他参数
            depth = ProjectLidar2PanoramaDepth(lidar.cloud, (frames[i].GetImageRows() + 1) / 2, 
                            (frames[i].GetImageCols() + 1) / 2, T_cl, 4);
        }
        else 
        {
            // 使用全分辨率生成深度图
            depth = ProjectLidar2PanoramaDepth(lidar.cloud, frames[i].GetImageRows(), 
                            frames[i].GetImageCols(), T_cl, 4);
        }
        
        // 深度补全：填充由于雷达点云稀疏而产生的深度图空洞
        // 雷达点云通常比较稀疏，直接投影得到的深度图会有很多空洞
        // DepthCompletion函数使用插值等方法填充这些空洞
        // config.max_depth: 最大有效深度值，超过此值的深度被认为无效
        // 函数返回CV_32F格式的真实深度值（单位通常是米）
        depth = DepthCompletion(depth, config.max_depth);
        
        // 如果需要保存深度图到本地
        if(save_depth)
        {
            // === 生成可视化图像 ===
            // 获取对应图像帧的彩色图像
            cv::Mat depth_with_color = frames[i].GetImageColor();
            
            // 如果使用半分辨率，需要将彩色图像也缩放到相同尺寸
            if(half_size)
                cv::pyrDown(depth_with_color, depth_with_color);  // 图像金字塔下采样
            
            // 将深度信息与RGB图像结合，生成可视化图像
            // 通常是将深度信息用伪彩色叠加在RGB图像上
            // config.max_depth_visual: 可视化时的最大深度值
            depth_with_color = CombineDepthWithRGB(depth, depth_with_color, config.max_depth_visual);
            
            // 保存可视化结果为JPEG图像
            cv::imwrite(visualize_path + "/depth_" + num2str(i) + ".jpg", depth_with_color);
            
            // === 保存原始深度图 ===
            // 将深度值乘以256，为转换为16位整数做准备
            // 这样可以在保持一定精度的同时大大节约存储空间
            depth *= 256.f;
            
            // 转换为16位无符号整数格式
            // CV_32F转换为CV_16U虽然会损失一些精度，但存储空间减少一半
            depth.convertTo(depth, CV_16U);
            
            // 将深度图以二进制格式保存到本地文件
            // 二进制格式比图像格式保存更精确，加载也更快
            ExportOpenCVMat(config.depth_path + num2str(i) + ".bin", depth);
        }
        
        // 更新进度条
        bar.Add();
    }
    
    // 输出日志信息，标记深度图计算流程的结束
    LOG(INFO) << "=================== Compute Depth Image end ===================";
    return true;  // 成功完成所有深度图的计算
}


bool SfM::MatchImagePairs(const int matches_threshold)
{
    // 输出日志信息，标记图像对匹配流程的开始
    LOG(INFO) << "========= Match Image Pairs begin ===========" << endl;
    
    // 创建容器存储通过匹配筛选的图像对
    vector<MatchPair> good_pair;
    
    // 记录被图像对覆盖的帧数量，主要用于统计和日志输出
    // 这个变量帮助了解有多少帧参与了匹配，评估匹配的覆盖度
    set<size_t> covered_frames;
    
    // 设置OpenMP并行线程数
    omp_set_num_threads(config.num_threads);

    // === CUDA加速相关设置 ===
#ifdef USE_CUDA
    // 检查是否可以使用CUDA加速
    // 需要同时满足：配置允许使用CUDA 且 系统有可用的CUDA设备
    bool use_cuda = config.use_cuda && (cv::cuda::getCudaEnabledDeviceCount() > 0);
    
    // 创建GPU内存中的描述符数组，用于存储所有图像的SIFT描述符
    vector<cv::cuda::GpuMat> d_descriptors(frames.size());
    
    // 如果使用CUDA，将所有图像的描述符从CPU内存上传到GPU内存
    if(use_cuda)
    {
        // 并行上传所有图像的描述符到GPU
        // 这样可以避免在匹配过程中重复进行CPU-GPU数据传输
        #pragma omp parallel for
        for(size_t i = 0; i < frames.size(); i++)
            d_descriptors[i].upload(frames[i].GetDescriptor());
    }
#else 
    // 如果编译时没有CUDA支持，则不使用CUDA
    bool use_cuda = false;
#endif
    
    // === SIFT特征匹配过程 ===
    // 输出匹配策略信息
    LOG(INFO) << "match image pair with SIFT, " << (use_cuda ? "use cuda" : "use cpu"); 
    LOG(INFO) << "init pair : " << image_pairs.size() << endl;
    
    // 创建进度条，显示匹配进度
    ProcessBar bar1(image_pairs.size(), 0.1);
    
    // 使用OpenMP并行处理所有图像对
    // schedule(dynamic)：动态调度，适合处理时间不均匀的任务
    // 因为不同图像对的特征点数量差异很大，匹配时间也会有显著差异
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < image_pairs.size(); i++)
    {
        // 更新进度条
        bar1.Add(1);
        
        // 获取当前图像对的源图像和目标图像索引
        size_t source = image_pairs[i].image_pair.first;
        size_t target = image_pairs[i].image_pair.second;
        
        // 存储SIFT匹配结果
        vector<cv::DMatch> matches;
        
        // 根据是否使用CUDA选择不同的匹配方法
#ifdef USE_CUDA
        if(use_cuda)
            // 使用GPU加速的SIFT匹配
            // 在GPU内存中直接进行描述符匹配，速度更快
            matches = MatchSIFT(d_descriptors[source], d_descriptors[target], config.sift_match_dist_threshold);
        else 
#endif
            // 使用CPU进行SIFT匹配
            // 从CPU内存中读取描述符进行匹配
            matches = MatchSIFT(frames[source].GetDescriptor(), frames[target].GetDescriptor(), config.sift_match_dist_threshold);
        
        // === 第一轮过滤：基于匹配数量 ===
        // 如果匹配点数量少于阈值，直接跳过这个图像对
        // 匹配点太少通常意味着两张图像重叠区域很小或者没有共同场景
        if(matches.size() < matches_threshold)
            continue;
        
        // === 第二轮过滤：基于匹配质量 ===
        // 找到所有匹配中距离最大的那个匹配
        // 这里distance越小表示匹配越好，所以最大distance对应最差的匹配
        vector<cv::DMatch>::iterator it_max = max_element(matches.begin(), matches.end());
        
        // 创建容器存储高质量的匹配
        vector<cv::DMatch> good_matches;
        
        // 应用比例测试（Ratio Test）进行匹配筛选
        // 只保留距离小于最大距离80%的匹配
        // 这是Lowe在SIFT论文中提出的经典筛选方法，可以有效去除模糊匹配
        for(vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end(); it++)
            if(it->distance < 0.8 * it_max->distance)
                good_matches.push_back(*it);
        
        // 再次检查筛选后的匹配数量是否足够
        if(int(good_matches.size()) < matches_threshold)
            continue;
        
        // === 保存通过筛选的图像对 ===
        // 使用临界区保护，确保多线程安全地访问共享数据
        #pragma omp critical
        {
            // 将通过筛选的图像对和其匹配结果保存
            good_pair.push_back(MatchPair(source, target, good_matches));
            
            // 记录参与匹配的图像帧
            covered_frames.insert(source);
            covered_frames.insert(target);
            
            // === 可选的匹配可视化 ===
            // 下面的代码被注释掉了，如果需要可以取消注释来生成匹配可视化图像
            // 这对调试和分析匹配质量很有帮助
            // cv::Mat matches_vertical = DrawMatchesVertical(frames[source].GetImageColor(), frames[source].GetKeyPoints(),
            //                         frames[target].GetImageColor(), frames[target].GetKeyPoints(), good_matches);
            // cv::imwrite(config.sfm_result_path + "/sift_" + num2str(source) + "_" + num2str(target) + ".jpg", matches_vertical);
        }
    }
    
    // === 更新图像对列表 ===
    // 将筛选后的图像对替换原来的图像对列表
    // swap操作比赋值更高效，直接交换两个容器的内容
    good_pair.swap(image_pairs);
    
    // === 输出统计信息 ===
    LOG(INFO) << "after filter with SIFT : " << image_pairs.size() << " image pairs, with " <<
             covered_frames.size() << " frames" << endl;
    
    // === 保存结果 ===
    // 将筛选后的图像对结果保存到文本文件
    // 这个文件可以用于后续流程的快速加载，避免重复计算
    ExportMatchPairTXT(config.sfm_result_path + "/after_sift_match.txt");
    
    // === 内存清理 ===
    // 释放所有图像的描述符内存
    // 描述符通常占用大量内存，匹配完成后可以释放以节省内存
    for(size_t i = 0; i < frames.size(); i++)
        frames[i].ReleaseDescriptor();
    
    // 输出日志信息，标记图像对匹配流程的结束
    LOG(INFO) << "========= Match Image Pairs end ===========" << endl;
    
    return true;  // 返回成功
}


bool SfM::FilterImagePairs(const int triangulation_num_threshold , const float triangulation_angle_threshold, const bool keep_no_scale )
{
    LOG(INFO) << "=========== Filter Image Pairs begin ================";
    
    // 设置OpenMP并行线程数
    omp_set_num_threads(config.num_threads);
    
    // 记录被覆盖的图像帧，用于统计
    set<size_t> covered_frames;
    
    // 记录初始图像对数量，用于后续统计过滤效果
    size_t num_pairs_init = image_pairs.size(); 
    
    // === 统计变量：记录各种过滤原因 ===
    // 这些统计数据有助于分析SfM性能瓶颈和调试算法
    size_t estimate_essential_fail = 0 ,            // 无法估计本质矩阵的图像对数量
            decompose_essential_fail = 0,           // 无法分解本质矩阵得到好的相对位姿的数量
            refine_pose_fail = 0,                   // 优化相对位姿时失败的数量
            no_scale = 0,                           // 无法计算尺度的数量
            no_connection = 0;                      // 不满足边双联通 bi-edge-connection 的数量

    // === 坐标系转换：从图像坐标到球面坐标 ===
    // 创建等距柱状投影对象，用于全景图像的坐标转换
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    
    // 将所有图像的2D特征点投影到单位球面上
    // 这是全景SLAM中的标准做法，因为全景图像覆盖360度视野
    vector<vector<cv::Point3f>> key_points_sphere(frames.size());
    for(size_t i = 0; i < frames.size(); i++)
    {
        const vector<cv::KeyPoint>& key_points = frames[i].GetKeyPoints();
        for(auto& kp : key_points)
            // 将图像像素坐标转换为单位球面上的3D坐标
            key_points_sphere[i].push_back(eq.ImageToCam(kp.pt));
    }
    
    // 存储通过几何验证的图像对
    vector<MatchPair> good_pair;
    
    // === 并行处理每个图像对 ===
    #pragma omp parallel for schedule(dynamic)
    for(const MatchPair& p : image_pairs)
    {
        // 获取当前图像对的索引
        size_t idx1 = p.image_pair.first;
        size_t idx2 = p.image_pair.second;
        
        // 获取两张图像在球面上的特征点
        const vector<cv::Point3f>& keypoints1 = key_points_sphere[idx1];
        const vector<cv::Point3f>& keypoints2 = key_points_sphere[idx2];
        
        // 存储每次迭代的结果
        vector<MatchPair> pair_each_iter;
        // 存储每个结果的内点数量和索引，用于排序选择最佳结果
        vector<pair<int,int>> inlier_each_pair;
        
        // === 多次迭代寻找最佳相对位姿 ===
        // 由于RANSAC的随机性，多次迭代可以提高找到最优解的概率
        for(int iter = 0; iter < 40; iter++)
        {
            // === 步骤1：估计本质矩阵 ===
            Eigen::Matrix3d essential;
            vector<size_t> inlier_idx;
            
            // 设置重投影误差上限为5度
            // 这个阈值考虑了全景图像的特点和特征点检测的精度
            double error_upper_bound = 5.0 * M_PI / 180.0;
            
            // 创建AC-RANSAC的NFA（Number of False Alarms）评估器
            // 参数：样本数量、模型最小样本数(8点法)、是否使用log scale
            ACRansac_NFA nfa(p.matches.size(), 8, false);
            
            // 使用AC-RANSAC算法估计本质矩阵
            // AC-RANSAC相比标准RANSAC更加稳健，能自动确定内点阈值
            essential = FindEssentialACRANSAC(p.matches, keypoints1, keypoints2, 300, 
                        error_upper_bound, nfa, inlier_idx);
            
            // 如果无法估计出有效的本质矩阵，跳过这次迭代
            if(essential.isZero())
                continue;
            
            // === 步骤2：分解本质矩阵 ===
            // 本质矩阵分解可以得到4组可能的相对位姿(R, t)
            eigen_vector<Eigen::Matrix3d> rotations;
            eigen_vector<Eigen::Vector3d> trans;
            DecomposeEssential(essential, rotations, trans);

            // === 步骤3：通过三角化验证正确的相对位姿 ===
            // 为4组候选位姿分别进行评估
            vector<double> parallax(rotations.size());                              // 视差角度
            vector<int> num_pts(rotations.size(), 0);                              // 成功三角化的点数
            vector<eigen_vector<Eigen::Vector3d>> triangulated_points(rotations.size()); // 三角化得到的3D点
            vector<vector<size_t>> inliers(rotations.size());                      // 每组位姿的内点索引
            
            // 创建内点标记数组，只使用计算本质矩阵时的内点进行三角化
            // 这样可以避免外点对位姿验证的干扰
            vector<bool> inlier(p.matches.size(), false);
            for(const size_t& idx : inlier_idx)
                inlier[idx] = true;
            
            // 对每组候选位姿进行三角化测试
            for(size_t j = 0; j < rotations.size(); j++)
                num_pts[j] = CheckRT(rotations[j], trans[j], inlier, p.matches, 
                            keypoints1, keypoints2, parallax[j], triangulated_points[j], inliers[j]);
            
            // === 步骤4：选择最佳位姿 ===
            // 找到能三角化出最多有效3D点的位姿
            vector<int>::iterator max_points_idx = max_element(num_pts.begin(), num_pts.end());
            int best_idx = max_points_idx - num_pts.begin();
            
            // 如果三角化的点数量太少，说明这个图像对质量不好，跳过
            if(*max_points_idx < triangulation_num_threshold)
            {
                continue;
            }
            
            // === 可选的视差检查（当前被注释） ===
            // 视差太小的图像对容易导致三角化不稳定
            // if(parallax[best_idx] < 1)
            // {
            //     little_parallax++;
            //     continue;
            // }
            
            // 检查是否有多个位姿都能三角化出大量点
            // 如果有多个位姿都表现很好，说明这个图像对的几何约束不够强
            int num_similar = count_if(num_pts.begin(), num_pts.end(), 
                            [max_points_idx](unsigned int num_points){return num_points > 0.8 * (*max_points_idx);});
            
            // 有多个位姿都能三角化出大量三维点，过滤掉这个图像对
            if(num_similar > 1)
            {
                continue;
            }

            // === 保存有效的相对位姿结果 ===
            // 注意：这里没有保存匹配的特征点对，因为对于每次迭代来说匹配点都是相同的
            MatchPair match_pair(p.image_pair.first, p.image_pair.second);
            match_pair.R_21 = rotations[best_idx];                    // 相对旋转矩阵
            match_pair.t_21 = trans[best_idx];                        // 相对平移向量
            match_pair.triangulated = triangulated_points[best_idx];  // 三角化得到的3D点
            match_pair.inlier_idx = inliers[best_idx];               // 内点索引

            pair_each_iter.push_back(match_pair);
            // 记录内点数量和结果索引，用于后续排序
            inlier_each_pair.push_back({match_pair.inlier_idx.size(), inlier_each_pair.size()});            
        }
        
        // 如果所有迭代都失败了，记录失败原因并跳过
        if(pair_each_iter.empty())
        {
            estimate_essential_fail++;
            continue;
        }

        #if 0
        // === 调试代码：输出每次迭代的结果 ===
        // 可以用于分析算法性能和调试
        ofstream f(config.sfm_result_path +  num2str(p.image_pair.first) + "_" + num2str(p.image_pair.second) + ".txt");
        for(const MatchPair& pair : pair_each_iter)
        {
            Eigen::Vector3d t_12 = -pair.R_21.transpose() * pair.t_21;
            f << "inliner : " <<  pair.inlier_idx.size() << endl;
            f << t_12.x() << " " << t_12.y() << " " << t_12.z() << endl;
            Eigen::AngleAxisd angleAxis(pair.R_21.transpose());
            f << angleAxis.axis().x() << " " << angleAxis.axis().y() << " " << angleAxis.axis().z() << " " << angleAxis.angle() * 180.0 / M_PI << endl;
        }
        #endif 

        // === 选择最终的最佳结果 ===
        // 按照内点数量从大到小排序
        // first - 内点数量， second - 在pair_each_iter中的索引
        sort(inlier_each_pair.begin(), inlier_each_pair.end(), 
            [](const pair<int,int>& a, const pair<int,int>& b){return a.first > b.first;});

        // 选择内点数量最多的结果
        int best_idx = inlier_each_pair[0].second;
        
        #if 0
        // === 可选的旋转角度约束（当前被注释） ===
        // 限制相邻帧之间旋转的角度，适用于车载等连续运动场景
        // 最大角度 = 帧数差异 * 1.5度，限制连续两帧之间旋转角度不能超过1.5度
        double rotation_angle_threshold = (p.image_pair.second - p.image_pair.first) * 1.5 / 180.0 * M_PI;
        for(const auto& idx : inlier_each_pair)
        {
            best_idx = idx.second;
            Eigen::AngleAxisd angleAxis(pair_each_iter[best_idx].R_21);
            if(angleAxis.angle() <= rotation_angle_threshold)
                break;
            best_idx = -1;
        }
        if(best_idx < 0)
        {
            decompose_essential_fail ++;
            continue;
        }
        #endif 
        
        // 将原始匹配点信息添加到最佳结果中
        pair_each_iter[best_idx].matches = p.matches;
        
        // === 步骤5：精化相对位姿 ===
        // 使用非线性优化进一步提高相对位姿的精度
        if(!RefineRelativePose(pair_each_iter[best_idx]))
        {
            refine_pose_fail ++;
            // continue;  // 即使精化失败也保留结果（当前设置）
        }
        
        // 线程安全地添加到结果列表
        #pragma omp critical
        {
            good_pair.push_back(pair_each_iter[best_idx]);
        }
    }

    // 更新图像对列表为通过几何验证的结果
    image_pairs = good_pair;
    
    // === 步骤6：设置相对平移的尺度 ===
    LOG(INFO) << "start to set relative translation scale";
    // 使用深度图信息为相对平移设置真实的物理尺度
    // 没有深度信息的相对位姿只能确定方向，无法确定距离
    SetTranslationScaleDepthMap(keep_no_scale);
    no_scale = good_pair.size() - image_pairs.size();
    
    size_t tmp = image_pairs.size();
    
    // === 步骤7：图连通性过滤 ===
    // 根据边双连通性过滤图像对，确保重建图的连通性
    // 只保留最大的双连通子图，去除孤立的图像对
    image_pairs = LargestBiconnectedGraph(image_pairs, covered_frames);
    no_connection = tmp - image_pairs.size();
    
    // === 输出统计信息 ===
    LOG(INFO) << "count of image pairs with valid relative motion : " << image_pairs.size() << " image pairs, with " <<
             covered_frames.size() << " frames" << endl;
    LOG(INFO) << "filter " << num_pairs_init - image_pairs.size() << " image pairs" << 
                "\r\n\t\t filter by compute essential fail : " << estimate_essential_fail << 
                "\r\n\t\t filter by decompose essential fail : " << decompose_essential_fail << 
                "\r\n\t\t filter by refine pose fail : " << refine_pose_fail <<
                "\r\n\t\t filter by no scale : " << no_scale << 
                "\r\n\t\t filter by not bi-connected : " << no_connection; 
    
    // === 重新排序图像对 ===
    // 经过并行操作后图像对的顺序被打乱，重新按图像索引排序
    // 排列成 0-1, 0-2, 0-3, 0-4, ..., 1-2, 1-3, 1-4, ..., 2-3, 2-4, ..., 3-4 的顺序
    // 这样便于后续处理和调试
    sort(image_pairs.begin(), image_pairs.end(), 
        [this](const MatchPair& mp1,const MatchPair& mp2)
        {
            if(mp1.image_pair.first < mp2.image_pair.first)
                return true;
            else 
                return mp1.image_pair.second < mp2.image_pair.second;
        }
        );

    LOG(INFO) << "=========== Filter Image Pairs end ================";
    return true;
}

bool SfM::RefineRelativePose(MatchPair& image_pair)
{
    return SfMLocalBA(frames[image_pair.image_pair.first], frames[image_pair.image_pair.second], PIXEL_RESIDUAL, image_pair);
}

bool SfM::SetTranslationScaleDepthMap(const Equirectangular& eq, MatchPair& pair)
{
    size_t idx1 = pair.image_pair.first;
    size_t idx2 = pair.image_pair.second;
    const cv::Mat& depth_image1 = frames[idx1].depth_map;
    const cv::Mat& depth_image2 = frames[idx2].depth_map;
    if(depth_image1.empty() || depth_image2.empty())
        return false;
    // 判断一下深度图是否为半尺寸的，如果是半尺寸的后面投影的结果都要除以2
    bool half_size = (depth_image1.rows == int((frames[idx1].GetImageRows() + 1 ) / 2));
    pair.points_with_depth = 0;
    vector<double> scale;
    for(const Eigen::Vector3d& p : pair.triangulated)
    {
        // 把当前点投影到图像1下，计算投影深度和真实深度的比值
        Eigen::Vector2d point_project1 = eq.CamToImage(p) / (1.0 + half_size);
        int row = round(point_project1.y()), col = round(point_project1.x());
        if(!eq.IsInside(cv::Point2i(col, row)))
            continue;
        double depth1 = p.norm();
        const float depth1_real = depth_image1.at<uint16_t>(row, col) / 256.0;
        if(depth1_real <= 0)
            continue;
        double scale1 = depth1_real / depth1;
        // 把当前点投影到图像2下，计算投影深度和真实深度的比值
        const Eigen::Vector3d point_in_frame2 = pair.R_21 * p + pair.t_21;
        Eigen::Vector2d point_project2 = eq.CamToImage(point_in_frame2)  / (1.0 + half_size);
        row = round(point_project2.y());
        col = round(point_project2.x());
        if(!eq.IsInside(cv::Point2i(col, row)))
            continue;

        double depth2 = point_in_frame2.norm();
        const float depth2_real = depth_image2.at<uint16_t>(row, col) / 256.0;
        if(depth2_real <= 0)
            continue;
        double scale2 = depth2_real / depth2;
        // 如果算出来的两个尺度差异太大，就认为不可靠，除去
        if(abs(scale1 - scale2) / min(scale1, scale2) > 0.2)
            continue;
        scale.push_back(scale1);
        scale.push_back(scale2);
    }
    if(scale.size() < 10)
        return false;
    // 对计算得到的scale进行一定的过滤，去掉不准确的scale，保留稳定的scale，并用这些稳定的scale算出最终的scale
    // 具体方法就是把scale分成直方图，只保留直方图中占比较高的几个bin，因为直方图的上限和下限都是根据所有scale中的
    // 最大值和最小值来计算的，所以如果有某些点的scale特别大或特别小，就会导致绝大部分其他点的scale都集中在某几个bin中，
    // 那么这些特别“离谱”的scale就很容易通过直方图剔除掉。这整体是一个迭代的过程，迭代的次数越多，那么scale的分布也就越
    // 集中，算出的scale也就越准确。但是相应的，保留下来的scale也会很少
    // 这种方法确实能得到比较稳定的尺度，但是会导致很多图像对没有尺度信息，因为过滤的太狠了，所以我又选了一个更简单的方法，
    // 也就是把当前图像对的尺度排序，选择中间的那个值作为尺度，但这种方法肯定没有上一种好。因此做了一个判断，如果没法用
    // 好方法得到尺度，那么就用这个差方法
    bool scale_is_good = true;
    vector<double> scale_preserve(scale);
    const size_t num_bins = 10;
    for(size_t iter = 0; iter < 2; iter++)
    {
        size_t num_scale = scale.size();
        if(num_scale < 10)
        {
            // LOG(INFO) << "Not enough scale factor for image pair " << pair.image_pair.first 
            //             << " - " << pair.image_pair.second << endl;
            scale_is_good = false;
            break;
        }
        double max_scale = *(max_element(scale.begin(), scale.end()));
        double min_scale = *(min_element(scale.begin(), scale.end()));
        if(max_scale / min_scale < 1.2)
            break;
        double interval = (max_scale - min_scale) / num_bins;
        // 在计算直方图的时候，最大的那个scale会超出直方图的范围，
        // 解决方法是在所有的scale上进行一点小小的偏移，
        // 比如把所有的数都减去0.0000001，这样就不会造成这种问题了。
        vector<vector<double>> histo(num_bins);
        for(const double& s : scale)
        {
            // 在很少的情况下，如果真的scale都特别小，那么可能就会出现bin_idx越界问题，为了避免这个问题，
            // 就用min 和 max来限制bin_idx的范围
            int bin_idx = int((s - min_scale - 1e-8) / interval);
            bin_idx = min(bin_idx, int(num_bins - 1));
            bin_idx = max(0, bin_idx); 
            histo[bin_idx].push_back(s);
        }
        scale.clear();
        for(const vector<double>& bin: histo)
        {
            if(bin.size() > 0.1 * num_scale)
                scale.insert(scale.end(), bin.begin(), bin.end());
        }
    }
    
    double final_scale = 0;
    if(scale_is_good)
    {
        for(const double& s : scale)
            final_scale += s;
        final_scale /= scale.size();
        pair.t_21 *= final_scale;
        pair.points_with_depth = scale.size() / 2;
        pair.upper_scale = *(max_element(scale.begin(), scale.end()));
        pair.lower_scale = *(min_element(scale.begin(), scale.end()));
    }
    else 
    {
        nth_element(scale_preserve.begin(), scale_preserve.begin() + scale_preserve.size() / 2, scale_preserve.end());
        final_scale = scale_preserve[scale_preserve.size() / 2];
        pair.t_21 *= final_scale;
        pair.upper_scale = 0;
        pair.lower_scale = 0;
        pair.points_with_depth = scale_preserve.size() / 2;
    }
    // 设置了尺度后，三角化的点也要乘以相应的尺度
    for(size_t i = 0; i < pair.triangulated.size(); i++)
        pair.triangulated[i] *= final_scale;
    return true;
}

bool SfM::SetTranslationScaleDepthMap(const bool keep_no_scale)
{
    // 记录对每张图像的深度图的引用数量，一旦数量降到零就代表可以释放当前图像的深度图了
    vector<size_t> depth_ref_count(frames.size(), 0);
    for(const MatchPair& p : image_pairs)
    {
        depth_ref_count[p.image_pair.first] ++;
        depth_ref_count[p.image_pair.second] ++;
    }

    // 互斥量，用于读取深度图和释放深度图
    vector<mutex> depth_mutex(frames.size());
    vector<string> depth_image_names = IterateFiles(config.depth_path, ".bin");
    // 找到引用数量最少的那个图像，也就是说从这张图像开始处理
    size_t start_idx = min_element(depth_ref_count.begin(), depth_ref_count.end()) - depth_ref_count.begin();
    vector<size_t> process_order;
    for(int idx = start_idx; idx < frames.size(); idx++)
        process_order.push_back(idx);
    for(int idx = 0; idx < start_idx; idx++)
        process_order.push_back(idx);
    vector<MatchPair> good_pair;
    set<pair<size_t,size_t>> pairs_processed;
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(const size_t& idx1 : process_order)
    {
        #pragma omp parallel for
        for(MatchPair& p : image_pairs)
        {
            if(pairs_processed.count(p.image_pair) > 0)
                continue;
            size_t idx2;
            if(p.image_pair.first == idx1)
                idx2 = p.image_pair.second;
            else if(p.image_pair.second == idx1)
                idx2 = p.image_pair.first;
            else 
                continue;
            
            // 读取两个frame对应的深度图
            {
                lock_guard<mutex> guard(depth_mutex[idx1]);
                if(frames[idx1].depth_map.empty())
                    ReadOpenCVMat(depth_image_names[idx1], frames[idx1].depth_map);
            }
            
            {
                lock_guard<mutex> guard(depth_mutex[idx2]);
                if(frames[idx2].depth_map.empty())
                    ReadOpenCVMat(depth_image_names[idx2], frames[idx2].depth_map);
            }
            bool valid = true;
            // 设置相对位姿的尺度,如果keep_no_scale=true那么即使当前图像对没有尺度也会保留下来
            if(!SetTranslationScaleDepthMap(eq, p) && !keep_no_scale)
            {
                valid = false;
            }
            #pragma omp critical
            {
                depth_ref_count[idx1]--;
                depth_ref_count[idx2]--;
                if(depth_ref_count[idx1] == 0)
                    frames[idx1].depth_map.release();
                if(depth_ref_count[idx2] == 0)
                    frames[idx2].depth_map.release();
                pairs_processed.insert(p.image_pair);
                if(valid)
                    good_pair.push_back(p);
            }
        }
    }
    good_pair.swap(image_pairs);
    for(Frame& f : frames)
        f.depth_map.release();
    return image_pairs.size() > 0;
}

bool SfM::SetTranslationScaleGPS(const std::string& gps_file, bool overwrite)
{
    if(!LoadGPS(gps_file)) 
        return false;
    for(MatchPair& p : image_pairs)
    {
        if(!overwrite && p.lower_scale >= 0 && p.upper_scale >= 0)
            continue;
        if(!frames[p.image_pair.first].IsGPSValid() || !frames[p.image_pair.second].IsGPSValid())
            continue;
        double scale_gps = (frames[p.image_pair.first].GetGPS() - frames[p.image_pair.second].GetGPS()).norm();
        double scale_pair = p.t_21.norm();
        double ratio = scale_gps / scale_pair;
        p.t_21 *= ratio;
        for(Eigen::Vector3d& point : p.triangulated)
            point *= ratio;
        p.lower_scale = (p.lower_scale > 0 ? p.lower_scale * ratio : 0);
        p.upper_scale = (p.upper_scale > 0 ? p.upper_scale * ratio : 0);
    }
    LOG(INFO) << "Set translation scale using GPS";
    return true;
}


std::vector<MatchPair> SfM::FilterByTriplet(const std::vector<MatchPair>& init_pairs, const double angle_threshold, std::set<size_t>& covered_frames)
{
    LOG(INFO) << "angle threshold for triplet filter: " << angle_threshold;
    covered_frames.clear();
    // 检测Triplet
    set<pair<size_t, size_t>> pairs;
    for(const MatchPair& p : init_pairs)
    {
        pairs.insert(p.image_pair);
        covered_frames.insert(p.image_pair.first);
        covered_frames.insert(p.image_pair.second);
    }
    int num_frames_before_filter = static_cast<int>(covered_frames.size());
    vector<Triplet> triplets = PoseGraph::FindTriplet(pairs);
    // 建立映射关系，可以方便的从图像对找到对应的旋转
    map<pair<size_t, size_t>, Eigen::Matrix3d> map_rotations;
    for(const MatchPair& p : init_pairs)
    {
        assert(p.image_pair.first < p.image_pair.second);
        map_rotations[p.image_pair] = p.R_21;
    }
    // 对初始生成的triplet进行过滤，角度误差超过一定阈值的都过滤掉
    vector<Triplet> valid_triplets;
    Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    for(size_t i = 0; i < triplets.size(); i++)
    {
        const Triplet& triplet = triplets[i];
        // 这里的 idx1 idx2 idx3 是依次增大的
        uint32_t idx1 = triplet.i;
        uint32_t idx2 = triplet.j;
        uint32_t idx3 = triplet.k;
        Eigen::Matrix3d R_21 = map_rotations[pair<size_t, size_t>(idx1, idx2)];
        Eigen::Matrix3d R_32 = map_rotations[pair<size_t, size_t>(idx2, idx3)];
        Eigen::Matrix3d R_13 = map_rotations[pair<size_t, size_t>(idx1, idx3)].transpose();
        Eigen::Matrix3d rot = R_13 * R_32 * R_21;
        double cos_theta = (identity.array() * rot.array()).sum() / 3.0;
        cos_theta = min(1.0, max(cos_theta, -1.0));
        double angle_error = acos(cos_theta) * 180.0 / M_PI;
        if(angle_error < angle_threshold)
            valid_triplets.push_back(triplet);
    }
    LOG(INFO) << "Triplets before filter : " << triplets.size() <<  ", after filter : " << valid_triplets.size();
    covered_frames.clear();
    pairs.clear();
    for(const Triplet& t : valid_triplets)
    {
        pairs.insert(pair<size_t, size_t>(t.i, t.j));
        pairs.insert(pair<size_t, size_t>(t.j, t.k));
        pairs.insert(pair<size_t, size_t>(t.i, t.k));
        covered_frames.insert(t.i);
        covered_frames.insert(t.j);
        covered_frames.insert(t.k);
    }
    LOG(INFO) << "Valid frames before triplet filter : " << num_frames_before_filter << ", after triplet filter: " << covered_frames.size();

    {
        vector<size_t> invalid_frames;
        for(size_t i = 0; i < frames.size(); i++)
            if(covered_frames.count(i) <= 0)
                invalid_frames.push_back(i);
        LOG(INFO) << "invalid frames : " << Join(invalid_frames);
    }

    vector<MatchPair> pairs_after_triplet_filter;
    for(const MatchPair& m : init_pairs)
    {
        if(pairs.count(m.image_pair) > 0)
            pairs_after_triplet_filter.push_back(m);
    }
    // 得到好的Triplet后还要重新建立一次pose graph并过滤
    pairs_after_triplet_filter = LargestBiconnectedGraph(pairs_after_triplet_filter, covered_frames);
    LOG(INFO) << "relative motion after pose graph filter : " << pairs_after_triplet_filter.size() << ", with " << covered_frames.size() << " frames";
    return pairs_after_triplet_filter;
}

/**
 * [功能描述]：从图像匹配对中提取最大的边双连通子图
 * 
 * 边双连通图的重要性：
 * - 在SfM中，图像对形成一个图结构，每张图像是节点，匹配关系是边
 * - 边双连通性确保图中任意两点间至少有两条边不相交的路径
 * - 这保证了位姿图的鲁棒性：即使删除任意一条边，图仍然连通
 * - 对于全局位姿估计和Bundle Adjustment至关重要
 * 
 * @param pairs [输入参数]：所有图像匹配对的列表，每个MatchPair包含两张图像的索引和匹配关系
 * @param nodes [输出参数]：最大边双连通子图中包含的节点(图像)索引集合，通过引用返回
 * @return：属于最大边双连通子图的图像匹配对列表
 */
std::vector<MatchPair> SfM::LargestBiconnectedGraph(const std::vector<MatchPair>& pairs, std::set<size_t>& nodes)
{
    // === 步骤1：构建边集合 ===
    // 将图像匹配对转换为图论中的边集合表示
    // 每个MatchPair代表两张图像之间存在足够的特征匹配，可以估计相对位姿
    set<pair<size_t, size_t>> edges;
    for(const MatchPair& p : pairs)
        // 将图像对 (image1_id, image2_id) 作为图中的边加入边集合
        // emplace比insert更高效，直接在容器中构造对象
        edges.emplace(p.image_pair);
    
    // === 步骤2：创建位姿图对象 ===
    // PoseGraph是专门处理SLAM中位姿图的类
    // 它实现了图论算法来分析图像网络的连通性和鲁棒性
    PoseGraph graph(edges);
    
    // === 步骤3：寻找最大边双连通子图 ===
    // 清空输出节点集合，准备存储新结果
    nodes.clear();
    
    // 调用图算法找到最大的边双连通子图中的所有节点
    // 边双连通子图的特点：
    // 1. 图中任意两个节点间至少存在两条边不相交的路径
    // 2. 删除任意一条边后，图仍然连通
    // 3. 这确保了位姿估计的鲁棒性和稳定性
    nodes = graph.KeepLargestEdgeBiconnected();
    
    // === 步骤4：处理边界情况 ===
    // 如果没有找到有效的边双连通子图，返回空结果
    // 这种情况可能发生在：
    // - 图像匹配质量很差
    // - 数据集太小或图像重叠不足
    // - 所有图像对都是孤立的
    if(nodes.empty())
        return vector<MatchPair>();
    
    // === 步骤5：过滤匹配对 ===
    // 只保留属于最大边双连通子图的图像匹配对
    vector<MatchPair> good_pair;
    for(const MatchPair& p : pairs)
    {
        // 检查当前匹配对的两张图像是否都在最大边双连通子图中
        // 只有当两张图像都在子图中时，这个匹配对才有效
        if(nodes.count(p.image_pair.first) > 0 &&     // 第一张图像在子图中
           nodes.count(p.image_pair.second) > 0)      // 第二张图像在子图中
            good_pair.push_back(p);                    // 保留这个匹配对
    }
    
    // 返回过滤后的匹配对列表
    // 这些匹配对形成一个边双连通的图结构，适合进行全局优化
    return good_pair;
}

bool SfM::RemoveFarPoints(double scale)
{
    if(structure.empty())
        return false;
    size_t num_filter = FilterTracksToFar(frames, structure, scale);
    LOG(INFO) << "Filter " << num_filter << " tracks with base-line, " << structure.size() << " points left";
    return true;
}


bool SfM::EstimateGlobalRotation(const int method)
{
    LOG(INFO) << "================ Estimate Global Rotation begin ================";
    
    // === 可选的GPS尺度设置（当前被注释） ===
    // 如果有GPS数据，可以用GPS信息设置平移的真实尺度
    // if(!config.gps_path.empty())
    //     SetTranslationScaleGPS(config.gps_path, true);
    
    // === 可选的直线运动过滤（当前被注释） ===
    // 过滤掉直线运动的图像序列，因为直线运动容易导致退化情况
    // FilterByStraightMotion(100);
    
    // === 步骤1：选择性使用有尺度的图像对 ===
    // 根据配置决定是否只使用有尺度信息的图像对进行旋转平均
    // 有尺度的图像对通常质量更高，约束更强
    if(!config.use_all_pairs_ra)
    {
        vector<MatchPair> pairs_with_scale;
        for(const MatchPair& p : image_pairs)
        {
            // 检查图像对是否有有效的尺度信息
            // upper_scale和lower_scale是通过深度图或其他传感器信息计算出的尺度边界
            if(p.upper_scale >= 0 && p.lower_scale >= 0)
                pairs_with_scale.push_back(p);
        }
        // 用有尺度的图像对替换原来的图像对列表
        pairs_with_scale.swap(image_pairs);
        LOG(INFO) << "Only use pairs with scale to estimate global rotation, " << image_pairs.size() << " pairs";
    }
    
    // === 调试输出（当前被注释） ===
    // 输出相对位姿到文件中，用于分析和调试
    // PrintRelativePose(config.sfm_result_path + "relpose-RA-before-filter.txt");

    // === 步骤2：确保图的边双连通性 ===
    set<size_t> covered_frames;
    // 保留最大的边双连通子图，确保旋转平均的鲁棒性
    // 边双连通性保证了即使删除任意一条边，图仍然连通
    // 这对全局旋转估计的稳定性至关重要
    image_pairs = LargestBiconnectedGraph(image_pairs, covered_frames);
    LOG(INFO) << "after filter with graph connection: " << image_pairs.size() << " pairs, " << covered_frames.size() << " frames";

    // === 步骤3：三元组一致性过滤 ===
    // 使用三元组（triplet）一致性检查进一步过滤图像对
    // 三元组一致性：对于三张图像A、B、C，R_AB * R_BC应该等于R_AC
    // 这是一个重要的几何约束，可以检测和去除不一致的相对旋转
    image_pairs = FilterByTriplet(image_pairs, 0.1, covered_frames);

    // === 调试输出（当前被注释） ===
    // PrintRelativePose(config.sfm_result_path + "relpose-RA.txt");

    // === 正式开始全局旋转估计 ===
    LOG(INFO) << "Global rotation estimation:\n" << "\t\tprepare to estimate " << covered_frames.size() << 
            " global rotations, with " << image_pairs.size() << " relative motion";

    // === 步骤4：重新索引图像ID ===
    // 为了算法效率，将图像ID重新映射为连续的0,1,2,...
    // 这样可以使用数组而不是map来存储结果，提高访问效率
    map<size_t, size_t> old_to_new, new_to_old;  // 双向映射表
    ReIndex(image_pairs ,old_to_new, new_to_old);
    
    // 更新所有图像对中的ID为新的连续ID
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = old_to_new[pair.image_pair.first];
        pair.image_pair.second = old_to_new[pair.image_pair.second];
    }
    
    // === 步骤5：执行全局旋转估计 ===
    // 创建全局旋转数组，大小等于参与估计的图像数量
    eigen_vector<Eigen::Matrix3d> global_rotations(old_to_new.size());
    bool success;
    
    // 根据选择的方法执行不同的旋转平均算法
    if(method == ROTATION_AVERAGING_L2)
    {
        // === L2旋转平均 ===
        LOG(INFO) << "Rotation averaging L2 begin";
        
        // 首先使用最小二乘法获得初始估计
        // 这个方法速度快，但对外点敏感
        if(!RotationAveragingLeastSquare(image_pairs, global_rotations))
        {
            LOG(ERROR) << "Rotation averaging L2 failed";
            return false;
        }
        
        // 然后使用迭代的L2方法进行精化
        // 这个方法考虑了旋转群的几何结构，结果更精确
        if(!RotationAveragingL2(config.num_threads, image_pairs, global_rotations))
            LOG(ERROR) << "Rotation averaging refine L2 failed";
    }
    else if (method == ROTATION_AVERAGING_L1)
    {
        // === L1旋转平均 ===
        LOG(INFO) << "Rotation averaging L1 begin";
        
        // L1旋转平均对外点更鲁棒，但计算复杂度更高
        // 参数：图像对、输出旋转、起始索引、结束索引(-1表示全部)
        if(!RotationAveragingL1(image_pairs, global_rotations, 0, -1))
        {
            LOG(ERROR) << "Rotation averaging L1 failed";
            return false;
        }
        
        #if 0
        // === 中间结果调试输出（当前被注释） ===
        // global rotation 算出来的是 R_cw (相机到世界)，frame里保存的是 R_wc (世界到相机)
        for(size_t i = 0; i < global_rotations.size(); i++)
            frames[new_to_old[i]].SetRotation(global_rotations[i].transpose());
        PrintGlobalPose(config.sfm_result_path + "frame_pose-after-L1.txt");
        #endif

        // L1之后再用L2进行精化
        // 结合两种方法的优点：L1的鲁棒性 + L2的精度
        if(!RotationAveragingL2(config.num_threads, image_pairs, global_rotations))
            LOG(ERROR) << "Rotation averaging refine L2 failed";
    }
    
    // === 步骤6：更新相对旋转和ID映射 ===
    // 用估计出的全局旋转更新图像对之间的相对旋转
    // 这对后续的平移平均算法很重要
    for(MatchPair& pair : image_pairs)
    {
        // 从全局旋转计算相对旋转
        // R_21 = R_2w * R_1w^T，表示从相机1到相机2的旋转
        const Eigen::Matrix3d& R_1w = global_rotations[pair.image_pair.first];   // 相机1的全局旋转
        const Eigen::Matrix3d& R_2w = global_rotations[pair.image_pair.second];  // 相机2的全局旋转
        pair.R_21 = R_2w * R_1w.transpose();  // 计算相对旋转
        
        // 将图像ID还原为原始ID
        pair.image_pair.first = new_to_old[pair.image_pair.first];
        pair.image_pair.second = new_to_old[pair.image_pair.second];
    }
    
    // === 步骤7：更新Frame中的全局旋转 ===
    for(size_t i = 0; i < global_rotations.size(); i++)
    {
        // 注意坐标系转换：
        // global rotation算出来的是R_cw（相机坐标系到世界坐标系）
        // 但Frame里保存的是R_wc（世界坐标系到相机坐标系）
        // 所以需要转置：R_wc = R_cw^T
        frames[new_to_old[i]].SetRotation(global_rotations[i].transpose());
    }
    
    LOG(INFO) << "===================== Estimate Global Rotation end ===============";
    return true;
}


void SfM::ReIndex(const std::vector<MatchPair>& pairs, std::map<size_t, size_t>& forward, std::map<size_t, size_t>& backward)
{
    forward.clear();
    backward.clear();
    set<size_t> new_id;
    set<size_t> old_id;
    for(const MatchPair& pair : pairs)
    {
        old_id.insert(pair.image_pair.first);
        old_id.insert(pair.image_pair.second);
    }
    for(const MatchPair& pair : pairs)
    {
        if(forward.find(pair.image_pair.first) == forward.end())
        {
            const size_t dist = distance(old_id.begin(), old_id.find(pair.image_pair.first));
            forward[pair.image_pair.first] = dist;
            backward[dist] = pair.image_pair.first;
        }
        if(forward.find(pair.image_pair.second) == forward.end())
        {
            const size_t dist = distance(old_id.begin(), old_id.find(pair.image_pair.second));
            forward[pair.image_pair.second] = dist;
            backward[dist] = pair.image_pair.second;
        }
    }
}

// 尚未完成
bool SfM::EstimateRelativeTwithRotation()
{
    // 找到所有已经计算了全局旋转的frame的id，其实不用id，用当前frame在frames里的索引是一样的
    // 因为这两个数是相同的
    set<uint32_t> frame_with_rotation;
    for(const Frame& f : frames)
    {
        if(f.GetPose().block<3,3>(0,0).isZero())
            continue;
        frame_with_rotation.insert(f.id);
    }
    vector<pair<size_t, size_t>> pair_with_rotation;
    for(const MatchPair& pair : image_pairs)
    {
        const size_t idx1 = pair.image_pair.first;
        const size_t idx2 = pair.image_pair.second;
        if(frame_with_rotation.count(idx1) > 0 && frame_with_rotation.count(idx2) > 0)
            pair_with_rotation.emplace_back(pair.image_pair);
    }
    vector<Triplet> triplets = PoseGraph::FindTriplet(pair_with_rotation);
    LOG(INFO) << "number of triplet with global rotation: " << triplets.size()  << endl;

    // 记录每个图像对（每条边）在估算位姿时用到的triplet，只记录triplet的索引就行
    // 也就是每条边被哪些triplet所包含
    // key = edge  value = triplet id
    map<pair<size_t, size_t>, vector<size_t>> triplet_per_edge;
    for(size_t i = 0; i < triplets.size(); i++)
    {
        const Triplet& trip = triplets[i];
        // (i,j) (i,k) (j,k) 这三条边在计算相对平移的时候都用到了第i个triplet
        // 注意ijk是有顺序的，i < j < k
        triplet_per_edge[pair<size_t, size_t>(trip.i, trip.j)].push_back(i);
        triplet_per_edge[pair<size_t, size_t>(trip.i, trip.k)].push_back(i);
        triplet_per_edge[pair<size_t, size_t>(trip.j, trip.k)].push_back(i);
    }
    // 记录每个triplet能看到的三维点的数量，也就是当前triplet里三个边能三角化的点的数量之和
    // key = triplet id   value = 三维点数量
    map<size_t, size_t> tracks_per_triplet;
    // 记录被triplet包含的edge，其实就是所有的triplet的edge的集合
    vector<pair<size_t, size_t>> valid_edges;
    for(const MatchPair& pair : image_pairs)
    {
        if(triplet_per_edge.count(pair.image_pair) == 0)
            continue;
        valid_edges.push_back(pair.image_pair);
        const vector<size_t>& triplet_id = triplet_per_edge.at(pair.image_pair);
        for(const size_t& id : triplet_id)
        {
            if(tracks_per_triplet.count(id) == 0)
                tracks_per_triplet[id] = pair.triangulated.size();
            else 
                tracks_per_triplet[id] += pair.triangulated.size();
        }
    }
    // 用一个map来记录image_pair 和 它的索引之间的关系，这样就可以快速的通过匹配的图像id找到它在
    // image_pairs 里的索引
    map<pair<size_t, size_t>, size_t> image_pair_to_idx;
    for(size_t i = 0; i < image_pairs.size(); i++)
        image_pair_to_idx[image_pairs[i].image_pair] = i;

    set<pair<size_t, size_t>> processed_edges;
    for(size_t i = 0; i < valid_edges.size(); i++)
    {
        if(processed_edges.count(valid_edges[i]) > 0)
            continue;
        const vector<size_t>& triplet_id = triplet_per_edge[valid_edges[i]];
        // 找到所有包含当前的edge的triplet，然后把这些triplet按照他们能看到的三维点降序排列
        // first - triplet id    second - triplet包含的三维点数量
        vector<pair<size_t, size_t>> triplet_tracks_sorted;
        for(const size_t& id : triplet_id)
            triplet_tracks_sorted.push_back(pair<size_t, size_t>(id, tracks_per_triplet[id]));
        sort(triplet_tracks_sorted.begin(), triplet_tracks_sorted.end(), 
            [this](pair<size_t, size_t> a, pair<size_t,size_t> b){return a.second > b.second;});
        for(const pair<size_t,size_t>& t : triplet_tracks_sorted)
        {
            size_t id = t.first;
            const Triplet& triplet = triplets[id];
            vector<pair<size_t, size_t>> pairs = {pair<size_t, size_t>(triplet.i, triplet.j),
                                                pair<size_t, size_t>(triplet.i, triplet.k),
                                                pair<size_t, size_t>(triplet.j, triplet.k)};
            // 如果这个triplet所包含的三条边都已经处理过了，那就跳过
            if(processed_edges.count(pairs[0]) > 0 && 
               processed_edges.count(pairs[1]) > 0 &&
               processed_edges.count(pairs[2]) > 0 )
               continue;
            // 根据每个triplet计算相对平移
            // 1.找到这三个边对应的matches
            vector<vector<cv::DMatch>> pair_matches = {image_pairs[image_pair_to_idx[pairs[0]]].matches,
                                                    image_pairs[image_pair_to_idx[pairs[1]]].matches,
                                                    image_pairs[image_pair_to_idx[pairs[2]]].matches};
            // 2. 根据特征点之间的match生成track，并且只保留长度为3的track，也就是三张图像上都有关联
            TrackBuilder tracks_builder;
            tracks_builder.Build(pairs, pair_matches);
            tracks_builder.Filter(3);
            map<uint32_t, set<pair<uint32_t, uint32_t>>> tracks;
            tracks_builder.ExportTracks(tracks);
            if(tracks.size() < 30)
                continue;
            // openmp 并行时要单独操作
            {
                processed_edges.insert(pairs[0]);
                processed_edges.insert(pairs[1]);
                processed_edges.insert(pairs[2]);
            }
        }
    }
    return false;
}


bool SfM::EstimateGlobalTranslation(const int method)
{
    LOG(INFO) << "==================== Estimate Global Translation start =================";

    if(!config.gps_path.empty())
        SetTranslationScaleGPS(config.gps_path, true);

    // 先根据全局旋转进行相对平移的估计，得到更准确的相对平移
    // 尚未完成
    // EstimateRelativeTwithRotation();


    PrintRelativePose(config.sfm_result_path + "relpose-TA-raw.txt");
    PrintGlobalPose(config.sfm_result_path + "global-TA-raw.txt");
    // FilterByIndexDifference(10, 100, 8450);
    // FilterByStraightMotion(30);
    
    /*************************************************************************************************/
    // 找到所有已经计算了全局旋转的frame的id，其实不用id，用当前frame在frames里的索引是一样的
    // 因为这两个数是相同的
    set<uint32_t> frame_with_rotation;
    for(const Frame& f : frames)
    {
        if(f.GetPose().block<3,3>(0,0).isZero())
            continue;
        frame_with_rotation.insert(f.id);
    }
    vector<MatchPair> pair_with_rotation;
    for(const MatchPair& pair: image_pairs)
    {
        if(frame_with_rotation.count(pair.image_pair.first) > 0 && 
            frame_with_rotation.count(pair.image_pair.second) > 0)
            pair_with_rotation.emplace_back(pair);
    }
    LOG(INFO) << "image pairs with global rotaion: " << pair_with_rotation.size();
    // 把匹配的图像对分成有尺度和无尺度两种
    vector<MatchPair> pair_with_scale, pair_without_scale;
    for(const MatchPair& p : pair_with_rotation)
    {
        if(p.upper_scale >= 0 && p.lower_scale >= 0)
            pair_with_scale.push_back(p);
        else 
            pair_without_scale.push_back(p);
    }
    LOG(INFO) << "image pairs with scale: " << pair_with_scale.size() << ", without scale: " << pair_without_scale.size(); 
    assert(pair_with_rotation.size() == pair_with_scale.size() + pair_without_scale.size());
    
    // 同样，仅保留最大的边双连通子图
    set<size_t> largest_component_nodes;
    if(config.use_all_pairs_ta)
    {
        image_pairs = LargestBiconnectedGraph(pair_with_rotation, largest_component_nodes);
        LOG(INFO) << "Estimate global translation with all image pairs";
    }
    else 
    {
        image_pairs = LargestBiconnectedGraph(pair_with_scale, largest_component_nodes);
        LOG(INFO) << "Estiamte global translation with scaled image pairs";
    }
    if(image_pairs.empty())
    {
        LOG(ERROR) << "no nodes are bi-edge connected";
        return false;
    }
    pair_with_rotation.clear();
    // 输出一下更多的关于图像对的信息，debug方便些,删去这部分也不影响
    {    
        size_t points_with_depth = 0;
        size_t histo[10] = {0};
        for(const MatchPair& pair : image_pairs)
        {
            points_with_depth += pair.points_with_depth;
            size_t histo_idx = pair.points_with_depth / 10;
            histo_idx = max(0, min((int)histo_idx, 9));
            histo[histo_idx]++;
        }
        LOG(INFO) << "Image pair statistic: points with depth " << 
                    "\n 0-9: " << histo[0] << 
                    "\n 10-19: " << histo[1] << 
                    "\n 20-29: " << histo[2] <<
                    "\n 30-39: " << histo[3] <<
                    "\n 40-49: " << histo[4] <<
                    "\n 50-59: " << histo[5] <<
                    "\n 60-69: " << histo[6] <<
                    "\n 70-79: " << histo[7] <<
                    "\n 80-89: " << histo[8] <<
                    "\n 90-inf: " << histo[9] <<
                    "\n points with depth per image pair: " << 1.f * points_with_depth / image_pairs.size();
    }

    PrintRelativePose(config.sfm_result_path + "relpose-TA.txt");
    
    LOG(INFO) << "Global translation estimation:\n       prepare to estimate " << largest_component_nodes.size() <<
                " camera translations, with " << image_pairs.size() << " relative motion";

    srand((unsigned)time(NULL));
    eigen_vector<Eigen::Vector3d> global_translations(largest_component_nodes.size(), Eigen::Vector3d::Zero());
    for(size_t i = 1; i < global_translations.size(); i++)
        global_translations[i] = Eigen::Vector3d::Random();

    // 1.进行全局的重映射，把图像id映射为连续的n个数字
    map<size_t, size_t> old_to_new_global, new_to_old_global;
    ReIndex(image_pairs, old_to_new_global, new_to_old_global);
    for(MatchPair& pair : pair_with_scale)
    {
        pair.image_pair.first = old_to_new_global[pair.image_pair.first];
        pair.image_pair.second = old_to_new_global[pair.image_pair.second];
    }
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = old_to_new_global[pair.image_pair.first];
        pair.image_pair.second = old_to_new_global[pair.image_pair.second];
    }
    size_t origin_idx = 0;
    bool success;

    // 2.处理有尺度的图像对，使用DLT算出他们的绝对平移,这里还要再经过一次id重映射，因为可能不是所有图像都被覆盖了
    // 或者使用GPS作为初始的相机位姿
    if(config.init_translation_DLT && !pair_with_scale.empty())
    {
        map<size_t, size_t> old_to_new, new_to_old;
        pair_with_scale = LargestBiconnectedGraph(pair_with_scale, largest_component_nodes);
        if(largest_component_nodes.size() <= 3)
        {
            LOG(ERROR) << "pairs with scale are not enough";
        }
        ReIndex(pair_with_scale, old_to_new, new_to_old);
        for(MatchPair& pair : pair_with_scale)
        {
            pair.image_pair.first = old_to_new[pair.image_pair.first];
            pair.image_pair.second = old_to_new[pair.image_pair.second];
        }
        LOG(INFO) << "Use DLT to init global translations: " << pair_with_scale.size() << " pairs, " << 
                    largest_component_nodes.size() << " images";
        eigen_vector<Eigen::Vector3d> global_translation_scale(largest_component_nodes.size());
        
        success = TranslationAveragingDLT(image_pairs, global_translation_scale);
        if(!success)
        {
            LOG(ERROR) << "Translation average DLT failed";
            return false;
        }
        // 在后面的方法里，一般都要要求设置某个图像为原点，也就是把某个图像的平移设置为(0,0,0)并固定不动，这里就把最小二乘法得到的
        // 结果中的第一个设置为原点
        origin_idx = new_to_old_global[new_to_old[0]];
        for(size_t i = 0; i < global_translation_scale.size(); i++)
        {
            Eigen::Matrix3d R_wc = frames[new_to_old_global[new_to_old[i]]].GetPose().block<3,3>(0,0);
            Eigen::Vector3d t_wc = -R_wc * global_translation_scale[i];
            frames[new_to_old_global[new_to_old[i]]].SetTranslation(t_wc);
        }
        // 把所有的图像位姿都变换到以origin_idx为坐标原点的世界坐标系下
        // 如果origin_idx=0那这个其实就没用，因为当前的位姿就是在以0为原点的坐标系下计算得到的
        SetToOrigin(origin_idx);
        for(size_t i = 0; i < global_translations.size(); i++)
        {
            map<size_t, size_t>::iterator it = new_to_old_global.find(i);
            if(it == new_to_old_global.end())
                continue;
            if(!frames[it->second].IsPoseValid())
                continue;
            global_translations[i] = frames[it->second].GetPose().inverse().block<3,1>(0,3);
        }
        // origin_idx已经是对应于图像的id了，但是由于不是所有图像都被image pair覆盖了，所以还要把origin_idx变换到新的索引之下
        origin_idx = old_to_new_global[origin_idx];

        // 用于debug，保存一下DLT结果并显示
        CameraCenterPCD(config.sfm_result_path + "/camera_center_DLT.pcd", GetGlobalTranslation(true));
        
    }
    
    if(config.init_translation_GPS && !config.gps_path.empty())
    {
        if(!LoadGPS(config.gps_path))
        {
            LOG(ERROR) << "fail to load GPS";
            return false;
        }

        success = InitGlobalTranslationGPS(frames, global_translations, new_to_old_global);
        if(!success)
        {
            LOG(ERROR) << "Use GPS to set init translation failed";
            return false;
        }
        for(size_t i = 0; i < global_translations.size(); i++)
        {
            Eigen::Matrix3d R_wc = frames[new_to_old_global[i]].GetPose().block<3,3>(0,0);
            Eigen::Vector3d t_wc = -R_wc * global_translations[i];
            frames[new_to_old_global[i]].SetTranslation(t_wc);
        }
        SetToOrigin(new_to_old_global[origin_idx]);
        // 用于debug，保存一下DLT结果并显示
        CameraCenterPCD(config.sfm_result_path + "/camera_center_GPS.pcd", GetGlobalTranslation(true));
    }
    
    // 3.进行平移平均
    if(method == TRANSLATION_AVERAGING_SOFTL1)
    {
        LOG(INFO) << "Translation average Soft L1 ";
        success = TranslationAveragingSoftL1(image_pairs, global_translations, origin_idx, 0.01, config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average Soft L1 failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_CHORDAL)
    {
        LOG(INFO) << "Translation average chordal ";
        success = TranslationAveragingL2Chordal(image_pairs, frames, global_translations, new_to_old_global, origin_idx, 0.5, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average chordal failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_L1)
    {
        LOG(INFO) << "Translation average L1 ";
        success = TranslationAveragingL1(image_pairs, global_translations, origin_idx, new_to_old_global);
        if(!success)
        {
            LOG(ERROR) << "Translation average L1 failed";
            return false;
        }
    }
    else if (method == TRANSLATION_AVERAGING_L2IRLS)
    {
        LOG(INFO) << "Translation average L2 IRLS";
        success = TranslationAveragingL2IRLS(image_pairs, global_translations, origin_idx, config.num_iteration_L2IRLS,
                                            config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average L2 IRLS failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_BATA)
    {
        LOG(INFO) << "Translation average BATA ";
        success = TranslationAveragingBATA(image_pairs, frames, global_translations, new_to_old_global, origin_idx, config.sfm_result_path);
        if(!success)
        {
            LOG(ERROR) << "Translation average BATA failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_LUD)
    {
        LOG(INFO) << "Translation average LUD ";
        success = TranslationAveragingLUD(image_pairs, frames, global_translations, new_to_old_global, origin_idx, config.num_iteration_L2IRLS, 
                                    config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average LUD failed";
            return false;
        }
    }
    else 
    {
        LOG(ERROR) << "Translaton average method not supported";
        return false;
    }
    
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = new_to_old_global[pair.image_pair.first];
        pair.image_pair.second = new_to_old_global[pair.image_pair.second];
    }
    set<size_t> frame_with_translation;
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        // global translation 算出来的是 t_cw, frame里保存的是 t_wc
        const Eigen::Vector3d& t_cw = global_translations[i];
        const Eigen::Matrix3d& R_wc = frames[new_to_old_global[i]].GetPose().block<3,3>(0,0);
        Eigen::Vector3d t_wc = - R_wc * t_cw;
        frames[new_to_old_global[i]].SetTranslation(t_wc);
        frame_with_translation.insert(new_to_old_global[i]);
    }
    // 仅保留有全局位姿的匹配对
    vector<MatchPair> good_pair;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].image_pair.first;
        const size_t idx2 = image_pairs[i].image_pair.second;
        if(frame_with_translation.count(idx1) && frame_with_translation.count(idx2) &&
            frame_with_rotation.count(idx1) && frame_with_rotation.count(idx2))
        {
            good_pair.push_back(image_pairs[i]);
        }
    }
    good_pair.swap(image_pairs);
    LOG(INFO) << "Image pairs with global pose: " << image_pairs.size() ;

    LOG(INFO) << "==================== Estimate Global Translation end =================";
    return true;
}

bool SfM::EstimateStructure()
{
    LOG(INFO) << "==================== Estimate Initial Structure start =================";
    structure = TriangulateTracks(frames, image_pairs);
    
    track_triangulated = true;
    if(config.colorize_structure)
    {
        LOG(INFO) << "Start to colorize initial structure";
        ColorizeStructure();
    }

    LOG(INFO) << "==================== Estimate Initial Structure end =================";
    return true;
}

/**
 * [功能描述]：执行全局束集调整(Global Bundle Adjustment)
 * 
 * Bundle Adjustment是SfM中的核心优化步骤：
 * - 同时优化所有相机位姿参数和3D点位置
 * - 最小化重投影误差，提高重建精度
 * - 通过非线性最小二乘法求解大规模稀疏优化问题
 * 
 * @param residual_type [输入]：残差计算类型
 *        - ANGLE_RESIDUAL_1/2: 使用角度残差（球面坐标系下的角度误差）
 *        - PIXEL_RESIDUAL: 使用像素残差（图像平面上的像素误差）
 * @param redisual_threshold [输入]：残差过滤阈值
 *        - 超过此阈值的观测点将被标记为外点并过滤掉
 *        - 单位取决于residual_type（角度用弧度，像素用像素）
 * @param refine_structure [输入]：是否优化3D点坐标
 *        - true: 同时优化相机位姿和3D点位置（完整BA）
 *        - false: 只优化相机位姿，固定3D点位置（位姿优化）
 * @param refine_rotation [输入]：是否优化相机旋转参数
 *        - 在某些情况下可能只想优化平移而固定旋转
 * @param refine_translation [输入]：是否优化相机平移参数
 *        - 在某些情况下可能只想优化旋转而固定平移
 * @return：优化是否成功完成
 */
bool SfM::GlobalBundleAdjustment(int residual_type, float redisual_threshold, bool refine_structure, bool refine_rotation, bool refine_translation)
{
    // === 步骤1：执行全局束集调整优化 ===
    // 调用核心的全局BA算法，同时优化所有相机位姿和3D结构
    // SfMGlobalBA是实际执行优化的函数，使用Ceres Solver等优化库
    // 参数说明：
    // - frames: 所有图像帧，包含相机位姿和特征点信息
    // - structure: 所有3D点轨迹，包含3D坐标和观测信息
    // - residual_type: 残差计算方式（像素残差 vs 角度残差）
    // - config.num_threads: 并行优化的线程数
    // - refine_structure/rotation/translation: 控制优化哪些参数
    if(!SfMGlobalBA(frames, structure, residual_type, 
                    config.num_threads, refine_structure, refine_rotation, refine_translation))
    {
        // 如果BA优化失败，记录错误并返回false
        // 失败原因可能包括：数值不稳定、初值太差、约束不足等
        LOG(ERROR) << "Global BA failed";
        return false;
    }
    
    // === 步骤2：基于残差过滤低质量的3D点轨迹 === 
    size_t num_filter;  // 记录被过滤掉的点轨迹数量
    
    // 根据残差类型选择对应的过滤方法
    if(residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_1 || residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_2)
    {
        // === 角度残差过滤 ===
        // 适用于全景相机或鱼眼相机等大视野相机
        // 角度残差衡量的是：3D点投影到单位球面上的方向 与 实际观测方向 之间的夹角
        // 这种残差在处理大畸变图像时比像素残差更稳定和准确
        num_filter = FilterTracksAngleResidual(frames, structure, redisual_threshold);
        LOG(INFO) << "Filter " << num_filter << " tracks by angle residual, " << structure.size() << " points left";
    }
    else if(residual_type == RESIDUAL_TYPE::PIXEL_RESIDUAL)
    {
        // === 像素残差过滤 ===
        // 适用于针孔相机模型
        // 像素残差衡量的是：3D点投影到图像平面的位置 与 实际特征点位置 之间的欧氏距离
        // 这是最常见的残差类型，直观易懂，适用于大多数相机
        num_filter = FilterTracksPixelResidual(frames, structure, redisual_threshold);
        LOG(INFO) << "Filter " << num_filter << " tracks by pixel residual, " << structure.size() << " points left";
    }
    
    // === 步骤3：返回成功标志 ===
    // 执行到这里说明BA优化和后处理都成功完成
    // 此时相机位姿和3D结构都已经得到精化，重投影误差得到最小化
    return true;
}

bool SfM::SetToOrigin(size_t frame_idx)
{
    if(frame_idx > frames.size())
    {
        LOG(ERROR) << "Invalid frame idx, no frame in frame list";
        return false;
    }
    if(!frames[frame_idx].IsPoseValid())
    {
        LOG(WARNING) << "Frame " << frame_idx << " pose is invalid, set another frame";
        for(frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            if(frames[frame_idx].IsPoseValid())
                break;
        }
        LOG(INFO) << "Set frame " << frame_idx << " as world coordinate";
    }
    // 这里的下标c 代表center，是指的新的世界坐标系
    const Eigen::Matrix4d T_wc = frames[frame_idx].GetPose();
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_iw = frames[i].GetPose().inverse();
        Eigen::Matrix4d T_ic = T_iw * T_wc;     // 新的世界坐标系到相机坐标系的变换
        frames[i].SetPose(T_ic.inverse());
    }
    if(track_triangulated)
    {
        Eigen::Matrix4d T_cw = T_wc.inverse();
        for(PointTrack& track : structure)
        {
            track.point_3d = (T_cw * track.point_3d.homogeneous()).hnormalized();
        }
    }
    return true;
}

bool SfM::ColorizeStructure()
{
    if(structure.empty())
    {
        LOG(ERROR) << "No structure to colorize";
        return false;
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());

    map<int, set<int>> structures_each_frame;
    for(int i = 0; i < structure.size(); i++)
    {
        for(const pair<uint32_t, uint32_t>& pair : structure[i].feature_pairs)
        {
            structures_each_frame[pair.first].insert(i);
        }
    }
    // 先使用Vector3d来记录所有颜色之和，因为使用Vector3i可能会数据溢出
    eigen_vector<Eigen::Vector3d> structure_color(structure.size(), Eigen::Vector3d::Zero());
    #pragma omp parallel for
    for(int frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        map<int, set<int>>::const_iterator it = structures_each_frame.find(frame_id);
        if(it == structures_each_frame.end())
            continue;
        cv::Mat img_color = frames[frame_id].GetImageColor();
        for(const int& structure_idx : it->second)
        {
            for(const pair<uint32_t, uint32_t>& pair : structure[structure_idx].feature_pairs)
            {
                if(pair.first != frame_id)
                    continue;
                
                const cv::Point2f pt = frames[pair.first].GetKeyPoints()[pair.second].pt;
                const cv::Point2i pt_round = cv::Point2i(round(pt.x), round(pt.y));
                if(!frames[frame_id].IsInside(pt_round))
                    continue;
                const cv::Vec3b bgr = img_color.at<cv::Vec3b>(pt_round);
                #pragma omp critical
                {
                    structure_color[structure_idx].x() += bgr[2];
                    structure_color[structure_idx].y() += bgr[1];
                    structure_color[structure_idx].z() += bgr[0];
                }
            }
        }
    }
    for(int structure_idx = 0; structure_idx < structure.size(); structure_idx++)
    {
        structure_color[structure_idx] /= structure[structure_idx].feature_pairs.size();
        structure[structure_idx].rgb = structure_color[structure_idx].cast<int>();
    }
    return true;
}

int SfM::CheckRT(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, 
            const std::vector<bool>& is_inlier, const std::vector<cv::DMatch>& matches, 
            const std::vector<cv::Point3f>& keypoints1, 
            const std::vector<cv::Point3f>& keypoints2,
            double& parallax, eigen_vector<Eigen::Vector3d>& triangulated_points,
            std::vector<size_t>& inlier_idx)
{
    assert(is_inlier.size() == matches.size());
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    double cos_parallax_threshold = cos(0.5 / 180.0 * M_PI);
    double sq_reproj_error_threshold = 6.0 * 6.0;
    vector<double> parallaxes;

    const Eigen::Vector3d camera_center1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d camera_center2 = - R_21.transpose() * t_21;   // t_12

    int num_points_tri = 0;
    triangulated_points.resize(0);
    inlier_idx.clear();
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(!is_inlier[i])
            continue;
        const cv::Point3f& p1 = keypoints1[matches[i].queryIdx];
        const cv::Point3f& p2 = keypoints2[matches[i].trainIdx];
        Eigen::Vector3d point_triangulated = Triangulate2View(R_21, t_21, p1, p2);
        // Eigen::Vector3d point_triangulated = Triangulate2ViewIDWM(R_21, t_21, p1, p2);
        if (!std::isfinite(point_triangulated(0))
            || !std::isfinite(point_triangulated(1))
            || !std::isfinite(point_triangulated(2))) 
            continue;
        // 计算视差
        Eigen::Vector3d norm1 = (point_triangulated - camera_center1).normalized();
        Eigen::Vector3d norm2 = (point_triangulated - camera_center2).normalized();

        double cos_parallax = norm2.dot(norm1);
        // 角度越小，那么对应的余弦就越大，因此如果当前夹角的余弦值大于阈值，就说明当前的夹角是很小的
        // 这种情况下三角化很可能就出错了
        const bool parallax_is_small = cos_parallax_threshold < cos_parallax;

        Eigen::Vector3d p1_eigen(p1.x, p1.y, p1.z);
        Eigen::Vector3d p2_eigen(p2.x, p2.y, p2.z);

        Eigen::Vector3d point_in_frame2 = R_21 * point_triangulated + t_21;

        // 计算三角化得到的点和它对应的图像点之间的夹角，如果夹角太大就认为三角化错误
        double reproj_error_angle1 = VectorAngle3D(norm1.data(), p1_eigen.data()) * 180.0 / M_PI;
        if(reproj_error_angle1 > 3)
            continue;

        double reproj_error_angle2 = VectorAngle3D(point_in_frame2.data(), p2_eigen.data()) * 180.0 / M_PI;
        if(reproj_error_angle2 > 3)
            continue;
        num_points_tri ++;
        parallaxes.push_back(cos_parallax);
        triangulated_points.push_back(point_triangulated);
        inlier_idx.push_back(i);
    }
    if(num_points_tri > 0)
    {
        // 把视差的余弦按照从小到大的顺序排列，然后找到其中的第50个，计算他所对应的视差
        // 其实就是把所有视差按从大到小排列，找到第50大的视差
        sort(parallaxes.begin(), parallaxes.end());
        size_t idx = min(50, static_cast<int>(parallaxes.size()));
        parallax = acos(parallaxes[idx - 1]) * 180.0 / M_PI;
    }
    else 
        parallax = 0;
    return num_points_tri;
}

void SfM::VisualizeTrack(const PointTrack& track, const string path)
{
    int track_id = track.id;
    Eigen::Vector3d point_world = track.point_3d;
    for(const auto& pair : track.feature_pairs)
    {
        cv::Mat img = frames[pair.first].GetImageColor();
        cv::Point2f pt = frames[pair.first].GetKeyPoints()[pair.second].pt;
        Equirectangular eq(img.rows, img.cols);
        cv::circle(img, pt, 20, cv::Scalar(0,0,255), 5);
        Eigen::Vector3d point_camera = (frames[pair.first].GetPose().inverse() * point_world.homogeneous()).hnormalized();
        Eigen::Vector2d point_image = eq.CamToImage(point_camera);
        pt.x = static_cast<float>(point_image.x());
        pt.y = static_cast<float>(point_image.y());
        cv::circle(img, pt, 20, cv::Scalar(255,0,0), 5);
        cv::imwrite(path + "/track" + num2str(track_id) + "_" + num2str(pair.first) + ".jpg", img);
    }
}

bool SfM::ExportMatchPairTXT(const std::string file_name)
{
    LOG(INFO) << "Save match pair at " << file_name;
    ofstream f(file_name);
    if(!f.is_open())
        return false;
    for(MatchPair& p:image_pairs)
    {
        f << p.image_pair.first << " " << p.image_pair.second << endl;
        f << p.R_21(0, 0) << " " << p.R_21(0, 1) << " " << p.R_21(0, 2) << " " << p.t_21(0) << " " 
          << p.R_21(1, 0) << " " << p.R_21(1, 1) << " " << p.R_21(1, 2) << " " << p.t_21(1) << " "
          << p.R_21(2, 0) << " " << p.R_21(2, 1) << " " << p.R_21(2, 2) << " " << p.t_21(2) << endl;
        f << "points with depth: " << p.points_with_depth << endl;
    }
    f.close();
    return true;
}

bool SfM::LoadMatchPairTXT(const std::string file_name)
{
    ifstream f(file_name);
    if(!f.is_open())
    {
        LOG(ERROR) << "Can not open file " << file_name;
        return false;
    }
    image_pairs.clear();
    size_t largets_idx = 0;
    while(!f.eof())
    {
        // 从文件中读取数据并保存到image pairs
        // 同时还要检查一下数据是否正确，可以通过检查读取的图像index来判断一下
        // 如果图像的index大于已有的图像数量，那么就可以肯定这个数据不对
        size_t i , j;
        Eigen::Matrix3d R_21;
        Eigen::Vector3d t_21;
        f >> i >> j;
        f >> R_21(0, 0) >> R_21(0, 1) >> R_21(0, 2) >> t_21(0) 
          >> R_21(1, 0) >> R_21(1, 1) >> R_21(1, 2) >> t_21(1)        
          >> R_21(2, 0) >> R_21(2, 1) >> R_21(2, 2) >> t_21(2);
        string str;
        getline(f, str);
        getline(f, str);
        str = str.substr(19);
        int num_points = str2num<int>(str);
        MatchPair p(i,j);
        p.R_21 = R_21;
        p.t_21 = t_21;
        p.points_with_depth = num_points;
        // 三维向量到反对称矩阵
        Eigen::Matrix3d t_21_hat = Eigen::Matrix3d::Zero();
        t_21_hat <<    0,          -t_21.z(),      t_21.y(),
                    t_21.z(),           0,        -t_21.x(),
                    -t_21.y(),      t_21.x(),           0;
        p.E_21 = t_21_hat * R_21;

        image_pairs.push_back(p);

        if(i > largets_idx)
            largets_idx = i;
        if(j > largets_idx)
            largets_idx = j;

        if(f.peek() == EOF)
            break;
    }
    if(largets_idx >= frames.size())
    {
        LOG(ERROR) << "Fail to load match pair at" << file_name << endl;
        image_pairs.clear();
        return false;
    }
    LOG(INFO) << "Successfully load " << image_pairs.size() << " match pairs from txt";
    return true;
}

bool SfM::ExportMatchPairBinary(const std::string folder)
{
    return ExportMatchPair(folder, image_pairs);
}

bool SfM::LoadMatchPairBinary(const std::string folder)
{
    // 读取的时候不需要太多的线程，没办法提升效率，而且可能还会导致效率降低
    if(!ReadMatchPair(folder, image_pairs, min(config.num_threads, 4)))
        return false;
    if(image_pairs[image_pairs.size() - 1].image_pair.second > frames.size())
    {
        image_pairs.clear();
        LOG(ERROR) << "Fail to load match pair, #images(in pairs) more than #frames";
        return false;
    }
    LOG(INFO) << "Successfully load " << image_pairs.size() << " match pairs from " << folder;
    return true;

}

bool SfM::ExportFrameBinary(const std::string folder)
{
    if(frames.empty())
        return false;
    return ExportFrame(folder, frames);
}

bool SfM::ExportStructureBinary(const std::string file_name)
{
    if(structure.empty())
        return false;
    return ExportPointTracks(file_name, structure);
}

bool SfM::LoadStructureBinary(const std::string file_name)
{
    structure.clear();
    return ReadPointTracks(file_name, structure);
}

bool SfM::LoadGPS(const std::string file_name)
{
    eigen_vector<Eigen::Vector3d> gps_list;
    vector<string> name_list;
    ReadGPS(config.gps_path, gps_list, name_list);
    if(gps_list.size() != frames.size())
    {
        LOG(ERROR) << "Fail to load GPS file, number of GPS != number of Frame";
        return false;
    }
    for(int i = 0; i < frames.size(); i++)
        frames[i].SetGPS(gps_list[i]);
    return true;
}

eigen_vector<Eigen::Matrix3d> SfM::GetGlobalRotation(bool with_invalid)
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

eigen_vector<Eigen::Vector3d> SfM::GetGlobalTranslation(bool with_invalid)
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

std::vector<std::string> SfM::GetFrameNames(bool with_invalid)
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

bool SfM::ExportStructurePCD(const string file_name)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    for(const PointTrack& t : structure)
    {
        const Eigen::Vector3d point = t.point_3d;
        const Eigen::Vector3i color = t.rgb;
        pcl::PointXYZRGB pt;
        pt.x = point.x();
        pt.y = point.y();
        pt.z = point.z();
        pt.r = static_cast<uchar>(color.x());
        pt.g = static_cast<uchar>(color.y());
        pt.b = static_cast<uchar>(color.z());
        cloud.push_back(pt);
    }
    pcl::io::savePCDFileASCII(file_name, cloud);
    return true;
}

const std::vector<Frame>& SfM::GetFrames() const 
{
    return frames;
}

const std::vector<Velodyne>& SfM::GetLidars() const 
{
    return lidars;
}

void SfM::SetLidars(const std::vector<Velodyne>& _lidars)
{
    lidars = _lidars;
}
SfM::~SfM()
{
}