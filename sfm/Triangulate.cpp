/*
 * @Author: Diantao Tu
 * @Date: 2021-12-15 09:58:23
 */

#include "Triangulate.h"

/**
 * [功能描述]：使用双视角三角化方法重建空间中的三维点
 * 基于两个相机的相对位姿和对应的观测方向，计算空间点在第一个相机坐标系下的三维坐标
 * @param [R_21]：从第一个相机坐标系到第二个相机坐标系的旋转矩阵
 * @param [t_21]：第一个相机原点在第二个相机坐标系下的位置向量
 * @param [p1]：空间点在第一个相机坐标系下的归一化观测方向（单位方向向量）
 * @param [p2]：空间点在第二个相机坐标系下的归一化观测方向（单位方向向量）
 * @return：空间点在第一个相机坐标系下的三维坐标
 */
Eigen::Vector3d Triangulate2View(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    // 将OpenCV的Point3f格式转换为Eigen向量格式
    Eigen::Vector3d point1(p1.x, p1.y, p1.z);  // 第一个相机的观测方向
    Eigen::Vector3d point2(p2.x, p2.y, p2.z);  // 第二个相机的观测方向
    
    // 计算第二个相机原点在第一个相机坐标系下的位置
    // trans_12 = -R_21^T * t_21，表示相机2到相机1的平移向量
    const Eigen::Vector3d trans_12 = -R_21.transpose() * t_21;
    
    // 将第二个相机的观测方向变换到第一个相机坐标系下
    // bearing_2_in_1 = R_21^T * point2，表示point2在相机1坐标系下的方向
    const Eigen::Vector3d bearing_2_in_1 = R_21.transpose() * point2;

    // 构建线性方程组 A * λ = b 来求解深度参数
    // 数学原理：设空间点为P，则有：
    // P = λ1 * point1 (在相机1坐标系下)
    // P = λ2 * bearing_2_in_1 + trans_12 (第二个相机观测变换到相机1坐标系下)
    // 因此：λ1 * point1 = λ2 * bearing_2_in_1 + trans_12
    
    // 构建系数矩阵A (2x2矩阵)
    Eigen::Matrix2d A;
    
    // A的第一行：基于点积约束 point1 · (λ1 * point1 - λ2 * bearing_2_in_1 - trans_12) = 0
    // 展开得：λ1 * (point1 · point1) - λ2 * (point1 · bearing_2_in_1) = point1 · trans_12
    A(0, 0) = point1.dot(point1);                    // λ1的系数
    A(0, 1) = -point1.dot(bearing_2_in_1);          // λ2的系数
    
    // A的第二行：基于点积约束 bearing_2_in_1 · (λ1 * point1 - λ2 * bearing_2_in_1 - trans_12) = 0
    // 展开得：λ1 * (bearing_2_in_1 · point1) - λ2 * (bearing_2_in_1 · bearing_2_in_1) = bearing_2_in_1 · trans_12
    A(1, 0) = bearing_2_in_1.dot(point1);           // λ1的系数
    A(1, 1) = -bearing_2_in_1.dot(bearing_2_in_1);  // λ2的系数

    // 构建常数向量b (2x1向量)
    const Eigen::Vector2d b{point1.dot(trans_12), bearing_2_in_1.dot(trans_12)};

    // 求解线性方程组 A * λ = b，得到深度参数λ1和λ2
    const Eigen::Vector2d lambda = A.inverse() * b;
    
    // 根据求解的深度参数计算两条射线上的最近点
    const Eigen::Vector3d pt_1 = lambda(0) * point1;                    // 第一条射线上的点
    const Eigen::Vector3d pt_2 = lambda(1) * bearing_2_in_1 + trans_12; // 第二条射线变换到相机1坐标系后的点
    
    // 返回两个最近点的中点作为三角化结果
    // 由于噪声和数值误差，两条射线通常不会完全相交，取中点可以最小化重投影误差
    return (pt_1 + pt_2) / 2.0;
}

// 这个不太准，不知道为啥, 所以不要用
Eigen::Vector3d Triangulate2View_2(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    Eigen::Vector3d point1(p1.x, p1.y, p1.z);
    Eigen::Vector3d point2(p2.x, p2.y, p2.z);

    const Eigen::Vector3d trans_12 = -R_21.transpose() * t_21;
    const Eigen::Vector3d bearing_2_in_1 = R_21.transpose() * point2 + trans_12 - trans_12;
    Eigen::Matrix<double, 3, 2> A;
    A(0,0) = point1.x();
    A(1,0) = point1.y();
    A(2,0) = point1.z();
    A(0,1) = -bearing_2_in_1.x();
    A(1,1) = -bearing_2_in_1.y();
    A(2,1) = -bearing_2_in_1.z();
    Eigen::Vector3d b = bearing_2_in_1 + trans_12 - point1;
    // 进行svd分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(A,
                                                 Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    // 构建SVD分解结果
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();

    // 构建S矩阵
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();

    for (unsigned int i = 0; i < D.size(); ++i) {

        if (D(i, 0) > 1e-6) {
            S(i, i) = 1 / D(i, 0);
        } else {
            S(i, i) = 0;
        }
    }

    // pinv_matrix = V * S * U^T
    Eigen::MatrixXd A_inv = V * S * U.transpose();

    // Eigen::Vector2d lambda = A.inverse() * b;
    // Eigen::Vector2d lambda = A_inv * b;
    Eigen::Vector2d lambda = (A.transpose() * A).inverse() * A.transpose() * b;
    // cout << "inverse A:" << endl;
    // cout << A_inv << endl << endl;
    // cout << (A.transpose() * A).inverse() * A.transpose() << endl << endl;

    const Eigen::Vector3d pt_1 = lambda(0) * point1;
    const Eigen::Vector3d pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
    return (pt_1 + pt_2) / 2.0;
}

Eigen::Vector3d Triangulate2ViewIDWM(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    const Eigen::Vector3d point1(p1.x, p1.y, p1.z);
    const Eigen::Vector3d point2(p2.x, p2.y, p2.z);
    Eigen::Vector3d Rp1 = R_21 * point1;
    const double p_norm = Rp1.cross(point2).norm();
    const double q_norm = Rp1.cross(t_21).norm();
    const double r_norm = point2.cross(t_21).norm();
    // Eq. (10)     这是在相机2的坐标系下，还要变换回相机1下
    Eigen::Vector3d triangulated = ( q_norm / (q_norm + r_norm) )
        * ( t_21 + (r_norm / p_norm) * (Rp1 + point2) );

    // Eq. (7)
    const Eigen::Vector3d lambda0_Rp1 = (r_norm / p_norm) * Rp1;
    const Eigen::Vector3d lambda1_p2 = (q_norm / p_norm) * point2;

    // Eq. (9) - test adequation
    if((t_21 + lambda0_Rp1 - lambda1_p2).squaredNorm() <  
        std::min(std::min(
        (t_21 + lambda0_Rp1 + lambda1_p2).squaredNorm(),
        (t_21 - lambda0_Rp1 - lambda1_p2).squaredNorm()),
        (t_21 - lambda0_Rp1 + lambda1_p2).squaredNorm()))
    {
        return R_21.transpose() * (triangulated - t_21);
    }
    else 
        return Eigen::Vector3d::Ones() * std::numeric_limits<double>::infinity();
}

// 最小化代数误差的三角化过程，从openMVG里抄来的，
// 当特征点进行了各向同性的正则化(isotropic normalization)之后，三角化的效果会更好
// 经过测试，效果确实好了一点
/**
 * [功能描述]：使用代数方法进行多视角三角化重建空间点
 * 该方法通过构建线性最小二乘问题，求解在多个相机视角约束下的最优三维点坐标
 * 相比于双视角方法，能够利用更多观测信息，提高重建精度和鲁棒性
 * @param [R_cw_list]：各个相机的旋转矩阵列表，表示从相机坐标系到世界坐标系的旋转
 * @param [t_cw_list]：各个相机的平移向量列表，表示从相机坐标系到世界坐标系的平移
 * @param [points]：同一空间点在各个相机坐标系下的归一化观测方向
 * @return：该点在世界坐标系下的三维坐标
 */
Eigen::Vector3d TriangulateNViewAlgebraic(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    // 确保至少有3个相机视角（多视角三角化的前提条件）
    assert(R_cw_list.size() > 2);
    
    // 初始化累积矩阵AtA，用于构建法方程组（normal equations）
    // 这是最小二乘法中 A^T * A * x = A^T * b 形式的系数矩阵
    Eigen::Matrix4d AtA = Eigen::Matrix4d::Zero();
    
    // 遍历每个相机视角，构建约束方程
    for(size_t i = 0; i < points.size(); i++)
    {
        // 将观测方向归一化，确保为单位向量
        // const Eigen::Vector3d point = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);  // 原始版本（已注释）
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z).normalized();
        
        // 构建相机投影矩阵 P = [R|t]，将世界坐标投影到相机坐标系
        // 这是一个3x4矩阵，前3列为旋转矩阵R，最后一列为平移向量t
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        
        // 构建约束矩阵cost
        // 数学原理：对于空间点P_w = [X, Y, Z, 1]^T，其在相机i中的投影应该与观测方向平行
        // 即：λ * point_norm = P * P_w，其中λ是深度（标量）
        // 重新整理：P * P_w - point_norm * (point_norm^T * P * P_w) = 0
        // 这等价于：(I - point_norm * point_norm^T) * P * P_w = 0
        // 因此：cost = P - point_norm * point_norm^T * P
        Eigen::Matrix<double, 3, 4> cost = pose - point_norm * point_norm.transpose() * pose;
        
        // 累积法方程组的系数矩阵：AtA += cost^T * cost
        // 这相当于将每个相机的约束方程加入到总的最小二乘问题中
        AtA += cost.transpose() * cost;
    }
    
    // 求解齐次线性方程组 AtA * x = 0
    // 使用特征值分解，最小特征值对应的特征向量即为解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
    
    // 获取最小特征值对应的特征向量（即齐次坐标下的解）
    Eigen::Vector4d p = eigen_solver.eigenvectors().col(0);
    
    // 将齐次坐标转换为欧几里得坐标（除以第4个分量进行归一化）
    // hnormalized()函数执行：[X, Y, Z, W]^T -> [X/W, Y/W, Z/W]^T
    Eigen::Vector3d point_world = p.hnormalized();
    
    // 检查特征值分解是否成功
    if(eigen_solver.info() == Eigen::Success)
        return point_world;  // 返回三角化结果
    else 
        // 如果特征值分解失败，返回无穷大向量表示失败
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
}

// 这是最常规的多视图三角化的方法，把输入的向量的z坐标都变成1，这样每个特征点可以提供两组约束
//             | x_1 * P_1^3 - P_1^1 |
//             | y_1 * P_1^3 - P_1^2 |      | X_1 |
//             | x_2 * P_2^3 - P_2^1 |      | X_2 |
//             | y_2 * P_2^3 - P_2^2 |   *  | X_3 |  =  0
//             | x_3 * P_3^3 - P_3^1 |      | X_4 |
//             | y_3 * P_3^3 - P_3^1 |
// 这里的 x_1 y_1 x_2 y_2 就代表每个特征点的第一维和第二维（因为第三维是1）   
// P_1^1 P_1^2 代表位姿矩阵的第一行，第二行。 P_1 = [R_1|t_1] 是第1张图像的位姿
Eigen::Vector3d TriangulateNView1(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    const size_t num_row = 2 * R_cw_list.size();
    Eigen::MatrixXd A(num_row, 4) ;
    for(size_t i = 0; i < points.size(); i++)
    {
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
        point_norm /= point_norm(2);
        // 把旋转平移组成3x4的矩阵 [R|t]
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        A.row(2 * i) = point_norm(0) * pose.row(2) - pose.row(0);
        A.row(2 * i + 1) = point_norm(1) * pose.row(2) - pose.row(1);
    }
    Eigen::Matrix<double, 4, 4> AtA = A.transpose() * A;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
    Eigen::Vector4d p = eigen_solver.eigenvectors().col(0);
    Eigen::Vector3d point_world = p.hnormalized();
    if(eigen_solver.info() == Eigen::Success)
        return point_world;
    else 
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
}

// 这是另一种三角化的方法，和上面的方法的区别就是没有把输入特征的z变为1，那么每一个特征可以多提供一个约束（每个特征3个约束）
// 但是相应的，需要求解的未知量也多了一个（三角化的三维点在特征点处投影的深度）
// 具体的推导过程见 公式推导.md
Eigen::Vector3d TriangulateNView2(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    const size_t num_views = points.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * num_views, 4 + num_views);
    for(size_t i = 0; i < points.size(); i++)
    {
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
        // 把旋转平移组成3x4的矩阵 [R|t]
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        A.block<3, 4>(3 * i, 0) = -pose;
        A.block<3, 1>(3 * i, 4 + i) = point_norm;
    }
    Eigen::VectorXd X(num_views + 4);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    X = svd.matrixV().col(A.cols() - 1);
    return X.head(4).hnormalized();
}

/**
 * [功能描述]：使用多个相机视角对空间中的同一个点进行三角化重建
 * 根据不同相机视角下观测到的对应点，计算出该点在世界坐标系下的三维坐标
 * @param [R_cw_list]：各个相机的旋转矩阵列表，表示从相机坐标系到世界坐标系的旋转
 * @param [t_cw_list]：各个相机的平移向量列表，表示从相机坐标系到世界坐标系的平移
 * @param [points]：同一空间点在各个相机坐标系下的观测点（归一化坐标，即相机坐标系下的方向向量）
 * @return：返回该点在世界坐标系下的三维坐标
 */
Eigen::Vector3d TriangulateNView(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    // 参数一致性检查：确保旋转矩阵、平移向量和观测点的数量一致
    assert(R_cw_list.size() == t_cw_list.size());
    assert(R_cw_list.size() == points.size());
    
    // 情况1：双视角三角化
    if(R_cw_list.size() == 2)
    {
        // 计算第二个相机相对于第一个相机的相对位姿
        // R_21表示从第一个相机坐标系到第二个相机坐标系的旋转
        // 推导：P_2 = R_2w * P_w = R_2w * R_w1 * P_1 = R_21 * P_1
        // 所以 R_21 = R_2w * R_w1 = R_cw_list[1] * R_cw_list[0].transpose()
        Eigen::Matrix3d R_21 = R_cw_list[1] * R_cw_list[0].transpose();
        
        // 计算第二个相机相对于第一个相机的相对平移
        // t_21表示第一个相机原点在第二个相机坐标系下的位置
        // 推导：t_2w = R_2w * t_w1 + t_21，其中 t_w1 = -R_w1 * t_1w
        // 化简得：t_21 = t_2w - R_21 * t_1w
        Eigen::Vector3d t_21 = t_cw_list[1] - R_21 * t_cw_list[0];
        
        // 使用双视角三角化方法计算点在第一个相机坐标系下的坐标
        Eigen::Vector3d point_triangulated = Triangulate2View(R_21, t_21, points[0], points[1]);
        
        // 将结果从第一个相机坐标系转换到世界坐标系
        // 第一个相机到世界坐标系的平移向量：t_w1 = -R_w1 * t_1w = -R_cw_list[0].transpose() * t_cw_list[0]
        Eigen::Vector3d t_w1 = - R_cw_list[0].transpose() * t_cw_list[0];
        
        // 坐标变换：P_w = R_wc * P_c + t_wc = R_cw_list[0].transpose() * point_triangulated + t_w1
        Eigen::Vector3d point_world = R_cw_list[0].transpose() * point_triangulated + t_w1;
        
        return point_world;
    }
    // 情况2：多视角三角化（超过两个相机视角）
    else if(R_cw_list.size() > 2)
    {
        // 使用代数方法进行多视角三角化，这是效果最好的方法
        // 注释中提到有三种方法，效果依次变差：
        // 1. TriangulateNViewAlgebraic - 代数方法（当前使用，效果最好）
        // 2. TriangulateNView2 - 方法2（已注释）
        // 3. TriangulateNView1 - 方法1（已注释，效果最差）
        Eigen::Vector3d point_triangulated = TriangulateNViewAlgebraic(R_cw_list, t_cw_list, points);
        
        // 多视角三角化直接返回世界坐标系下的结果，无需额外变换
        return point_triangulated;
    }
    // 错误情况：视角数量无效（少于2个）
    else 
    {
        // 记录错误日志
        LOG(ERROR) << "Invalid number in triangulate";
        
        // 返回无穷大向量表示三角化失败
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
    }
}