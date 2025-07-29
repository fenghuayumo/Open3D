// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <thread>
#include <future>
#include <deque>
#include <algorithm>
#include <fmt/format.h>
#include <unordered_set>
#include <parallel-hashmap/parallel_hashmap/phmap.h>
//#include <tbb/parallel_for.h>
#include "open3d/Open3D.h"
#include "mesh_utils.hpp"
#include "remesher/remesh_botsch.h"
#define USE_OPENMP 1
template <typename T>
void wait_all(T&& futures) {
    for (auto& f : futures) {
        f.get();
    }
}

template <typename T>
std::vector<T> wait_and_get_all(std::vector<std::future<T>>&& futures) {
    std::vector<T> res;
    for (auto& f : futures) {
        f.wait();
        res.push_back(f.get());
    }
    return res;
}

class ThreadPool {
public:
    ThreadPool() : ThreadPool{std::thread::hardware_concurrency()} {}
    ThreadPool(size_t max_num_threads, bool force = false) {
        if (!force) {
            max_num_threads =
                    std::min((size_t)std::thread::hardware_concurrency(),
                             max_num_threads);
        }
        start_threads(max_num_threads);
    }
    virtual ~ThreadPool() {
        wait_until_queue_completed();
        shutdown_threads(m_threads.size());
    }

    template <class F>
    auto enqueue_task(F&& f, bool high_priority = false)
            -> std::future<std::invoke_result_t<F>> {
        using return_type = std::invoke_result_t<F>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::forward<F>(f));

        auto res = task->get_future();

        {
            std::lock_guard<std::mutex> lock{m_task_queue_mutex};

            if (high_priority) {
                m_task_queue.emplace_front([task]() { (*task)(); });
            } else {
                m_task_queue.emplace_back([task]() { (*task)(); });
            }
        }

        m_worker_condition.notify_one();
        return res;
    }

    void start_threads(size_t num) {
        m_num_threads += num;
        for (size_t i = m_threads.size(); i < m_num_threads; ++i) {
            m_threads.emplace_back([this, i] {
                while (true) {
                    std::unique_lock<std::mutex> lock{m_task_queue_mutex};

                    // look for a work item
                    while (i < m_num_threads && m_task_queue.empty()) {
                        // if there are none, signal that the queue is completed
                        // and wait for notification of new work items.
                        m_task_queue_completed_condition.notify_all();
                        m_worker_condition.wait(lock);
                    }

                    if (i >= m_num_threads) {
                        break;
                    }

                    std::function<void()> task{move(m_task_queue.front())};
                    m_task_queue.pop_front();

                    // Unlock the lock, so we can process the task without
                    // blocking other threads
                    lock.unlock();

                    task();
                }
            });
        }
    }
    void shutdown_threads(size_t num) {
        auto num_to_close = std::min(num, m_num_threads);

        {
            std::lock_guard<std::mutex> lock{m_task_queue_mutex};
            m_num_threads -= num_to_close;
        }

        // Wake up all the threads to have them quit
        m_worker_condition.notify_all();
        for (auto i = 0u; i < num_to_close; ++i) {
            m_threads.back().join();
            m_threads.pop_back();
        }
    }
    void set_n_threads(size_t num) {
        if (m_num_threads > num) {
            shutdown_threads(m_num_threads - num);
        } else if (m_num_threads < num) {
            start_threads(num - m_num_threads);
        }
    }

    void wait_until_queue_completed() {
        std::unique_lock<std::mutex> lock{m_task_queue_mutex};
        m_task_queue_completed_condition.wait(
                lock, [this]() { return m_task_queue.empty(); });
    }
    void flush_queue() {
        std::lock_guard<std::mutex> lock{m_task_queue_mutex};
        m_task_queue.clear();
    }

    template <typename Int, typename F>
    void parallel_for_async(Int start,
                            Int end,
                            F body,
                            std::vector<std::future<void>>& futures) {
        Int local_num_threads = (Int)m_num_threads;

        Int range = end - start;
        Int chunk = (range / local_num_threads) + 1;

        for (Int i = 0; i < local_num_threads; ++i) {
            futures.emplace_back(enqueue_task([i, chunk, start, end, body] {
                Int inner_start = start + i * chunk;
                Int inner_end = std::min(end, start + (i + 1) * chunk);
                for (Int j = inner_start; j < inner_end; ++j) {
                    body(j);
                }
            }));
        }
    }

    template <typename Int, typename F>
    std::vector<std::future<void>> parallel_for_async(Int start,
                                                      Int end,
                                                      F body) {
        std::vector<std::future<void>> futures;
        parallel_for_async(start, end, body, futures);
        return futures;
    }

    template <typename Int, typename F>
    void parallel_for(Int start, Int end, F body) {
        wait_all(parallel_for_async(start, end, body));
    }

private:
    size_t m_num_threads = 0;
    std::vector<std::thread> m_threads;

    std::deque<std::function<void()>> m_task_queue;
    std::mutex m_task_queue_mutex;
    std::condition_variable m_worker_condition;
    std::condition_variable m_task_queue_completed_condition;
};
template <typename Int, typename F>
void parallel_for(Int start, Int end, F body) {
#ifdef USE_OPENMP
#pragma omp parallel for
    for (int i = start; i < end; i++) {
        body(i);
    }
#else
    static ThreadPool pool;
    return pool.parallel_for(start, end, body);
#endif
}

namespace o3d {

bool check_non_manifold(const open3d::geometry::TriangleMesh& mesh) {
    auto non_manifold_verts = mesh.GetNonManifoldVertices();
    auto non_manifold_edges = mesh.GetNonManifoldEdges();
    auto self_intersecting = mesh.GetSelfIntersectingTriangles();
    auto l = non_manifold_verts.size() + non_manifold_edges.size() + self_intersecting.size();

    return l > 0;
}

std::vector<Eigen::Vector3d> matrix_to_vertices(const Eigen::MatrixXd& matrix) {
    if (matrix.cols() != 3) {
        throw std::invalid_argument(
                "Matrix must have 3 columns for 3D vertices");
    }
    std::vector<Eigen::Vector3d> vertices;
    vertices.reserve(matrix.rows());
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        vertices.emplace_back(matrix.row(i).transpose());
    }
    return vertices;
}

std::vector<Eigen::Vector3i> matrix_to_faces(const Eigen::MatrixXi& matrix) {
    if (matrix.cols() != 3) {
        throw std::invalid_argument(
                "Matrix must have 3 columns for triangle faces");
    }
    std::vector<Eigen::Vector3i> faces;
    faces.reserve(matrix.rows());
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        faces.emplace_back(matrix.row(i).transpose());
    }
    return faces;
}

Eigen::MatrixXd vertices_to_matrix(
        const std::vector<Eigen::Vector3d>& vertices) {
    Eigen::MatrixXd matrix(vertices.size(), 3);

    for (size_t i = 0; i < vertices.size(); ++i) {
        matrix.row(i) = vertices[i].transpose();
    }
    return matrix;
}

Eigen::MatrixXi faces_to_matrix(const std::vector<Eigen::Vector3i>& faces) {
    Eigen::MatrixXi matrix(faces.size(), 3);

    for (size_t i = 0; i < faces.size(); ++i) {
        matrix.row(i) = faces[i].transpose();
    }
    return matrix;
}

/**
 * 修复后的平均边长计算函数
 */
double average_edge_length(const Eigen::MatrixXd& vertices,
                           const Eigen::MatrixXi& faces) {
    // 检查输入维度
    if (vertices.cols() != 3) {
        throw std::invalid_argument("vertices must have 3 columns");
    }
    if (faces.cols() != 3) {
        throw std::invalid_argument("faces must have 3 columns");
    }

    // 使用set来存储唯一的边
    std::set<std::pair<int, int>> unique_edges;

    // 1. 提取所有边并去重
    for (int i = 0; i < faces.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v0_idx = faces(i, j);
            int v1_idx = faces(i, (j + 1) % 3);

            // 确保索引在有效范围内
            if (v0_idx < 0 || v0_idx >= vertices.rows() || v1_idx < 0 ||
                v1_idx >= vertices.rows()) {
                throw std::out_of_range("Vertex index out of range");
            }

            // 确保较小的索引在前
            if (v0_idx > v1_idx) std::swap(v0_idx, v1_idx);
            unique_edges.insert(std::make_pair(v0_idx, v1_idx));
        }
    }

    // 2. 计算边长
    double total_length = 0.0;
    int edge_count = 0;

    for (const auto& edge : unique_edges) {
        int v0_idx = edge.first;
        int v1_idx = edge.second;

        // 正确获取顶点坐标
        Eigen::Vector3d v0 = vertices.row(v0_idx);
        Eigen::Vector3d v1 = vertices.row(v1_idx);

        total_length += (v1 - v0).norm();
        edge_count++;
    }

    if (edge_count == 0) {
        throw std::runtime_error("No edges found in the mesh");
    }

    // 3. 返回平均边长
    return total_length / edge_count;
}

/**
 * 使用Eigen优化的版本
 */
double average_edge_length_eigen(const Eigen::MatrixXd& vertices,
                                 const Eigen::MatrixXi& faces) {
    // 检查输入维度
    if (vertices.cols() != 3) {
        throw std::invalid_argument("vertices must have 3 columns");
    }
    if (faces.cols() != 3) {
        throw std::invalid_argument("faces must have 3 columns");
    }

    // 使用set来存储唯一的边
    std::set<std::pair<int, int>> unique_edges;

    // 1. 提取所有边并去重
    for (int i = 0; i < faces.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v0 = faces(i, j);
            int v1 = faces(i, (j + 1) % 3);

            // 确保较小的索引在前
            if (v0 > v1) std::swap(v0, v1);
            unique_edges.insert(std::make_pair(v0, v1));
        }
    }

    // 2. 使用Eigen的向量化操作计算边长
    Eigen::VectorXd lengths(unique_edges.size());
    int idx = 0;

    for (const auto& edge : unique_edges) {
        const Eigen::Vector3d& v0 = vertices.row(edge.first);
        const Eigen::Vector3d& v1 = vertices.row(edge.second);
        lengths(idx++) = (v1 - v0).norm();
    }

    // 3. 返回平均边长
    return lengths.mean();
} 

struct Edge {
    uint64_t e;
    Edge(int v0, int v1) {
        if (v0 > v1) std::swap(v0, v1);
        e = (static_cast<uint64_t>(v0) << 32) | v1;
    }
    bool operator==(const Edge& o) const { return e == o.e; }
};

struct EdgeHash {
    size_t operator()(const Edge& e) const {
        return phmap::Hash<uint64_t>{}(e.e);
    }
};

std::vector<int> boundary_vertices_optimized(const Eigen::MatrixXi& F) {
    phmap::parallel_flat_hash_set<Edge, EdgeHash> edges;
    phmap::parallel_flat_hash_set<Edge, EdgeHash> boundary_edges;
    std::mutex mtx;  // 用于同步的互斥锁

    // 并行处理面片
#pragma omp parallel for
    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            Edge edge(F(i, j), F(i, (j + 1) % 3));

#pragma omp critical  // 保护共享数据结构的插入/删除操作
            {
                if (edges.contains(edge)) {
                    boundary_edges.erase(edge);
                } else {
                    edges.insert(edge);
                    boundary_edges.insert(edge);
                }
            }
        }
    }

    // 将边界边转换为向量以便并行处理
    std::vector<Edge> boundary_edges_vec(boundary_edges.begin(),
                                    boundary_edges.end());

    phmap::parallel_flat_hash_set<int> boundary_verts;

    // 并行处理边界边
#pragma omp parallel for
    for (size_t i = 0; i < boundary_edges_vec.size(); ++i) {
        const Edge& e = boundary_edges_vec[i];
#pragma omp critical  // 保护对boundary_verts的插入操作
        {
            boundary_verts.insert(static_cast<int>(e.e >> 32));  // 高位是v0
            boundary_verts.insert(
                    static_cast<int>(e.e & 0xFFFFFFFF));  // 低位是v1
        }
    }

    return std::vector<int>(boundary_verts.begin(), boundary_verts.end());
}

// 优化的顶点重排序
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> reorder_vertices_optimized(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<int>& features) {
    const int n_vertices = V.rows();
    std::vector<int> index_map(n_vertices);

    // 标记特征顶点
    std::vector<bool> is_feature(n_vertices, false);
    parallel_for(0, (int)features.size(),
                      [&](int i) { is_feature[features[i]] = true; });

    // 并行构建新顺序
    std::vector<int> new_order(n_vertices);
    int feature_count = features.size();

    // 特征顶点在前
    parallel_for(0, feature_count, [&](int i) {
        new_order[i] = features[i];
        index_map[features[i]] = i;
    });

    // 其他顶点在后
    int non_feature_pos = feature_count;
    for (int v = 0; v < n_vertices; v++) {
        if (!is_feature[v]) {
            new_order[non_feature_pos] = v;
            index_map[v] = non_feature_pos++;
        }
    }

    // 并行重排顶点
    Eigen::MatrixXd new_V(n_vertices, 3);
    parallel_for(0, n_vertices,
                      [&](int i) { new_V.row(i) = V.row(new_order[i]); });

    // 并行更新面索引
    Eigen::MatrixXi new_F = F;
    int face_count = F.rows();
    parallel_for(0, face_count, [&](int i) {
        for (int j = 0; j < 3; j++) {
            new_F(i, j) = index_map[F(i, j)];
        }
    });

    return {new_V, new_F};
}
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
    }
};
// 非流形边检测
std::vector<std::pair<int, int>> non_manifold_edges(const Eigen::MatrixXi& F) {
    std::unordered_map<std::pair<int, int>, int, PairHash> edge_counts;
    std::vector<std::pair<int, int>> non_manifolds;

    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            int v0 = F(i, j);
            int v1 = F(i, (j + 1) % 3);
            if (v0 > v1) std::swap(v0, v1);
            auto edge = std::make_pair(v0, v1);
            edge_counts[edge]++;
        }
    }

    for (const auto& [edge, count] : edge_counts) {
        if (count > 2) {  // 正常边应该被2个面共享
            non_manifolds.push_back(edge);
        }
    }

    return non_manifolds;
}

// 合并并去重特征顶点，保持原始顺序
std::vector<int> merge_and_unique_features(const std::vector<int>& features,
                                           const std::vector<int>& boundaries) {
    std::vector<int> merged = features;
    std::unordered_set<int> feature_set(features.begin(), features.end());

    for (int v : boundaries) {
        if (feature_set.find(v) == feature_set.end()) {
            merged.push_back(v);
            feature_set.insert(v);
        }
    }

    return merged;
}
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> remesh_botsch_f(
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& F,
        int num_iterations,
        double target_length,
        const std::vector<int>& input_features = {},
        bool project=true) {
    // 1. 检查特征顶点是否唯一
    std::vector<int> features = input_features;
    if (!features.empty()) {
        sort(features.begin(), features.end());
        auto last = unique(features.begin(), features.end());
        if (last != features.end()) {
            std::cerr << "Warning: Feature array is not unique. Using unique "
                    "entries."
                      << std::endl;
            features.erase(last, features.end());
        }
    }

    // 2. 合并特征顶点和边界顶点
    auto boundaries = boundary_vertices_optimized(F);
    features = merge_and_unique_features(features, boundaries);

    // 3. 重新排序顶点
    auto [new_V, new_F] = reorder_vertices_optimized(V, F, features);

    // 4. 检查是否所有顶点都是特征/边界顶点
    if (features.size() == V.rows()) {
        std::cerr << "Warning: All vertices are feature/boundary vertices. "
                "Operation may have no effect."
                  << std::endl;
    }

    // 5. 检查非流形边
    auto nm_edges = non_manifold_edges(F);
    if (!nm_edges.empty()) {
        throw std::runtime_error("Input mesh is non-manifold.");
    }

    // 6. 这里应该调用实际的remesh_botsch实现
    Eigen::VectorXi eigen_feature =
            Eigen::Map<Eigen::VectorXi>(features.data(), features.size());
    remesh_botsch(V, F, target_length, num_iterations, eigen_feature,
     project);

    // 返回结果 (这里只是示例，实际应该返回remesh后的结果)
    return {V, F};
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>> remesh(
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<Eigen::Vector3i>& faces,
        int target_face_num) {
    Eigen::MatrixXd V = vertices_to_matrix(vertices);
    Eigen::MatrixXi F = faces_to_matrix(faces);
    double avg_len = average_edge_length(V, F);
    int cur_face_num = faces.size();

    double ratio = static_cast<double>(cur_face_num) / target_face_num * 0.9;
    double target_len = std::sqrt(ratio) * avg_len;

    std::cout << "[INFO] remesh: " << vertices.size() << " vertices, "
              << faces.size() << " faces --> target: " << target_face_num
              << " faces, target edge length: " << target_len << std::endl;
    
    remesh_botsch_f(V, F, 4, target_len);
    auto retV = matrix_to_vertices(V);
    auto retF = matrix_to_faces(F);
    return std::make_pair(std::move(retV), std::move(retF));
}

//double compute_avg_spacing(const std::vector<double>& points,
//                         int k = 5) {
//    std::vector<Eigen::Vector3d> o3d_vertices(points.size() / 3);
//    memcpy(o3d_vertices.data(), points.data(), points.size() * sizeof(double));
//    // 构建KDTree加速近邻搜索
//    open3d::geometry::KDTreeFlann kdtree;
//    open3d::geometry::PointCloud pcd;
//    pcd.points_ = o3d_vertices;
//    kdtree.SetGeometry(pcd);
//
//    double total_distance = 0.0;
//    int valid_neighbors = 0;
//
//    // 遍历每个点，计算k近邻平均距离
//    for (size_t i = 0; i < points.size(); ++i) {
//        std::vector<int> indices;
//        std::vector<double> distances;
//        if (kdtree.SearchKNN(points[i], k + 1, indices, distances) >
//            1) {  // 包含自身点
//            for (size_t j = 1; j < distances.size();
//                 ++j) {  // 跳过自身（距离=0）
//                total_distance += std::sqrt(distances[j]);
//                valid_neighbors++;
//            }
//        }
//    }
//
//    return valid_neighbors > 0 ? total_distance / valid_neighbors : 0.0;
//}

 std::vector<float> filter_outlier_points(const std::vector<double>& points,
                                        int nb_points,
                                        float radius) {
    std::vector<Eigen::Vector3d> o3d_vertices(points.size() / 3);
    memcpy(o3d_vertices.data(), points.data(), points.size() * sizeof(double));
    open3d::geometry::PointCloud pcd;
    pcd.points_ = o3d_vertices;
    auto [filtered_pcd, indices] = pcd.RemoveRadiusOutliers(nb_points, radius);
    std::vector<float> verts(filtered_pcd->points_.size() * 3);
    for (auto i = 0; i < filtered_pcd->points_.size(); i++) {
        verts[i * 3] = filtered_pcd->points_[i].x();
        verts[i * 3 + 1] = filtered_pcd->points_[i].y();
        verts[i * 3 + 2] = filtered_pcd->points_[i].z();
    }
    return verts;
 }

 O3D_API double compute_avg_spacing(
                const double* points,
                int num_pts,
                int nb_points,
                float radius, 
                int k) {
     std::vector<Eigen::Vector3d> o3d_vertices(num_pts);
     memcpy(o3d_vertices.data(), points, num_pts * sizeof(Eigen::Vector3d));
     // 构建KDTree加速近邻搜索
     open3d::geometry::PointCloud pcd;
     pcd.points_ = o3d_vertices;
     auto [filtered_pcd, indices] = pcd.RemoveRadiusOutliers(nb_points, radius);
     // Check if we have any points left after filtering
     if (filtered_pcd->points_.empty()) {
         std::cerr
                 << "Warning: All points filtered out, using original points\n";
         filtered_pcd = std::make_shared<open3d::geometry::PointCloud>(pcd);
     }
     auto temp_pcd = *filtered_pcd;  // Make a copy to ensure data stability
     std::cout << "filter pt size: " << temp_pcd.points_.size()
               << std::endl;
     open3d::geometry::KDTreeFlann kdtree;
     kdtree.SetGeometry(temp_pcd);
     double total_distance = 0.0;
     int valid_neighbors = 0;
     std::cout << "calc neighbor dist " << std::endl;
     // 遍历每个点，计算k近邻平均距离
     auto numpts = filtered_pcd->points_.size();
     for (size_t i = 0; i < filtered_pcd->points_.size(); ++i) {
         std::vector<int> indices;
         std::vector<double> distances;
         if (kdtree.SearchKNN(filtered_pcd->points_[i], k + 1, indices,
                              distances) >
             1) {  // 包含自身点
             for (size_t j = 1; j < distances.size();
                  ++j) {  // 跳过自身（距离=0）
                 total_distance += std::sqrt(distances[j]);
                 valid_neighbors++;
             }
         }
     }

     return valid_neighbors > 0 ? total_distance / valid_neighbors : 0.0;
 }

void TSDF::export_mesh(
    const std::string& filePath,
    const std::vector<double*>& viewMatrixs,
    const std::vector<CameraIntrincs> intrinsics,
    const std::vector<uint8_t*> rgbs,
    const std::vector<float*>& depths,
    float voxel_length,
    int target_triangles,
    int target_tex_resolution,
    float sdf_trunc,
    float depth_trunc,
    int threshold_ntri)
{
    auto volume = open3d::pipelines::integration::ScalableTSDFVolume(
        voxel_length,
        sdf_trunc,
        open3d::pipelines::integration::TSDFVolumeColorType::RGB8
    );
    try {
        std::cout << "start marching cube " << std::endl;
        auto num_imgs = viewMatrixs.size();
        std::vector<open3d::t::geometry::Image> t_images(num_imgs);
        std::vector<open3d::core::Tensor> t_intrics(num_imgs);
        std::vector<open3d::core::Tensor> t_extrincs(num_imgs);
        for (auto i = 0; i < viewMatrixs.size(); i++) {
            const auto& viewMatData = viewMatrixs[i];
            open3d::geometry::Image depth_o3d;
            open3d::geometry::Image img_o3d;
            const auto fx_cx = intrinsics[i];
            auto width = fx_cx.width;
            auto height = fx_cx.height;
            img_o3d.Prepare(width,height,3,1);
            depth_o3d.Prepare(width, height,1, 4);
            memcpy(img_o3d.data_.data(), rgbs[i], width * height * 3);
            memcpy(depth_o3d.data_.data(), depths[i], width * height * sizeof(float));
            auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(img_o3d, depth_o3d, 1,depth_trunc,false);
            auto intrinc = open3d::camera::PinholeCameraIntrinsic(
                    fx_cx.width, fx_cx.height, fx_cx.fx, fx_cx.fy, fx_cx.cx,
                    fx_cx.cy);
            Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
            extrinsic << viewMatData[0], viewMatData[1],viewMatData[2], viewMatData[3],
                        viewMatData[4], viewMatData[5],viewMatData[6], viewMatData[7],
                        viewMatData[8], viewMatData[9],viewMatData[10], viewMatData[11],
                        viewMatData[12], viewMatData[13],viewMatData[14], viewMatData[15];
            volume.Integrate(*rgbd,intrinc,extrinsic);
            t_images[i] = open3d::t::geometry::Image::FromLegacy(img_o3d);
            t_intrics[i] = open3d::core::Tensor(std::vector<double>{fx_cx.fx,0,fx_cx.cx,0,fx_cx.fy,fx_cx.cy,0,0,1}, {3, 3},
                open3d::core::Dtype::Float64);
            t_extrincs[i] = open3d::core::Tensor(viewMatData, {4, 4},open3d::core::Dtype::Float64);
        }
        auto mesh = volume.ExtractTriangleMesh(progress_);
        mesh->Scale(1,{0,0,0});
        std::cout << "marching cube finished" << std::endl;
        if (threshold_ntri > 0) {
            //open3d::utility::VerbosityContextManager
            auto [triangle_clusters, cluster_n_triangles, cluster_area] =
                    mesh->ClusterConnectedTriangles();
           auto clean_mesh = *mesh;
           std::vector<bool> triangles_to_remove(triangle_clusters.size());
           parallel_for<size_t>(
                   0, triangle_clusters.size(), [&](size_t tri_cluster_id) {
               triangles_to_remove[tri_cluster_id] =
                               cluster_n_triangles
                                       [triangle_clusters[tri_cluster_id]] <
                               threshold_ntri;
           });
           clean_mesh.RemoveTrianglesByMask(triangles_to_remove);
           clean_mesh.RemoveUnreferencedVertices();
           clean_mesh.MergeCloseVertices(1e-5);
           clean_mesh.RemoveDegenerateTriangles();
           clean_mesh.RemoveDuplicatedVertices();
           clean_mesh.RemoveDuplicatedTriangles();
           clean_mesh.RemoveNonManifoldEdges();
           auto non_v = clean_mesh.GetNonManifoldVertices();
           std::vector<size_t> non_vt(non_v.begin(), non_v.end());
           clean_mesh.RemoveVerticesByIndex(non_vt);
           progress_ = 0.55f;
           std::cout << "post process mesh " << std::endl;
         /*  auto smooth_mesh =
                   clean_mesh.FilterSmoothTaubin(5);
           clean_mesh = *smooth_mesh->SimplifyQuadricDecimation(
                   target_triangles, std::numeric_limits<double>::infinity(),1.0f);*/
           auto [vertices,triangles] = remesh(clean_mesh.vertices_,clean_mesh.triangles_,target_triangles);
           std::cout << "remeshed " << std::endl;
           clean_mesh = open3d::geometry::TriangleMesh(vertices,triangles);
           clean_mesh.RemoveUnreferencedVertices();
           clean_mesh.MergeCloseVertices(1e-5);
           clean_mesh.RemoveDegenerateTriangles();
           clean_mesh.RemoveDuplicatedVertices();
           clean_mesh.RemoveDuplicatedTriangles();
           clean_mesh.RemoveNonManifoldEdges();
           auto non_vi = clean_mesh.GetNonManifoldVertices();
           std::vector<size_t> non_vti(non_vi.begin(), non_vi.end());
           clean_mesh.RemoveVerticesByIndex(non_vti);
           progress_ = 0.75f;
           //clean_mesh.OrientTriangles();
           std::cout << "compute uv " << std::endl;
           if (std::filesystem::path(filePath).extension() == ".obj") {
                //compute uv
               //if (!check_non_manifold(clean_mesh)) {
               
                   auto tmesh = open3d::t::geometry::TriangleMesh::FromLegacy(
                           clean_mesh);
                   tmesh.ComputeUVAtlas(target_tex_resolution,1,0.33,8);
                   auto img = tmesh.ProjectImagesToAlbedo(
                           t_images, t_intrics, t_extrincs,
                           target_tex_resolution);
                   clean_mesh = tmesh.ToLegacy();
                   auto albedo = *img.ToLegacy().FlipVertical();
                   clean_mesh.textures_.push_back(albedo);
                   //auto albedo_path = std::filesystem::path(filePath).parent_path() /
                   //        "albedo.png";
                   //open3d::io::WriteImage(albedo_path.string(), img.ToLegacy());
                   open3d::io::WriteTriangleMesh(filePath, clean_mesh);
                   progress_ = 1.0f;
                   return;
               //}
           }
           open3d::io::WriteTriangleMesh(filePath, clean_mesh);
           return;
        }
        mesh->ComputeVertexNormals();
        progress_ = 1.0f;
        open3d::io::WriteTriangleMesh(filePath,*mesh);
    }
    catch(std::exception& e) {
        throw std::runtime_error(fmt::format("Marching Cube Failed: {}", e.what()));
    }
}

O3D_API std::array<Vec3, 8> generate_obj_bound(
        const std::vector<double>& rgbs, 
        const std::vector<double>& xyzs,
        bool use_aabb,
        size_t nb_radius,
        float radius) {
        open3d::geometry::PointCloud pcd;
        std::vector<Eigen::Vector3d> o3d_vertices(xyzs.size() / 3);
        std::vector<Eigen::Vector3d> o3d_colors(rgbs.size() / 3);
        memcpy(o3d_vertices.data(), xyzs.data(), xyzs.size() * sizeof(double));
        memcpy(o3d_colors.data(), rgbs.data(), rgbs.size() * sizeof(double));
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        auto [filter_pcd,_] = pcd.RemoveRadiusOutliers(nb_radius,radius);
        std::array<Vec3,8> bd_points = {};
        if (use_aabb) {
            auto aabb = filter_pcd->GetAxisAlignedBoundingBox();
            auto aabb_center = aabb.GetCenter();
            auto aabb_extent = aabb.GetExtent();
            std::array<Eigen::Vector3d, 8> local_points = {
                    Eigen::Vector3d{-1, -1, -1},
                    {-1, -1, 1},
                    {-1, 1, -1},
                    {-1, 1, 1},
                    {1, -1, -1},
                    {1, -1, 1},
                    {1, 1, -1},
                    {1, 1, 1}};
            for (int i = 0; i < 8; i++) {
                auto global_pt = aabb_center + local_points[i].cwiseProduct(aabb_extent / 2.0);
                bd_points[i].x = global_pt.x();
                bd_points[i].y = global_pt.y();
                bd_points[i].z = global_pt.z();
            }
        } else {
            auto oobb = filter_pcd->GetOrientedBoundingBox();
            auto obb_center = oobb.GetCenter();
            auto obb_extent = oobb.extent_;
            auto obb_rot = oobb.R_;
            std::array<Eigen::Vector3d, 8> local_points = {
                    Eigen::Vector3d{-1, -1, -1},
                    {-1, -1, 1},
                {-1, 1, -1 },
                { -1, 1, 1 },
                { 1, -1, -1 },
                { 1, -1, 1 },
                { 1, 1, -1 },
                { 1, 1, 1 }
            };
            for (int i = 0; i < 8; i++) {
                auto global_pt =
                        obb_center + (obb_rot.transpose()  * local_points[i])
                                             .cwiseProduct(obb_extent / 2.0);
                bd_points[i].x = global_pt.x();
                bd_points[i].y = global_pt.y();
                bd_points[i].z = global_pt.z();
            }
        }
        return bd_points;
   }
    
   std::tuple<uint8_t*, float*, size_t> uniform_down_sample_points(
        const std::vector<uint8_t>& rgbs,
        const std::vector<float>& xyzs,
        size_t every_k) {
       static std::vector<uint8_t> new_rgbs;
       static std::vector<float> new_xyzs;
       if (xyzs.size() <= 0) {
           new_xyzs.clear();
           new_rgbs.clear();
           return {};
       }
        std::vector<float> valid_xyzs;
        std::vector<uint8_t> valid_rgbs;
        size_t num = xyzs.size() / 3;
        open3d::geometry::PointCloud pcd;
        std::vector<Eigen::Vector3d> o3d_vertices(num);
        std::vector<Eigen::Vector3d> o3d_colors(num);
        parallel_for<size_t>(0, num, [&](size_t i) {
            o3d_colors[i] = Eigen::Vector3d(rgbs[i * 3] / 255.0f,
                                            rgbs[i * 3 + 1] / 255.0f,
                                            rgbs[i * 3 + 2] / 255.0f);
            o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3], xyzs[i * 3 + 1],
                                              xyzs[i * 3 + 2]);
        });
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        auto sparse_pcd = pcd.UniformDownSample(every_k);
        auto spcd_size = sparse_pcd->points_.size();
        new_rgbs.resize(spcd_size * 3);
        new_xyzs.resize(spcd_size * 3);

        parallel_for<size_t>(0, spcd_size, [&](size_t i) { 
            new_xyzs[i * 3] = sparse_pcd->points_[i].x();
            new_xyzs[i * 3 + 1] = sparse_pcd->points_[i].y();
            new_xyzs[i * 3 + 2] = sparse_pcd->points_[i].z();

            auto& color = sparse_pcd->colors_[i];
            new_rgbs[i * 3] = uint8_t(
                    std::round(std::min(1., std::max(0., color.x())) * 255.));
            new_rgbs[i * 3 + 1] = uint8_t(
                    std::round(std::min(1., std::max(0., color.y())) * 255.));
            new_rgbs[i * 3 + 2] = uint8_t(
                    std::round(std::min(1., std::max(0., color.z())) * 255.));
        });
        return { new_rgbs.data(), new_xyzs.data(), new_xyzs.size()};
    }
    void uniform_down_sample_points_to_file(
                const std::string& filePath,
                const std::vector<uint8_t>& rgbs,
                const std::vector<float>& xyzs,
                size_t every_k,
                bool to_blender_coord) {
        
        open3d::geometry::PointCloud pcd;
        size_t num = xyzs.size() / 3;
        std::vector<Eigen::Vector3d> o3d_vertices(num);
        std::vector<Eigen::Vector3d> o3d_colors(num);
        parallel_for<size_t>(0, num, [&](size_t i) { 
            o3d_colors[i] = Eigen::Vector3d(rgbs[i * 3] / 255.0f, rgbs[i * 3 + 1] / 255.0f,
                                            rgbs[i * 3 + 2] / 255.0f);
            if (to_blender_coord) {
                o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3],
                                                  xyzs[i * 3 + 2],
                                                  -xyzs[i * 3 + 1]);
            } else {
                o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3], 
                                                  xyzs[i * 3 + 1],
                                                  xyzs[i * 3 + 2]);
            }
        });
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        try {
            auto sparse_pcd = pcd.UniformDownSample(every_k);
            open3d::io::WritePointCloud(filePath, *sparse_pcd);
        } catch (...) {
            std::cout << "error " << std::endl;
        }
    }

       std::tuple<uint8_t*, float*, size_t> voxel_down_sample_points(
            const std::vector<uint8_t>& rgbs,
            const std::vector<float>& xyzs,
            double voxel_size) {
        static std::vector<uint8_t> new_rgbs;
        static std::vector<float> new_xyzs;
        if (xyzs.size() <= 0) {
            new_xyzs.clear();
            new_rgbs.clear();
            return {};
        }
        std::vector<float> valid_xyzs;
        std::vector<uint8_t> valid_rgbs;
        size_t num = xyzs.size() / 3;
        open3d::geometry::PointCloud pcd;
        std::vector<Eigen::Vector3d> o3d_vertices(num);
        std::vector<Eigen::Vector3d> o3d_colors(num);
        parallel_for<size_t>(0, num, [&](size_t i) {
            o3d_colors[i] = Eigen::Vector3d(rgbs[i * 3] / 255.0f,
                                            rgbs[i * 3 + 1] / 255.0f,
                                            rgbs[i * 3 + 2] / 255.0f);
            o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3], xyzs[i * 3 + 1],
                                              xyzs[i * 3 + 2]);
        });
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        auto sparse_pcd = pcd.VoxelDownSample(voxel_size);
        auto spcd_size = sparse_pcd->points_.size();
        new_rgbs.resize(spcd_size * 3);
        new_xyzs.resize(spcd_size * 3);

        parallel_for<size_t>(0, spcd_size, [&](size_t i) {
            new_xyzs[i * 3] = sparse_pcd->points_[i].x();
            new_xyzs[i * 3 + 1] = sparse_pcd->points_[i].y();
            new_xyzs[i * 3 + 2] = sparse_pcd->points_[i].z();

            auto& color = sparse_pcd->colors_[i];
            new_rgbs[i * 3] = uint8_t(
                    std::round(std::min(1., std::max(0., color.x())) * 255.));
            new_rgbs[i * 3 + 1] = uint8_t(
                    std::round(std::min(1., std::max(0., color.y())) * 255.));
            new_rgbs[i * 3 + 2] = uint8_t(
                    std::round(std::min(1., std::max(0., color.z())) * 255.));
        });
        return {new_rgbs.data(), new_xyzs.data(), new_xyzs.size()};
    }
    void voxel_down_sample_points_to_file(const std::string& filePath,
                                            const std::vector<uint8_t>& rgbs,
                                            const std::vector<float>& xyzs,
                                          double voxel_size,
                                            bool to_blender_coord) {
        open3d::geometry::PointCloud pcd;
        size_t num = xyzs.size() / 3;
        std::vector<Eigen::Vector3d> o3d_vertices(num);
        std::vector<Eigen::Vector3d> o3d_colors(num);
        parallel_for<size_t>(0, num, [&](size_t i) {
            o3d_colors[i] = Eigen::Vector3d(rgbs[i * 3] / 255.0f,
                                            rgbs[i * 3 + 1] / 255.0f,
                                            rgbs[i * 3 + 2] / 255.0f);
            if (to_blender_coord) {
                o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3], xyzs[i * 3 + 2],
                                                  -xyzs[i * 3 + 1]);
            } else {
                o3d_vertices[i] = Eigen::Vector3d(xyzs[i * 3], xyzs[i * 3 + 1],
                                                  xyzs[i * 3 + 2]);
            }
        });
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        try {
            auto sparse_pcd = pcd.VoxelDownSample(voxel_size);
            auto bdbox = sparse_pcd->GetAxisAlignedBoundingBox();
            auto diagonal_length = bdbox.GetExtent().norm();
            if (sparse_pcd->points_.size() >= 500000) {
                //get the bounding box of the point cloud
                sparse_pcd = sparse_pcd->VoxelDownSample(diagonal_length * 0.005);
            } else if (sparse_pcd->points_.size() >= 300000) {
                sparse_pcd =
                        sparse_pcd->VoxelDownSample(diagonal_length * 0.003);
            } else if (sparse_pcd->points_.size() >= 150000) {
                sparse_pcd =
                        sparse_pcd->VoxelDownSample(diagonal_length * 0.0015);
            }
            open3d::io::WritePointCloud(filePath, *sparse_pcd);
        } catch (...) {
            std::cout << "error " << std::endl;
        }
    }
    void write_point_cloud(
        const std::string& filePath,
        const std::vector<float>& points_xyz,
        const std::vector<uint8_t>&
        points_rgb) {
        open3d::geometry::PointCloud pcd;
        std::vector<Eigen::Vector3d> o3d_vertices(points_xyz.size() / 3);
        std::vector<Eigen::Vector3d> o3d_colors(points_rgb.size() / 3);
        parallel_for<size_t>(0, o3d_colors.size(), [&](size_t i) {
            o3d_colors[i] = Eigen::Vector3d(points_rgb[i * 3] / 255.0f,
                                            points_rgb[i * 3 + 1] / 255.0f,
                                            points_rgb[i * 3 + 2] / 255.0f);
            o3d_vertices[i] = Eigen::Vector3d(points_xyz[i * 3], points_xyz[i * 3 + 1],
                                    points_xyz[i * 3 + 2]);
        });
        pcd.points_ = o3d_vertices;
        pcd.colors_ = o3d_colors;
        open3d::io::WritePointCloud(filePath, pcd);
    }
    void write_triangle_mesh(
            const std::string& filePath,
			const std::vector<double>&	vertices,
			const std::vector<uint32_t>&	indices
		)
    {
        std::vector<Eigen::Vector3d> o3d_vertices(vertices.size() / 3);
        std::vector<Eigen::Vector3i> triangles(indices.size() / 3);

        memcpy(o3d_vertices.data(), vertices.data(), vertices.size() * sizeof(double));
        memcpy(triangles.data(), indices.data(), indices.size() * sizeof(uint32_t));
        open3d::geometry::TriangleMesh mesh(o3d_vertices,triangles);
        mesh.ComputeVertexNormals();
        open3d::io::WriteTriangleMesh(filePath,mesh);
    }

}