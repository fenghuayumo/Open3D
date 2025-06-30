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
#include "open3d/Open3D.h"
#include "mesh_utils.hpp"

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

void TSDF::export_mesh(
    const std::string& filePath,
    const std::vector<double*>& viewMatrixs,
    const std::vector<CameraIntrincs> intrinsics,
    const std::vector<uint8_t*> rgbs,
    const std::vector<float*>& depths,
    float voxel_length,
    int target_triangles,
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
        }
        auto mesh = volume.ExtractTriangleMesh();
        mesh->Scale(1,{0,0,0});
        if (threshold_ntri > 0) {
            //open3d::utility::VerbosityContextManager
           auto [triangle_clusters, cluster_n_triangles, cluster_area] = mesh->ClusterConnectedTriangles();
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

           clean_mesh.RemoveDegenerateTriangles();
           clean_mesh.RemoveDuplicatedVertices();
           clean_mesh.RemoveDuplicatedTriangles();
           clean_mesh.RemoveNonManifoldEdges();

           auto smooth_mesh =
                   clean_mesh.FilterSmoothTaubin(5);
           smooth_mesh = smooth_mesh->SimplifyQuadricDecimation(
                   target_triangles, std::numeric_limits<double>::infinity(),1.0f);
   
           smooth_mesh->ComputeVertexNormals();
           smooth_mesh->OrientTriangles();
           if (std::filesystem::path(filePath).extension() == ".obj") {
                //compute uv
           }
           open3d::io::WriteTriangleMesh(filePath, *smooth_mesh);
           return;
        }
        mesh->ComputeVertexNormals();
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
       /* for (auto i = 0; i < num; i++) {
            auto x = xyzs[i*3];
            auto y = xyzs[i*3+1];
            auto z = xyzs[i*3+2];
            if (std::isinf(x) || std::isinf(y) || std::isinf(z) || std::isnan(x) || std::isnan(y) || std::isnan(z))
                continue;
            {
                valid_xyzs.push_back(x);
                valid_xyzs.push_back(y);
                valid_xyzs.push_back(z);

                auto r = rgbs[i * 3];
                auto g = rgbs[i * 3 + 1];
                auto b = rgbs[i * 3 + 2];
                valid_rgbs.push_back(r);
                valid_rgbs.push_back(g);
                valid_rgbs.push_back(b);
            }
        }
        num = valid_xyzs.size() / 3;*/
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
        /* for (auto i = 0; i < num; i++) {
             auto x = xyzs[i*3];
             auto y = xyzs[i*3+1];
             auto z = xyzs[i*3+2];
             if (std::isinf(x) || std::isinf(y) || std::isinf(z) ||
         std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
             {
                 valid_xyzs.push_back(x);
                 valid_xyzs.push_back(y);
                 valid_xyzs.push_back(z);

                 auto r = rgbs[i * 3];
                 auto g = rgbs[i * 3 + 1];
                 auto b = rgbs[i * 3 + 2];
                 valid_rgbs.push_back(r);
                 valid_rgbs.push_back(g);
                 valid_rgbs.push_back(b);
             }
         }
         num = valid_xyzs.size() / 3;*/
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
            if (sparse_pcd->points_.size() >= 500000) {
                //get the bounding box of the point cloud
                auto bdbox =
                        sparse_pcd->GetAxisAlignedBoundingBox();
                auto diagonal_length = bdbox.GetExtent().norm();
                sparse_pcd = sparse_pcd->VoxelDownSample(diagonal_length * 0.002);
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