//
// Created by Jason Mohoney on 7/9/20.
//

#include "data/batch.h"

#include "configuration/constants.h"
#include "reporting/logger.h"

using std::get;

Batch::Batch(bool train) : device_transfer_(0), host_transfer_(0), timer_(false) {
    status_ = BatchStatus::Waiting;
    train_ = train;
    device_id_ = -1;
    clear();
}

Batch::~Batch() { clear(); }

void Batch::to(torch::Device device) {
    device_id_ = device.index();

    if (device.is_cuda()) {
        device_transfer_ = CudaEvent(device_id_);
        host_transfer_ = CudaEvent(device_id_);
    }

    if (edges_.defined()) {
        edges_ = edges_.to(device);
    }

    if (neg_edges_.defined()) {
        neg_edges_ = neg_edges_.to(device);
    }

    if (root_node_indices_.defined()) {
        root_node_indices_ = root_node_indices_.to(device);
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(device);
    }

    if (node_labels_.defined()) {
        node_labels_ = node_labels_.to(device);
    }

    if (src_neg_indices_mapping_.defined()) {
        src_neg_indices_mapping_ = src_neg_indices_mapping_.to(device);
    }

    if (dst_neg_indices_mapping_.defined()) {
        dst_neg_indices_mapping_ = dst_neg_indices_mapping_.to(device);
    }

    if (src_neg_filter_.defined()) {
        src_neg_filter_ = src_neg_filter_.to(device);
    }

    if (dst_neg_filter_.defined()) {
        dst_neg_filter_ = dst_neg_filter_.to(device);
    }

    if (node_embeddings_.defined()) {
        // Try to initialize the gradients at here. 
        //SPDLOG_INFO("What the Hack?????????/");
        //node_gradients_ = torch::zeros(node_embeddings_.sizes());
        //node_gradients_ = node_gradients_.to(device);
        node_embeddings_ = node_embeddings_.to(device);
    }

    if (node_embeddings_state_.defined()) {
        node_embeddings_state_ = node_embeddings_state_.to(device);
    }

    if (node_features_.defined()) {
        node_features_ = node_features_.to(device);
    }

    if (encoded_uniques_.defined()) {
        encoded_uniques_ = encoded_uniques_.to(device);
    }

    if (dense_graph_.node_ids_.defined()) {
        dense_graph_.to(device);
    }

    if (device.is_cuda()) {
        device_transfer_.record();
    }

    status_ = BatchStatus::TransferredToDevice;
}

void Batch::accumulateGradients(float learning_rate) {
    if (node_embeddings_.defined()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        node_gradients_ = torch::zeros(node_embeddings_.sizes(), grad_opts);
        // How to modify the gradient signals is the problem
        for (int i = 0; i < edges_.sizes()[0]; i++) 
        {
            // Get the global node index
            int sourceNode = edges_[i][0].item<int>();
            int endNode = edges_[i][2].item<int>();
            // Need a way to map them to the correct indices
            Indices correctIDs = dense_graph_.getNodeIDs();
            // Assume that in the embeddings: 
            // Column 0: current importance,
            // Column 1: New importance, 
            // Column 2: Out degree of one node.
            // node_gradients_[sourceNode][1] += 1e-9;
            // node_gradients_[endNode][1] += 1e-10;
            // node_gradients_[endNode][1] += node_embeddings_[sourceNode][0].item<float>() / (node_embeddings_[sourceNode][2].item<float>() + 1e-3);          
        }
        SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);
        // node_state_update_ = node_gradients_.pow(2);
        // node_embeddings_state_.add_(node_state_update_);
        // node_gradients_ = -learning_rate * (node_gradients_ / (node_embeddings_state_.sqrt().add_(1e-10)));

        SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);
    }

    node_embeddings_state_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;
}
// In this case, we want to modify the gradient signal
// in a way that it do the PageRank updates.
/**
void Batch::accumulateGradients(float learning_rate) {
    if (node_embeddings_.defined()) {
        // Now we have node_embeddings_on GPU, edges on GPU.
        node_gradients_ = node_embeddings_;
        for (int i = 0; i < edges_.sizes()[0]; i++) 
        {
            for (int j = 0; j < 3; j++) 
            {
                int sourceNode = edges_[i][0];
                int endNode = edges_[i][2];

            }
        }
        // SPDLOG_INFO("gradientSize Dim {}", node_embeddings_.sizes()[0]);
        // SPDLOG_INFO("gradientSize Dim {}", node_embeddings_.sizes()[1]);
        SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);
        // Node embedding: (Previous Importance, current increment?, out_degree)
        node_state_update_ = node_gradients_.pow(1);
        node_embeddings_state_.add_(node_state_update_);
        // Add node_gradient here
        node_gradients_ = -node_gradients_;
        SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);
    }

    node_embeddings_state_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;
}
*/

void Batch::embeddingsToHost() {
    if (node_gradients_.defined() && node_gradients_.device().is_cuda()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(node_gradients_, true);
        Gradients temp_grads2 = torch::empty(node_state_update_.sizes(), grad_opts);
        temp_grads2.copy_(node_state_update_, true);
        node_gradients_ = temp_grads;
        node_state_update_ = temp_grads2;
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(torch::kCPU);
    }

    if (encoded_uniques_.defined()) {
        encoded_uniques_ = encoded_uniques_.to(torch::kCPU);
    }

    host_transfer_.record();
    host_transfer_.synchronize();
    status_ = BatchStatus::TransferredToHost;
}

void Batch::clear() {
    root_node_indices_ = torch::Tensor();
    unique_node_indices_ = torch::Tensor();
    node_embeddings_ = torch::Tensor();
    node_gradients_ = torch::Tensor();
    node_state_update_ = torch::Tensor();
    node_embeddings_state_ = torch::Tensor();

    node_features_ = torch::Tensor();
    node_labels_ = torch::Tensor();

    src_neg_indices_mapping_ = torch::Tensor();
    dst_neg_indices_mapping_ = torch::Tensor();

    edges_ = torch::Tensor();
    neg_edges_ = torch::Tensor();
    src_neg_indices_ = torch::Tensor();
    dst_neg_indices_ = torch::Tensor();

    dense_graph_.clear();
    encoded_uniques_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();
}