//
// Created by Jason Mohoney on 2/28/20.
//

#include "pipeline/trainer.h"

#include "reporting/logger.h"


using std::get;
using std::tie;

PipelineTrainer::PipelineTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch) {
    dataloader_ = dataloader;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, true, progress_reporter_, pipeline_config);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, true, progress_reporter_, pipeline_config);
    }
}

void PipelineTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->initializeBatches(false);

    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        pipeline_->start();
        pipeline_->waitComplete();
        pipeline_->pauseAndFlush();
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);

        if (pipeline_->model_->device_models_.size() > 1) {
            pipeline_->model_->all_reduce();
        }

        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
}

SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->initializeBatches(false);

    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {
            // gets data and parameters for the next batch
            shared_ptr<Batch> batch = dataloader_->getBatch();
            if (dataloader_->epochs_processed_ == 0 && dataloader_->batches_processed_ == 0) 
            {
                SPDLOG_INFO("=======Initializing the embeddings======");
                std::ifstream in("outDegrees.txt");
                std::unordered_map<int, int> outDegreeMap;
                for (std::string nodeIdx, outDegree;
                     std::getline(in, nodeIdx, ' ') && std::getline(in, outDegree);)
                    {
                        outDegreeMap[std::stoi(nodeIdx)] = std::stoi(outDegree);
                    }
                batch->node_gradients_ = torch::zeros(batch->node_embeddings_.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
                auto embeddingAccess = batch->node_embeddings_.accessor<float,2>();
                //auto gradAccess = batch->node_gradients_.accessor<float,2>();
                auto gradientAccess = batch->node_gradients_.accessor<float, 2>();
                float init_value = 1 / batch->node_embeddings_.size(0);
                for (int i = 0; i < batch->node_embeddings_.size(0); i++) 
                {
                    gradientAccess[i][1] = -embeddingAccess[i][1];
                    gradientAccess[i][0] = init_value - embeddingAccess[i][0];
                    auto tmpValue = embeddingAccess[i][2];
                    float outDegree;
                    if (outDegreeMap.find(i) == outDegreeMap.end()) 
                    {
                        outDegree = 1;
                    }
                    else 
                    {
                        outDegree = outDegreeMap[i];
                    }
                    gradientAccess[i][2] = outDegree - tmpValue;
                }
                if (dataloader_->graph_storage_->embeddingsOffDevice()) 
                {
                    batch->embeddingsToHost();
                }
                dataloader_->updateEmbeddings(batch, false);              
            }
            /**
            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                // transfers batch to the GPU
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }
            */

           // dataloader_->loadCPUParameters(batch);
            /**
            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }
            */

            // batch->dense_graph_.performMap();  
            // Need this function to let things work, maybe
            // batch->embeddingsToHost();
            // Directly start updates.

            torch::Tensor src; 
            torch::Tensor dst;
            if (batch->node_embeddings_.defined()) 
            {
                batch->node_gradients_ = torch::zeros(batch->node_embeddings_.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
                src = batch->edges_.select(1, 0);
                dst = batch->edges_.select(1, -1);
            
                // Need to do the right updates on this batch.
                // We directly initialize the gradients here. 
                long sizeOfBatch = src.size(0);
//                SPDLOG_INFO("Embedding Dim 0 is {}", batch->node_embeddings_.size(0));
//                SPDLOG_INFO("Embedding Dim 1 is {}", batch->node_embeddings_.size(-1));

                // Create the gradient matrix, needed??
                // batch->node_gradients_ = torch::zeros(node_embeddings_.sizes());
                // Use efficient accessor:
                auto embeddingAccess = batch->node_embeddings_.accessor<float,2>();
                //auto gradAccess = batch->node_gradients_.accessor<float,2>();
                auto srcAccess = src.accessor<long, 1>();
                auto dstAccess = dst.accessor<long, 1>();
                auto gradientAccess = batch->node_gradients_.accessor<float, 2>();
                // compute the updates for pagerank
                for (long i = 0; i < sizeOfBatch; i++) 
                {
                    auto tmpGradient = gradientAccess[dstAccess[i]][1];
                    tmpGradient += embeddingAccess[srcAccess[i]][0] / (embeddingAccess[srcAccess[i]][2]);
                    gradientAccess[dstAccess[i]][1] = tmpGradient;
                }
                // gradientAccess[0][1] += 0.00005;
                // SPDLOG_INFO("Test Gradient #1 {}", gradientAccess[0][1]);
                SPDLOG_INFO("Test Embedding {}", embeddingAccess[0][1]);
                // SPDLOG_INFO("Unique Indices dim {}", batch->unique_node_indices_.size(0));
                // SPDLOG_INFO("Gradient dim {}", batch->node_gradients_ .size(0));
                // Maybe something is wrong with updateEmbeddings?
                // dataloader_->updateEmbeddings(batch, false);
                // batch->node_gradients_ = torch::empty(1);
            }
            // Checkout the potential error:
/*             if (dataloader_->graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) 
            {
               SPDLOG_INFO("Hot spot 1 passed"); 
            }
            else 
            {
               SPDLOG_INFO("Hot spot 1 failed"); 
            }
            if (batch->unique_node_indices_.size(0) == batch->node_gradients_.size(0)) 
            {
               SPDLOG_INFO("Hot spot 2 passed"); 
            }
            else 
            {
                SPDLOG_INFO("Hot spot 2 failed, indices {}", batch->unique_node_indices_.size(0));
                SPDLOG_INFO("Hot spot 2 failed, gradients {}", batch->node_gradients_.size(0));  
            }
            if (batch->unique_node_indices_.sizes().size() == 1) 
            {
               SPDLOG_INFO("Hot spot 3 passed"); 
            }
            else 
            {
               SPDLOG_INFO("Hot spot 3 failed, indices {}", batch->unique_node_indices_.sizes().size()); 
            }
            if (batch->node_gradients_.defined()) 
            {
               SPDLOG_INFO("Hot spot 4 passed"); 
            }
            else 
            {
               SPDLOG_INFO("Hot spot 4 failed with no defined gradients");  
            }
            if (dataloader_->graph_storage_->storage_ptrs_.node_embeddings->data_.size(1) == batch->node_gradients_.size(1)) 
            {
                SPDLOG_INFO("Hot spot 5 passed"); 
            }
            else 
            {
                SPDLOG_INFO("Hot spot 5 failed, buffer size {}", dataloader_->graph_storage_->storage_ptrs_.node_embeddings->data_.size(1)); 
                SPDLOG_INFO("Hot spot 5 failed, gradient size {}", batch->node_gradients_.size(1));
            } */
            // modify to be pr
            // model_->train_batch(batch);
            // model_->train_pr(batch);
            if (batch->node_embeddings_.defined()) 
            {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) 
                {
                    batch->embeddingsToHost();
                }
                dataloader_->updateEmbeddings(batch, false);
            } 
            // We need to do a final update, if at the end of one epoch.
            if (dataloader_->batches_left_ == 1) 
            {
                SPDLOG_INFO("========Reaches the end of an epoch, performs importance swapping.");
                batch->node_gradients_ = torch::zeros(batch->node_embeddings_.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
                auto gradientAccess = batch->node_gradients_.accessor<float, 2>();
                auto embeddingAccess = batch->node_embeddings_.accessor<float,2>();
                for (int i = 0; i < batch->node_embeddings_.size(0); i++) 
                {
                    auto currentImportance = embeddingAccess[i][1];
                    auto prevImportance = embeddingAccess[i][0];
                    gradientAccess[i][0] = currentImportance * 0.85 + 0.15 - prevImportance;
                    // Restore the embedding 
                    gradientAccess[i][1] = -currentImportance; 
                }
                if (dataloader_->graph_storage_->embeddingsOffDevice()) 
                {
                    batch->embeddingsToHost();
                }
                dataloader_->updateEmbeddings(batch, false);
            }
            // transfer gradients and update parameters
            /** Do nothing now
            if (batch->node_embeddings_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }

                dataloader_->updateEmbeddings(batch, false);
            }
            */
            // In the case of PageRank, we gotta update the embeddings once more :)
            /**
            if (batch->node_embeddings_.defined() && dataloader_->batches_left_ == 1) 
            {
                batch->node_gradients_ = torch::zeros(batch->node_embeddings_.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
                auto gradientAccess = batch->node_embeddings_.accessor<float,2>();
                auto gradientAccess = batch->node_embeddings_.accessor<float,2>();
                for (long i = 0; i < batch->node_embeddings_.size(0); i++) 
                {
                    
                    embeddingAccess[i][1] *= 0.85;
                    embeddingAccess[i][1] += 0.15;
                    embeddingAccess[i][0] = embeddingAccess[i][1];
                    embeddingAccess[i][1] = 0;
                }
            }
            */
            // Clear batch here, not immediately after updating!
            batch->clear();

            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);
        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        
        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
}
